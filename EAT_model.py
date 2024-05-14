import os
import cv2
import dlib
import glob
import yaml
import gzip
import torch
import time

import imageio
import argparse
import subprocess
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
import tensorflow as tef
from scipy.io import wavfile

import torch.nn.functional as F
from scipy.signal import resample
from skimage import transform as tf
from skimage.transform import resize
from scipy.spatial import ConvexHull
from skimage import io, img_as_float32
from extract_ds_features import return_deepfeature
from modules.generator import OcclusionAwareSPADEGeneratorEam
from modules.keypoint_detector import KPDetector, HEEstimator

from modules.prompt import EmotionDeepPrompt, EmotionalDeformationTransformer
from modules.model_transformer import get_rotation_matrix, keypoint_transformation
from modules.transformer import Audio2kpTransformerBBoxQDeepPrompt as Audio2kpTransformer

class EAT_infer():
    def __init__(self, root_wav, img, emotype, prepare_data_config, extractor, pose_path) -> None:
        
        # User info
        self.root_wav = root_wav
        
        # self.root_wav = "./demo/test_wav/output.wav"
        self.img      = img
        self.emotype  = emotype
        
        # audio parameters
        self.MEL_PARAMS_25  = {
                                "n_mels": 80,
                                "n_fft": 2048,
                                "win_length": 640,
                                "hop_length": 640}
        
        self.to_melspec     = torchaudio.transforms.MelSpectrogram(**self.MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.latent_dim     = 16
        

        self.expU           = torch.from_numpy(np.load('./expPCAnorm_fin/U_mead.npy')[:,:32])
        self.expmean        = torch.from_numpy(np.load('./expPCAnorm_fin/mean_mead.npy'))
        self.template       = np.load('./demo/M003_template.npy')

        # emotional label
        self.emo_label      = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
        self.emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
        
        # EAT model config 
        self.config         = prepare_data_config
        
        # Face detector and keypoints extractor
        self.detector       = dlib.get_frontal_face_detector()
        self.predictor      = dlib.shape_predictor('./demo/shape_predictor_68_face_landmarks.dat')
        self.extractor      = extractor
        
        # Crop the source images
        self.source_latent, self.crop_img = self.crop_and_extract_keypoints(self.img) # 获取源keypoint
        
        self.tf_config=tef.compat.v1.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True   # 动态申请机制
        self.tf_config.inter_op_parallelism_threads = 4  # 外部操作线程
        self.tf_config.intra_op_parallelism_threads = 4  # 内部操作线程
        
        self.pose_path       = pose_path
    # def load_config(self, config_path):
    #     with open(config_path) as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)
    #     return config

    """ normalized the extractor keypoint"""
    def _normalize_kp(self, kp_source, kp_driving, kp_driving_initial,
                 use_relative_movement=True, use_relative_jacobian=True):
        kp_new = {k: v for k, v in kp_driving.items()}
        if use_relative_movement:
            kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
            kp_new['value'] = kp_value_diff + kp_source['value']

            if use_relative_jacobian:
                jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
                kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
        return kp_new
    
    """ Load audio to numpy astype float"""
    # def _load_wav_tensor(self, wave_path):
    #     wave, sr = sf.read(wave_path)
    #     wave_tensor = torch.from_numpy(wave).float()
    #     return wave_tensor
    
    def _load_wav_tensor(self, wave_path, target_sr=16000):
        wave, sr = sf.read(wave_path)
        if sr != target_sr:
            wave = resample(wave, int(len(wave) * target_sr / sr))
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    """ Prepare the data with some preprocessing for EAT model"""
    def prepare_test_data(self, source_latent, opt, emotype):
        he_source        = {}
        # print(source_latent[1].keys())
        for k in source_latent[1].keys():
            he_source[k] = torch.from_numpy(source_latent[1][k][0]).unsqueeze(0).cuda()
        
        y_trg            = self.emo_label.index(emotype)
        z_trg            = torch.randn(self.latent_dim)
        
        deepfeature      = return_deepfeature(self.root_wav, self.tf_config) # N,16,29
        tef.compat.v1.reset_default_graph()
        tef.keras.backend.clear_session()
        
        
        # Debug  the fixed audio
        # deepfeature_npy  = "./demo/video_processed/obama/deepfeature32/obama.npy"
        # deepfeature      = np.load(deepfeature_npy)
        
        # end_time = time.time()
        # print(f"<==================== Extract deepspeech finished ====================>")
        # print(f"Cost {end_time - start_time}")
        
        pose_gz          = gzip.GzipFile(self.pose_path, "r")
        poseimg          = np.load(pose_gz)
        
        if poseimg.shape[0] < deepfeature.shape[0]:
            num_tiles = deepfeature.shape[0] // poseimg.shape[0] + 1  # 加一是为了确保长度至少为N
            poseimg = np.tile(poseimg, (num_tiles, 1, 1, 1))
            poseimg = poseimg[:deepfeature.shape[0]]
        
        driving_latent_path = "./preprocess/latents/mytemplate.npy"
        driving_latent      = np.load(driving_latent_path, allow_pickle=True)
        
        he_driving          = driving_latent[1]  
        valid_scope         = deepfeature.shape[0]
        
        for x in driving_latent[1].keys():
            driving_latent[1][x] = driving_latent[1][x][:valid_scope]
        num_frames = deepfeature.shape[0]  
        
        
        wave_tensor     = self._load_wav_tensor(self.root_wav)
        if len(wave_tensor.shape) > 1:
            wave_tensor = wave_tensor[:, 0]
            
        # Problem：RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (1024, 1024) at dimension 2 of input [1, 1, 1024]
        # if the asr result are totally english, the tts can't generate normal audio.wav file, causing the load failed.
        mel_tensor    = self.to_melspec(wave_tensor)
        mel_tensor    = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        name_len      = min(mel_tensor.shape[1], poseimg.shape[0], deepfeature.shape[0])

        audio_frames  = []
        poseimgs      = []
        deep_feature  = []
        pad, deep_pad = np.load('pad.npy', allow_pickle=True)

        # output length will depend on the shortest length of the audio and driven poses
        if name_len < num_frames:
            diff = num_frames - name_len
            if diff > 2:
                print(f"Attention: the frames are {diff} more than name_len, we will use name_len to replace num_frames")
                num_frames = name_len
                for k in he_driving.keys():
                    he_driving[k] = he_driving[k][:name_len, :]
                    
        for rid in range(0, num_frames):
            audio = []
            poses = []
            deeps = []
            for i in range(rid - opt['num_w'], rid + opt['num_w'] + 1):
                if i < 0:
                    audio.append(pad)
                    poses.append(poseimg[0])
                    deeps.append(deep_pad)
                elif i >= name_len:
                    audio.append(pad)
                    poses.append(poseimg[-1])
                    deeps.append(deep_pad)
                else:
                    audio.append(mel_tensor[:, i])
                    poses.append(poseimg[i])
                    deeps.append(deepfeature[i])

            audio_frames.append(torch.stack(audio, dim=1))
            poseimgs.append(poses)
            deep_feature.append(deeps)

        audio_frames    = torch.stack(audio_frames, dim=0)
        poseimgs        = torch.from_numpy(np.array(poseimgs))
        deep_feature    = torch.from_numpy(np.array(deep_feature)).to(torch.float)
        print("============================prepare end==================================")  # 标记一下这个地方的值
        return audio_frames, poseimgs, deep_feature, he_source, he_driving, num_frames, y_trg, z_trg

    def shape_to_np(self, shape, dtype="int"):
        coords        = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    # 估计latent code H W C RGB 256*256
    def estimate_latent(self, driving_video, kp_detector, he_estimator):
        with torch.no_grad():
            predictions         = []  
            driving             = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
            kp_canonical        = kp_detector(driving[:, :, 0])
            he_drivings         = {'yaw': [], 'pitch': [], 'roll': [], 't': [], 'exp': []}
            
            for frame_idx in range(driving.shape[2]):
                driving_frame = driving[:, :, frame_idx]
                he_driving    = he_estimator(driving_frame)
                for k in he_drivings.keys():
                    he_drivings[k].append(he_driving[k])
        return [kp_canonical, he_drivings]
    
    # # 1, 3, 256, 256  RBG 
    def save_img_tensor(self, img_tensor):
        img_nor = img_tensor.squeeze(0).permute(1, 2, 0)
        img_np  = img_nor.cpu().numpy()
        cv2.imwrite("./demo/mytest/out_result/test_pipeline.png", img_np)
    
    def crop_and_extract_keypoints(self, image):
        with torch.no_grad():
            kp_detector, he_estimator = self.extractor
            # Debug check the state of model 
            print("Modle State ", "training " if kp_detector.training else "eval")
            print("Modle State ", "training" if he_estimator.training else "eval")
            
            # image   = cv2.imread(image_path)
            # numpy RGB 
            image   = np.clip(image, 0, 255).astype(np.uint8)  # RGB
            # image_save  = image.copy()
            # cv2.imwrite('./demo/imgs/test_gradio_img.png', image[...,::-1])
            
            gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            rects   = self.detector(gray, 1)  # detect human face
            
            if len(rects) != 1:
                return 0
            
            for (j, rect) in enumerate(rects):
                shape   = self.predictor(gray, rect)  # detect 68 points
                shape   = self.shape_to_np(shape)

            pts2        = np.float32(self.template[:47,:])
            pts1        = np.float32(shape[:47,:])         # eye and nose
            tform       = tf.SimilarityTransform()
            tform.estimate(pts2, pts1) #Set the transformation matrix with the explicit parameters.
            
            dst = tf.warp(image, tform, output_shape=(256, 256))
            dst = np.array(dst * 255, dtype=np.uint8)  # BGR 0-255
            
            # print("dst", dst.shape) # (256, 256, 3)
            # driving_frames       = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)   
            driving_frames       = dst
            resize_driving_video = resize(driving_frames, (256, 256))[..., :3]
            driving_video        = [resize_driving_video]
            
            kc, he = self.estimate_latent(driving_video, kp_detector, he_estimator)
            kc     = kc['value'].cpu().numpy()
            for k in he:
                he[k] = torch.cat(he[k]).cpu().numpy()
                
            keypoint_arr_list = [kc, he]
            keypoint_arr      = np.asanyarray(keypoint_arr_list, dtype="object")
            
            return keypoint_arr, driving_frames
    
    @torch.no_grad()
    def __call__(self, model_list): 
        start_time = time.time()
        generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt                = model_list
        
        # Debugcheck the state of model 
        print("Modle State ", "training" if generator.training else "eval")
        # Debug check the state of model 
        print("Model State ", "training" if kp_detector.training else "eval")
        # Debug check the state of model 
        print("Model State ", "training" if audio2kptransformer.training else "eval")
        # Debug check the state of model 
        print("Model Satte ", "training " if sidetuning.training else "eval")
        # Debug check the state of model 
        print("Model State ", "training " if emotionprompt.training else "eavl")
        
        prepared_data = self.prepare_test_data(self.source_latent, self.config['model_params']['audio2kp_params'], self.emotype)
        audio_frames, poseimgs, deep_feature, he_source, he_driving, num_frames, y_trg, z_trg = prepared_data
        
        with torch.no_grad():
            # CHW  RGB
            cropped_img_np_rgb  = img_as_float32(self.crop_img).transpose(2, 0, 1).astype(np.float32)
            cropped_img         = torch.from_numpy(cropped_img_np_rgb).unsqueeze(0).cuda()  # 1, 3, 256, 256
            
            # check the input img 
            # self.save_img_tensor(cropped_img) 
            
            kp_canonical = kp_detector(cropped_img, with_feature=True)     # {'value': value, 'jacobian': jacobian}   
            kp_cano      = kp_canonical['value']

            x = {}
            x['mel']     = audio_frames.unsqueeze(1).unsqueeze(0).cuda()
            x['z_trg']   = z_trg.unsqueeze(0).cuda()
            x['y_trg']   = torch.tensor(y_trg, dtype=torch.long).cuda().reshape(1)
            x['pose']    = poseimgs.cuda()
            x['deep']    = deep_feature.cuda().unsqueeze(0)
            x['he_driving'] = { 'yaw': torch.from_numpy(he_driving['yaw']).cuda().unsqueeze(0), 
                                'pitch': torch.from_numpy(he_driving['pitch']).cuda().unsqueeze(0), 
                                'roll': torch.from_numpy(he_driving['roll']).cuda().unsqueeze(0), 
                                't': torch.from_numpy(he_driving['t']).cuda().unsqueeze(0),
                            }
            
            ### emotion prompt
            emoprompt, deepprompt = emotionprompt(x)
            
            a2kp_exps = []
            emo_exps  = []
            T = 5
            if T == 1:
                for i in range(x['mel'].shape[1]):
                    xi = {}
                    xi['mel']    = x['mel'][:, i, :, :, :].unsqueeze(1)
                    xi['z_trg']  = x['z_trg']
                    xi['y_trg']  = x['y_trg']
                    xi['pose']   = x['pose'][i,:,:,:,:].unsqueeze(0)
                    xi['deep']   = x['deep'][:,i,:,:,:].unsqueeze(1)
                    xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i,:].unsqueeze(0), 
                                'pitch': x['he_driving']['pitch'][:,i,:].unsqueeze(0), 
                                'roll': x['he_driving']['roll'][:,i,:].unsqueeze(0), 
                                't': x['he_driving']['t'][:,i,:].unsqueeze(0), 
                                }
                    he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                    emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                    a2kp_exps.append(he_driving_emo_xi['emo'])
                    emo_exps.append(emo_exp)
            elif T is not None:
                for i in range(x['mel'].shape[1]//T + 1):
                    if i*T >= x['mel'].shape[1]:
                        break
                    xi = {}
                    xi['mel']   = x['mel'][:,i*T:(i+1)*T,:,:,:]
                    xi['z_trg'] = x['z_trg']
                    xi['y_trg'] = x['y_trg']
                    xi['pose']  = x['pose'][i*T:(i+1)*T,:,:,:,:]
                    xi['deep']  = x['deep'][:,i*T:(i+1)*T,:,:,:]
                    xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i*T:(i+1)*T,:], 
                                'pitch': x['he_driving']['pitch'][:,i*T:(i+1)*T,:], 
                                'roll': x['he_driving']['roll'][:,i*T:(i+1)*T,:], 
                                't': x['he_driving']['t'][:,i*T:(i+1)*T,:], 
                                }

                    # print("-----------------------------------------------------")
                    # print("xi['mel']", xi['mel'].shape)                                     # torch.Size([1, 5, 1, 80, 11])
                    # print("xi['z_trg']",  xi['z_trg'].shape)                                # torch.Size([1, 16])
                    # print("xi['y_trg']",  xi['y_trg'].shape)                                # xi['y_trg'] torch.Size([1])
                    # print("xi['pose']", xi['pose'].shape)                                   # xi['pose'] torch.Size([5, 11, 1, 64, 64])
                    # print("xi['deep']",  xi['deep'].shape)                                  # xi['deep'] torch.Size([1, 5, 11, 16, 29])
                    
                    # print("xi['he_driving']['yaw']", xi['he_driving']['yaw'].shape)         # torch.Size([1, 5, 66])
                    # print("xi['he_driving']['pitch']", xi['he_driving']['pitch'].shape)     # torch.Size([1, 5, 66])
                    # print("xi['he_driving']['roll']", xi['he_driving']['roll'].shape)       # torch.Size([1, 5, 66])
                    # print("xi['he_driving']['t']", xi['he_driving']['t'].shape)             # torch.Size([1, 5, 3])
                    
                    # print("kp_canonical", kp_canonical['value'].shape)                      # torch.Size([1, 15, 3])
                    # print("-----------------------------------------------------")
                    
                    # shape '[240, 4096]' is invalid for input of size 2949120
                    he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                    emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                    a2kp_exps.append(he_driving_emo_xi['emo'])
                    emo_exps.append(emo_exp)
            
            if T is None:
                he_driving_emo, input_st = audio2kptransformer(x, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                emo_exps = sidetuning(input_st, emoprompt, deepprompt).reshape(-1, 45)
            else:
                he_driving_emo = {}
                he_driving_emo['emo'] = torch.cat(a2kp_exps, dim=0)
                emo_exps = torch.cat(emo_exps, dim=0).reshape(-1, 45)

            exp    = he_driving_emo['emo']
            device = exp.get_device()
            
            exp = torch.mm(exp, self.expU.t().to(device))
            exp = exp + self.expmean.expand_as(exp).to(device)
            exp = exp + emo_exps

            source_area = ConvexHull(kp_cano[0].cpu().numpy()).volume
            exp = exp * source_area

            he_new_driving = {'yaw': torch.from_numpy(he_driving['yaw']).cuda(), 
                            'pitch': torch.from_numpy(he_driving['pitch']).cuda(), 
                            'roll': torch.from_numpy(he_driving['roll']).cuda(), 
                            't': torch.from_numpy(he_driving['t']).cuda(), 
                            'exp': exp}
            he_driving['exp'] = torch.from_numpy(he_driving['exp']).cuda()

            kp_source = keypoint_transformation(kp_canonical, he_source, False)
            mean_source = torch.mean(kp_source['value'], dim=1)[0]
            kp_driving = keypoint_transformation(kp_canonical, he_new_driving, False)
            mean_driving = torch.mean(torch.mean(kp_driving['value'], dim=1), dim=0)
            kp_driving['value'] = kp_driving['value'] + (mean_source - mean_driving).unsqueeze(0).unsqueeze(0)

            bs = kp_source['value'].shape[0]
            predictions_gen = []
            for i in tqdm(range(num_frames)):
                kp_si = {}
                kp_di = {}
                kp_si['value'] = kp_source['value'][0].unsqueeze(0)
                kp_di['value'] = kp_driving['value'][i].unsqueeze(0)

                # generate frame for 
                generated = generator(cropped_img, kp_source=kp_si, kp_driving=kp_di, prompt=emoprompt)
                # predict_frame = (np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8) # RGB
                predict_frame = generated['prediction'] # B C H W
                predictions_gen.append(predict_frame)
                # print(predict_frame.shape)        # (256, 256, 3)
            cat_predict_frames =  torch.cat(predictions_gen, 0)
            
            end_time = time.time()
            print("<=================== EAT call End ===================>")
            print(cat_predict_frames.shape)
            print(f"{end_time - start_time} \n")

            del model_list
            return cat_predict_frames   # 0-255

if __name__=="__main__":
    root_wav = './demo/video_processed/M003_neu_1_001'
    save_dir = "./demo/test"
    eat_infer_model = EAT_infer(root_wav, save_dir)

    ckpt = "./ckpt/deepprompt_eam3d_all_final_313.pth.tar"
    emotype = "hap"
    save_out_dir = "./output"
    eat_infer_model(ckpt, emotype)