import cv2
import time
import yaml
import glob
import gzip
import time 
import torch

import logging
import os, sys
import imageio
import requests
import argparse
import datetime
import torchaudio

from GPT_SoVits.GPT_SoVITS.utils import * 
# import utils

import numpy as np
import onnxruntime
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
import torch.nn.functional as F
from EAT_model import EAT_infer


from skimage.transform import resize
from scipy.spatial import ConvexHull
from skimage import io, img_as_float32
from modelscope.utils.logger import get_logger
from GPT_SoVits.synthesize_audio import GPT_SoVITS_inference
from sync_batchnorm import DataParallelWithCallback  
from modules.generator import OcclusionAwareSPADEGeneratorEam
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.prompt import EmotionDeepPrompt, EmotionalDeformationTransformer
from modules.model_transformer import get_rotation_matrix, keypoint_transformation
from modules.transformer import Audio2kpTransformerBBoxQDeepPrompt as Audio2kpTransformer


logger = get_logger()
logger.setLevel(logging.ERROR)

"""
对于这整个流程需要做一个流程图

1. 对于前台传过来的音频，后台经过ASR，将音频转为文字
2. 大模型对输入的文本进行分析和处理，输出文字
3. 对于输出的文字进行TTS转换，将文字转为视频，后期如果时间允许，可以接入GPT-SoVits等模块对语音进行克隆
4. 将音频输入到EAT模块中，生成对应的emotional的talking head 
5. 将Talking head的视频传入到前台进行展示播放
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_video(predicted_frame_list, audio_path, save_path):
    audio_path_basename = os.path.basename(audio_path)[:-4]
    save_video = os.path.join(save_path, audio_path_basename + ".mp4")
    imageio.mimsave(save_path, predicted_frame_list, fps=25.0)
    cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)

class metahuman():
    def __init__(self, audio_path, wav_path, img_path, emotype) -> None:
        model_start_time = time.time()
        from modelscope.pipelines import pipeline as asr_pipeline
        from modelscope.utils.constant import Tasks
        
        self.onnx_path     = "/mnt/sdb/cxh/liwen/EAT_code/restoration/GFPGANv1.4.onnx"
        self.EAT_ckpt      = "/mnt/sdb/cxh/liwen/EAT_code/ckpt/deepprompt_eam3d_all_final_313.pth.tar"
        
        self.audio_in_path = audio_path     
        self.wav_path      = wav_path            
        self.img_path      = img_path
        
        start_time = time.time()
        self.asr_pipeline   = asr_pipeline(
                            task=Tasks.auto_speech_recognition,
                            model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                            model_revision="v2.0.4"
                        )
        end_time            = time.time()
        print(f"<=================== ASR end init ===================>")
        print(f"{end_time -start_time} \n")
        
        
        # from TTS.synthesize_all import SpeechSynthesis
        # self.tts_pipeline   = SpeechSynthesis('./TTS/config/AISHELL3')
        start_time              = time.time()
        self.gpt_model_path     = "/mnt/sdb/cxh/liwen/GPT-SoVITS/GPT_weights/liwen-e25.ckpt"
        self.sovits_model_path  = "/mnt/sdb/cxh/liwen/GPT-SoVITS/SoVITS_weights/liwen_e25_s125.pth"
        self.ref_audio_path     = "/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/reference_audio/reference_ZH.wav"
        
        self.prompt_text        = "人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        self.text_language      = "中文"
        self.gpt_infer_model    = GPT_SoVITS_inference(self.gpt_model_path, self.sovits_model_path, self.ref_audio_path, self.prompt_text)
        self.sample_rate        = 32000
        
        
        end_time                = time.time()
        print(f"<=================== GPT_SoVITS_inference end init ===================>")
        print(f"{end_time -start_time} \n")
        

        self.emotype     = emotype
        self.config_path = "/mnt/sdb/cxh/liwen/EAT_code/config/deepprompt_eam3d_st_tanh_304_3090_all.yaml"
        self.config      = self.load_yaml(self.config_path)

        # load extractor related parameters
        self.config_path_extractor = 'config/vox-256-spade.yaml'
        self.extractor_ckpt        = './ckpt/pretrain_new_274.pth.tar'
        self.config_extractor      = self.load_yaml(self.config_path_extractor)
        
        # load extractor checkpoint
        self.extractor = self.load_checkpoints_extractor(self.config_extractor, self.extractor_ckpt)
        self.generator, self.kp_detector, self.audio2kptransformer, self.sidetuning, self.emotionprompt = self.build_EAT_model(self.config) 
        self.load_ckpt_for_EAT_model(self.EAT_ckpt, 
                                     self.kp_detector, 
                                     self.generator, 
                                     self.audio2kptransformer, 
                                     self.sidetuning, 
                                     self.emotionprompt)
        
        # set model state eval 
        self.audio2kptransformer.eval()
        self.generator.eval()
        self.kp_detector.eval()
        self.sidetuning.eval()
        self.emotionprompt.eval()

        # add a model list
        self.model_list = ( self.generator, 
                            self.kp_detector,
                            self.audio2kptransformer, 
                            self.sidetuning, 
                            self.emotionprompt)
        
        self._tensorrt_init(self.onnx_path)
        model_end_time = time.time()
        print("<=================== Model init end ===================>")
        print(f"{model_end_time - model_start_time} \n")

    """ load config file """
    def load_yaml(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    """ build 5 EAT model """
    def build_EAT_model(self, config, device_ids=[0]):
        start_time = time.time()
        generator = OcclusionAwareSPADEGeneratorEam(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        if torch.cuda.is_available():
            print('cuda is available')
            generator.to(device_ids[0])
            
        kp_detector         = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            kp_detector.to(device_ids[0])
            
        audio2kptransformer = Audio2kpTransformer(**config['model_params']['audio2kp_params'], face_ea=True)
        if torch.cuda.is_available():
            audio2kptransformer.to(device_ids[0])
            
        sidetuning          = EmotionalDeformationTransformer(**config['model_params']['audio2kp_params'])
        if torch.cuda.is_available():
            sidetuning.to(device_ids[0])
            
        emotionprompt       = EmotionDeepPrompt()
        if torch.cuda.is_available():
            emotionprompt.to(device_ids[0])
        end_time = time.time()
        print(f"<==================== End building EAT Models ====================>")
        print(f"Cost {end_time - start_time} \n")
        return generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt
    
    
    """ load ckpt for 5 mian model """
    def load_ckpt_for_EAT_model(self, ckpt, kp_detector, generator, audio2kptransformer, sidetuning, emotionprompt):
        start_time = time.time()
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        if audio2kptransformer is not None:
            audio2kptransformer.load_state_dict(checkpoint['audio2kptransformer'])
            
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
            
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
            
        if sidetuning is not None:
            sidetuning.load_state_dict(checkpoint['sidetuning'])
            
        if emotionprompt is not None:
            emotionprompt.load_state_dict(checkpoint['emotionprompt'])
        end_time = time.time()
        print(f"<==================== End loading checkpoint for EAT Models ====================>")
        print(f"Cost {end_time - start_time} \n")
    
    
    """ Init extractor and load ckpt for extractor """
    def load_checkpoints_extractor(self, config, checkpoint_path, cpu=False):
        start_time = time.time()
        kp_detector     = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()
        he_estimator    = HEEstimator(**config['model_params']['he_estimator_params'],
                                **config['model_params']['common_params'])
        if not cpu:
            he_estimator.cuda()
        if cpu:
            checkpoint  = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint  = torch.load(checkpoint_path)

        kp_detector.load_state_dict(checkpoint['kp_detector'])
        he_estimator.load_state_dict(checkpoint['he_estimator'])
        
        if not cpu:
            kp_detector  = DataParallelWithCallback(kp_detector)
            he_estimator = DataParallelWithCallback(he_estimator)
            
        kp_detector.eval()
        he_estimator.eval()

        end_time = time.time()
        print(f"<==================== End loading checkpoint for Extractor Models ====================>")
        print(f"Cost {end_time - start_time} \n")
        return kp_detector, he_estimator


    """" Tensorrt session init """
    def _tensorrt_init(self, onnx_path):
        start_time = time.time()
        self.session = onnxruntime.InferenceSession(onnx_path, 
                        providers=["TensorrtExecutionProvider","CUDAExecutionProvider"])
        self.io_binding = self.session.io_binding()
        end_time = time.time()
        print(f"<==================== TensorRT backend build successful! ====================>")
        print(f"Cost {end_time - start_time} \n")
        
        
    """" asr--> LLM --> TTS => audio.wav """
    def generate_audio(self):
        start_time       = time.time()
        asr_result_map   = self.asr_pipeline(self.audio_in_path)
        asr_result       = asr_result_map[0]['text']
        
        data             = {
                            "prompt": asr_result + ", \
                            请使用中文进行回复，并且单次回复不要超过80个字", "history": []
                        }
        response         = requests.post("http://0.0.0.0:8000",  
                                    json=data,
                                    headers={"Content-Type": "application/json"})
        chat_result_dict = eval(response.text)             
        chat_input       = chat_result_dict["response"]
        
        abs_wavpath      = os.path.join(self.wav_path, "temp.wav")
        audio            = self.gpt_infer_model.get_tts_wav(chat_input, self.text_language)
        sf.write(abs_wavpath, audio, self.sample_rate, 'PCM_24')
        
        end_time         = time.time()
        print(f"<==================== Interactive model end====================>")
        print(f"Cost {end_time - start_time} \n")
        return abs_wavpath
        
    """ Use GFPGAN restorate the frame"""
    def frame_restoration(self, frame_list): # B H W C RGB
        start_time = time.time()
        img_list = []
        result_list = []
        for index in range(frame_list.shape[0]):
            img_tensor = frame_list[index]
            # print(img_tensor.shape)
            resized_img = F.interpolate(img_tensor.unsqueeze(0), size=(512,512), mode='bicubic', align_corners=False)
            resized_normal_img = 2*(resized_img  - 0.5)  # -1 ~ 1
            img_list.append(resized_normal_img)
        
        input_buff     = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous() # 判断Tensor按行展开后的顺序与其storage的顺序是否一致
        output_buff    = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
        
        for i in tqdm(range(len(img_list))):
            input_tensor = img_list[i].cuda()
            input_buff   = input_tensor
            self.io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=input_buff.data_ptr())
            self.io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output_buff.data_ptr())
            self.session.run_with_iobinding(self.io_binding)
            outpred1 = torch.squeeze(output_buff)    # (3, 512, 512)
            outpred1 = torch.clamp(outpred1, -1, 1)  # 限制 -1 ~ 1
            outpred1 = torch.add(outpred1, 1)        # 0 - 1
            outpred1 = torch.div(outpred1, 2)        # 0 - 0.5
            outpred1 = torch.mul(outpred1, 255)[[2,1,0],:,:].permute(1,2,0).cpu().numpy()
            result_list.append(outpred1.astype(np.uint8))
        end_time = time.time()
        
        print(f"==================== End GFPGAN restorate frames ====================")
        print(f"Cost {end_time - start_time}")
        return result_list
    
    def write_images_restoration(self, result_list, res_path="/mnt/sdb/cxh/liwen/EAT_code/demo/test/restoration_result"):
        for index in range(len(result_list)):
            # result_list[index] = cv2.cvtColor(result_list[index], cv2.COLOR_RGB2BGR)
            cv2.imwrite("%s/%d_results.png"%(res_path, index), result_list[index])
    
    # BCHW
    def save_image_from_eat(self, image_tensor_list, res_path="/mnt/sdb/cxh/liwen/EAT_code/demo/test_gfpgan"):
        img_list = []
        for i in range(image_tensor_list.shape[0]):
            img_tensor = image_tensor_list[i]*255
            img_numpy = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_bgr = img_numpy[:, :, ::-1]
            img_list.append(img_bgr)
        
        for index in range(len(img_list)):
            cv2.imwrite("%s/%d_results.png"%(res_path, index),img_list[index])     
            
    def concat_video_RGB(self, frame_list, audio_path, save_video_path):
        now = datetime.datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        fname      = formatted_now + ".mp4"
        temp_dir   = "/mnt/sdb/cxh/liwen/EAT_code/save_videos/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        video_path = os.path.join(temp_dir, fname)
        
        imageio.mimsave(video_path, frame_list, fps=25.0)

        save_video = os.path.join(save_video_path, fname)
        cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
        os.system(cmd)
        os.remove(video_path)
    
    def concat_video_BGR(frames, save_video_path, fps=25):
        now = datetime.datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        fname      = formatted_now + ".mp4"
        
        save_video = os.path.join(save_video_path, fname)
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
    @torch.no_grad()
    def __call__(self):
        start_time = time.time()
        save_video_path = "/mnt/sdb/cxh/liwen/EAT_code/save_videos"
        start_time  = time.time()
        # abs_wavpath = self.generate_audio()
        abs_wavpath = "/mnt/sdb/cxh/liwen/EAT_code/demo/video_processed/M030_ang_3_004/deepfeature32/M030_ang_3_004.npy"
        infer_model = EAT_infer(abs_wavpath, self.img_path, self.emotype, self.config_path, self.extractor)
        frame_list  = infer_model(self.model_list)   # B H W C RGB
        # self.save_image_from_eat(frame_list)
        result_list = self.frame_restoration(frame_list)
        # self.write_images_restoration(result_list)
        
        self.concat_video(result_list, abs_wavpath, save_video_path)
        print("frame counts", str(len(result_list)) + "\n")
        end_time = time.time()
        print(f"<==================== Pipeline finished ====================>")
        print(f"Cost {end_time - start_time} \n")
        return result_list
        
    def test_gfpgan(self, image_dir="", out_dir=""):
        res_path = "/mnt/sdb/cxh/liwen/EAT_code/demo/test_gfpgan"
        images_path   = glob.glob("*.png")
        image_np_list = []
        for image in images_path:
            image_np  = cv2.imread(image)
            image_np  = image_np[:,:,::-1] 
        image_np_list.append(image_np)
        # 0-255 BGR 
        result_list = self.frame_restoration(image_np_list)
        self.write_images_restoration(result_list, res_path)
        
            
if __name__=="__main__":
    emotype = "sad"
    audio_path = "/mnt/sdb/cxh/liwen/EAT_code/demo/video_processed/M030_ang_3_004/M030_ang_3_004.wav"
    
    wav_path = "/mnt/sdb/cxh/liwen/EAT_code/audio_temp"
    img_path = "/mnt/sdb/cxh/liwen/EAT_code/demo/imgs/out.jpg"
    
    start_time = time.time()
    model = metahuman(audio_path, wav_path, img_path, emotype)
    end_time = time.time()
    print(f"Metahuman init total cost {end_time - start_time} \n")
    
    start_time = time.time()
    model()
    end_time = time.time()
    print(f"Inference Cost Time {end_time - start_time}")
    

