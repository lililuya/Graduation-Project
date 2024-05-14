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
import torchaudio

from GPT_SoVits.GPT_SoVITS.utils import * 
# import utils

import numpy as np
import onnxruntime
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
from datetime import datetime
import torch.nn.functional as F
from EAT_model import EAT_infer


from skimage.transform import resize
from scipy.spatial import ConvexHull
from skimage import io, img_as_float32
from modelscope.utils.logger import get_logger
from GPT_SoVits.synthesize_audio_EN import GPT_SoVITS_inference  # 选择双语模型
from sync_batchnorm import DataParallelWithCallback  
from modules.generator import OcclusionAwareSPADEGeneratorEam
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.prompt import EmotionDeepPrompt, EmotionalDeformationTransformer
from modules.model_transformer import get_rotation_matrix, keypoint_transformation
from modules.transformer import Audio2kpTransformerBBoxQDeepPrompt as Audio2kpTransformer


logger = get_logger()
logger.setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
对于这整个流程需要做一个流程图

1. 对于前台传过来的音频，后台经过ASR，将音频转为文字
2. 大模型对输入的文本进行分析和处理，输出文字
3. 对于输出的文字进行TTS转换，将文字转为视频，后期如果时间允许，可以接入GPT-SoVits等模块对语音进行克隆
4. 将音频输入到EAT模块中，生成对应的emotional的talking head 
5. 将Talking head的视频传入到前台进行展示播放
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def save_video(predicted_frame_list, audio_path, save_path):
    audio_path_basename = os.path.basename(audio_path)[:-4]
    save_video = os.path.join(save_path, audio_path_basename + ".mp4")
    imageio.mimsave(save_path, predicted_frame_list, fps=25.0)
    cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    
# 回收垃圾
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class metahuman():
    # audio_path, wav_path, img_path, emotype
    def __init__(self, 
                 my_config_file_path    = "./config/my_yaml_wholepipeline_gptsovits.yaml",
                 build_model_cofig_path = "./config/deepprompt_eam3d_st_tanh_304_3090_all.yaml",
                 extractor_config_path  = "./config/vox-256-spade.yaml"):
        model_start_time = time.time()
        from modelscope.pipelines import pipeline as asr_pipeline
        from modelscope.utils.constant import Tasks
        from funasr import AutoModel
        
        # print("my_config_file_path", my_config_file_path)
        # load config
        self.my_config_dict      = self.getConfigYaml(my_config_file_path)
        self.build_model_config  = self.getConfigYaml(build_model_cofig_path)
        self.config_extractor    = self.getConfigYaml(extractor_config_path)
        
        self.onnx_path     = self.my_config_dict["onnx_path"]
        self.EAT_ckpt      = self.my_config_dict["EAT_ckpt"]
        start_time = time.time()
        
        
        self.asr_pipeline_zh   = asr_pipeline(
                            task=Tasks.auto_speech_recognition,
                            model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                            model_revision="v2.0.4"
                        )
        
        self.asr_pipeline_en   = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                                           vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                                           punc_model="ct-punc-c", punc_model_revision="v2.0.4")
        
        end_time            = time.time()
        print(f"<=================== ASR end init ===================>")
        print(f"{end_time -start_time} \n")
        
        
        # from TTS.synthesize_all import SpeechSynthesis
        # self.tts_pipeline   = SpeechSynthesis('./TTS/config/AISHELL3')
        start_time              = time.time()
        self.sample_rate           = self.my_config_dict["sample_rate"]
        
        # en
        self.gpt_model_path_zh     = self.my_config_dict["gpt_model_path_zh"]
        self.sovits_model_path_zh  = self.my_config_dict["sovits_model_path_zh"]
        self.ref_audio_path_zh     = self.my_config_dict["ref_audio_path_zh"]
        self.prompt_text_zh        = self.my_config_dict["prompt_text_zh"]
        
        # zh
        self.gpt_model_path_en     = self.my_config_dict["gpt_model_path_en"]
        self.sovits_model_path_en  = self.my_config_dict["sovits_model_path_en"]
        self.ref_audio_path_en     = self.my_config_dict["ref_audio_path_en"]
        self.prompt_text_en        = self.my_config_dict["prompt_text_en"]
        
        # fufu
        self.gpt_model_path_fufu     = self.my_config_dict["gpt_model_path_fufu"]
        self.sovits_model_path_fufu  = self.my_config_dict["sovits_model_path_fufu"]
        self.ref_audio_path_fufu     = self.my_config_dict["ref_audio_path_fufu"]
        self.prompt_text_fufu        = self.my_config_dict["prompt_text_fufu"]
        
        # liu
        self.gpt_model_path_liu     = self.my_config_dict["gpt_model_path_liu"]
        self.sovits_model_path_liu  = self.my_config_dict["sovits_model_path_liu"]
        self.ref_audio_path_liu     = self.my_config_dict["ref_audio_path_liu"]
        self.prompt_text_liu        = self.my_config_dict["prompt_text_liu"]
        
        # 原10.5G 显存占用12.3G  一个模型展1.8G  # 13个G左右，但是每次调用之后会占用一定的显存，没有清楚掉
        self.gpt_infer_model_zh    = GPT_SoVITS_inference(self.gpt_model_path_zh, 
                                                       self.sovits_model_path_zh, 
                                                       self.ref_audio_path_zh, 
                                                       self.prompt_text_zh,
                                                       "中文")
        
        self.gpt_infer_model_en    = GPT_SoVITS_inference(self.gpt_model_path_en, 
                                                       self.sovits_model_path_en, 
                                                       self.ref_audio_path_en, 
                                                       self.prompt_text_en,
                                                       "英文")
        
        self.gpt_infer_model_fufu  = GPT_SoVITS_inference(self.gpt_model_path_fufu,
                                                          self.sovits_model_path_fufu,
                                                          self.ref_audio_path_fufu,
                                                          self.prompt_text_fufu,
                                                          "中文")
        
        self.gpt_infer_model_liu = GPT_SoVITS_inference(self.gpt_model_path_liu,
                                                          self.sovits_model_path_liu,
                                                          self.ref_audio_path_liu,
                                                          self.prompt_text_liu,
                                                          "中文")
        
        end_time                   = time.time()
        print(f"<=================== GPT_SoVITS_inference end init ===================>")
        print(f"{end_time -start_time} \n")

        # load extractor related parameters
        self.extractor_ckpt        = self.my_config_dict["extractor_ckpt"]
        
        # load extractor checkpoint
        self.extractor = self.load_checkpoints_extractor(self.config_extractor, self.extractor_ckpt)
        self.generator, self.kp_detector, self.audio2kptransformer, self.sidetuning, self.emotionprompt = self.build_EAT_model(self.build_model_config, self.EAT_ckpt) 
        # self.load_ckpt_for_EAT_model(self.EAT_ckpt,   # 整合到函数里面
        #                              self.kp_detector, 
        #                              self.generator, 
        #                              self.audio2kptransformer, 
        #                              self.sidetuning, 
        #                              self.emotionprompt)

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

    # """对输入的音频进行判断，如果不是路径就保存再传递路径"""
    # def ASR_ZH(self, audio_in_path):
    #     if not isinstance(audio_in_path, str):
    #         save_folder = "./demo/test_asr/ZH"
    #         save_file = os.path.join(save_folder, "asr_audio.wav")
    #         samplerate, data = file_obj
    #         sf.write(save_file, data, samplerate = samplerate)  # 假设采样率为 44100
    #         print("音频文件已保存到:", save_file)
    #         audio_in_path = save_file
    #     asr_result_map    = self.asr_pipeline_zh(audio_in_path)
    #     asr_result        = asr_result_map[0]['text']
    #     return asr_result
    
    # """对输入的音频进行判断，如果不是路径就保存再传递路径"""
    # def ASR_EN(self, audio_path):
    #     if not isinstance(audio_in_path, str):
    #         save_folder = "./demo/test_asr/EN"
    #         save_file   = os.path.join(save_folder, "asr_audio.wav")
    #         samplerate, data = file_obj
    #         sf.write(save_file, data, samplerate = samplerate)  # 假设采样率为 44100
    #         print("音频文件已保存到:", save_file)
    #         audio_in_path = save_file
    #     res = model.generate(input = audio_path, 
    #         batch_size_s=300, 
    #         hotword='魔搭')
    #     return res[0]['text']
    
    # 对输入的文件进行判断，如果是对象直接保存，不是的话直接调用asr_pipline
    def ASR_Linguage(self, audio_in_path, language):
        if language == "中文":
            language = "zh"
        elif language =="英文":
            language = "en"
        else:
            raise ValueError("No suitable lanuage!")
            exit()
            
        """
        来自User test
        或者来自 User metahuman
        """
        if not isinstance(audio_in_path, str):
            save_folder = "./audio_file/audio_asr_test"  # 指定存放测试音频路径文件
            
            save_file = os.path.join(save_folder, f"asr_audio_{language}.wav")
            samplerate, data = audio_in_path
            sf.write(save_file, data, samplerate = samplerate)  # 假设采样率为 44100
            print("音频文件已保存到:", save_file)
            audio_in_path = save_file
        
        if language   == "zh":
            asr_result_map    = self.asr_pipeline_zh(audio_in_path)
            asr_result        = asr_result_map[0]['text']
            return asr_result
        elif language == "en":
            res = self.asr_pipeline_en.generate(input = audio_in_path, 
                                 batch_size_s=300, 
                                 hotword='魔搭')
            return res[0]['text']
        
    
    # 实际不计算梯度的闭包函数
    def no_grad_decorator(func):
        def wrapper(*args, **kwargs):
            with torch.no_grad():
                return func(*args, **kwargs)
        return wrapper
    """
    现在多了一个选项，说话人还是语种，先选语种再选说话人，并且分为测试数据保存和模块推理保存
        如果是测试传过来的：那就没有abs_wavpath，这个项为None
        如果是模块传过来的，会在call中组织abspath的具体的值
    """
    @torch.no_grad()   # 不计算梯度
    def specialTTS(self, input_text, language, voice_type, abs_wavpath = None):
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        role = ""
        # 根据abs的path判断即可，如果是来自User的测试数据，路径的父目录肯定是audio_tts,如果传过来是空值，重新给个路径
        if abs_wavpath is None:  # 说明test传过来的
            language_mapping = lambda language: "zh" if language == "中文" else "en"  # 调用时，相当于函数的调用
            if voice_type == "李文":
                role = "liwen"
            elif voice_type == "芙芙":
                role = "fufu"
            elif voice_type == "流萤":
                role ="liuying"
            audio_name = f"{language_mapping(language)}_{formatted_now}_test_{role}.wav"
            abs_wavpath = os.path.join("./audio_file/audio_tts_test", audio_name) # 写入配置文件
        
        if language == "英文":
            if role == "liwen":
                audio_tts_en = self.gpt_infer_model_en.get_tts_wav(input_text, language)
            else:
                return None
        elif language == "中文":
            if role == "liwen":
                audio_tts_zh = self.gpt_infer_model_zh.get_tts_wav(input_text, language)
            elif role == "fufu":
                audio_tts_zh = self.gpt_infer_model_fufu.get_tts_wav(input_text, language)
            elif role == "liuying":
                audio_tts_zh = self.gpt_infer_model_liu.get_tts_wav(input_text, language)
        else:
            print("error language")
            return None
        sf.write(abs_wavpath, audio_tts_zh, self.sample_rate, 'PCM_24')  # 写入指定路径
        import gc; gc.collect()
        # 清空显存
        torch.cuda.empty_cache()
        return abs_wavpath
        
    
    def postChatGLM(self, asr_result, language_type):
        data    = {
                    "prompt": asr_result + f", \
                    请用{language_type}回答，不要超过50个词", "history": []}
        
        response = requests.post("http://0.0.0.0:8000",  
                                    json=data,
                                    headers={"Content-Type": "application/json"})
        
        print("__call__,response.text", response.text)
        chat_result_dict = eval(response.text)             
        # chat_result_dict = response.text
        chat_input       = chat_result_dict["response"]
        return chat_input
    
    
    """ load config file """
    def getConfigYaml(self, yaml_file_path):
        with open(yaml_file_path, 'r') as config_file:
            try:
                my_config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
                return my_config_dict
            except ValueError:
                print('INVALID YAML file format.. Please provide a good yaml file')
                exit(-1)
    
    
    """ build 5 EAT model """
    def build_EAT_model(self, config, ckpt,  device_ids=[0]):
        start_time = time.time()
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        generator = OcclusionAwareSPADEGeneratorEam(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        generator.load_state_dict(checkpoint['generator'])
        
        kp_detector         = KPDetector(**config['model_params']['kp_detector_params'],
                                         **config['model_params']['common_params'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        
        audio2kptransformer = Audio2kpTransformer(**config['model_params']['audio2kp_params'], face_ea=True)
        audio2kptransformer.load_state_dict(checkpoint['audio2kptransformer'])
        
        sidetuning          = EmotionalDeformationTransformer(**config['model_params']['audio2kp_params'])
        sidetuning.load_state_dict(checkpoint['sidetuning'])
        
        emotionprompt       = EmotionDeepPrompt()
        emotionprompt.load_state_dict(checkpoint['emotionprompt'])
        
        if torch.cuda.is_available():
            print('cuda is available')
            sidetuning.to(device_ids[0])
            emotionprompt.to(device_ids[0])
            audio2kptransformer.to(device_ids[0])
            kp_detector.to(device_ids[0])
            generator.to(device_ids[0])
        
        # 设置为评估模式
        audio2kptransformer.eval()
        generator.eval()
        kp_detector.eval()
        sidetuning.eval()
        emotionprompt.eval()
        
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
        start_time      = time.time()
        # bind session
        # self.io_binding = ##
        end_time        = time.time()
        print(f"<==================== TensorRT backend build successful! ====================>")
        print(f"Cost {end_time - start_time} \n")
        
        
    """" 
        pipeline: asr--> LLM --> TTS => audio.wav 
        audio_in_path: audio file path, *.wav
        tts_audio_path: tts root
        language_type: user choice
        """
        
    def generate_audio(self, audio_in_path, tts_audio_path, language_type, voice_type):
        start_time  = time.time()
        # 对语言进行判断定
        if language_type== "中文":
            abs_wavpath = os.path.join(tts_audio_path, "tts_zh.wav")
        elif language_type == "英文":
            abs_wavpath = os.path.join(tts_audio_path, "tts_en.wav")
        asr_result  = self.ASR_Linguage(audio_in_path, language_type)
        chat_input  = self.postChatGLM(asr_result, language_type)
        audio_tts   = self.specialTTS(chat_input, language_type, voice_type, abs_wavpath)
        
        # 对TTS的返回值进行判断
        if audio_tts is None:
            raise ValueError("Invalid audio_tts value")
        end_time    = time.time()
        print(f"<==================== Interactive model end====================>")
        print(f"Cost {end_time - start_time} \n")
        return abs_wavpath, chat_input
        
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

        #定义输入与输出
        # input_buff     = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous() # 判断Tensor按行展开后的顺序与其storage的顺序是否一致
        # output_buff    = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
        
        # loop every frame to restoration
        # ......Hide details
        
        end_time = time.time()
        
        print(f"==================== End GFPGAN restorate frames ====================")
        print(f"Cost {end_time - start_time}")
        return result_list
    
    def write_images_restoration(self, result_list, res_path="./save_videos/restoration"):
        for index in range(len(result_list)):
            # result_list[index] = cv2.cvtColor(result_list[index], cv2.COLOR_RGB2BGR)
            cv2.imwrite("%s/%d_results.png"%(res_path, index), result_list[index])
        return res_path
    
    def test_gfpgan(self, image_dir="", out_dir=""):
        res_path = "./demo/test_gfpgan"
        images_path   = glob.glob("*.png")
        image_np_list = []
        for image in images_path:
            image_np  = cv2.imread(image)
            image_np  = image_np[:,:,::-1] 
        image_np_list.append(image_np)
        # 0-255 BGR 
        result_list = self.frame_restoration(image_np_list)
        restoration_path = self.write_images_restoration(result_list, res_path)
    
    # BCHW
    def save_image_from_eat(self, image_tensor_list, res_path="./demo/test_original"):
        img_list = []
        for i in range(image_tensor_list.shape[0]):
            img_tensor = image_tensor_list[i]*255
            img_numpy = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_bgr = img_numpy[:, :, ::-1]
            img_list.append(img_bgr)
        
        for index in range(len(img_list)):
            cv2.imwrite("%s/%d_results.png"%(res_path, index),img_list[index])     
            
    def concat_video_RGB(self, frame_list, audio_path, save_video_path):
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        # fname      = formatted_now + ".mp4"
        fname         = f"talking_restoration_{formatted_now}.mp4"
        
        temp_dir   = "./gradio_mp4/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        video_path = os.path.join(temp_dir, fname)
        imageio.mimsave(video_path, frame_list, fps=25.0)
        save_video = os.path.join(save_video_path, fname)
        cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
        os.system(cmd)
        return video_path
    
    def concat_video_EAT(self, frames, audio_path, save_video_path):
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        fname      = f"talking_{formatted_now}.mp4"
        # fname         = "talking.mp4"
        temp_dir   = "./gradio_mp4/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, fname)
        
        frame_list = []
        for frame in frames:
            frame = (frame.squeeze(0).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
            # frame = np.clip(frame, 0 , 255)
            frame_list.append(frame)
        imageio.mimsave(video_path, frame_list, fps=25.0)

        save_video = os.path.join(save_video_path, fname)
        # cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
        # os.system(cmd)
        # 声音好像没合上去
        template   = 'ffmpeg -loglevel error -y -i {} -i {} -vcodec copy -shortest {}'
        cmd = template.format(video_path, audio_path, save_video)
        subprocess.run(cmd, shell=True)
        return video_path
    
    def transfer_emotype(self, emotype):
        if emotype == "生气":
            emotype="ang"
        elif emotype == "满足":
            emotype="con"
        elif emotype == "厌恶":
            emotype = "dis"
        elif emotype == "恐惧":
            emotype = "fea"
        elif emotype == "开心":
            emotype = "hap"
        elif emotype == "中性":
            emotype = "neu"
        elif emotype == "伤心":
            emotype = "sad"
        elif emotype == "惊喜":
            emotype = "sur"
        return emotype
    
    # call调用函数才会被调用
    def save_audio(self, file_obj, save_folder):
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
        audio_name  = f"asr_audio_{formatted_now}.wav"
        save_file = os.path.join(save_folder, audio_name)
        samplerate, data = file_obj
        sf.write(save_file, data, samplerate = samplerate)  # 假设采样率为 44100
        print("音频文件已保存到:", save_file)
        return save_file
    
    
    def torch_gc(self,):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        CUDA_DEVICE = f"cuda:{cuda_visible_devices}"
        if torch.cuda.is_available():
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
    def judge_pose(self, pose):
        if pose=="驱动动作1":
            pose_path = "./preprocess/poseimg/mytemplate.npy.gz"
            return pose_path
        elif pose=="驱动动作2":
            pose_path = "./demo/video_processed/obama/poseimg/obama.npy.gz"
            return pose_path
        elif pose=="驱动动作3":
            pose_path = "./demo/video_processed/W015_neu_1_002/poseimg/W015_neu_1_002.npy.gz"
            return pose_path
            
            
    """
    parameters:
        img:
        emotype: input emotional label
        language_type: 
    """
    
    """audio_file obj
    (48000, array([[0, 0],
       [0, 0],
       [0, 0],
       ...,
       [1, 0],
       [0, 0],
       [0, 0]], dtype=int16))
    """
    @torch.no_grad()
    def __call__(self, img, emotype, audio_file, language_type, pose, voice_type, tts_audio_path="./audio_file/audio_tts"):
        pose_path = self.judge_pose(pose)
        save_folder = "./audio_file/audio_asr"
        save_file  = self.save_audio(audio_file, save_folder)  # 保存输入到ASR的音频文件到指定的目录audio_path
        audio_path = save_file
        save_video_path = "./save_videos"  # 拿到配置里面去
        restoration_video_path = "./save_videos/restoration"
        emotype = self.transfer_emotype(emotype)
        start_time  = time.time()
        abs_wavpath, chat_input = self.generate_audio(audio_path, tts_audio_path, language_type, voice_type)
        infer_model = EAT_infer(abs_wavpath, img, emotype, self.build_model_config, self.extractor, pose_path)
        frame_list  = infer_model(self.model_list)   # B H W C RGB
        # 保存一下EAT的模型输出结果
        video_path  = self.concat_video_EAT(frame_list, abs_wavpath, save_video_path)
        restoration_frame_list = self.frame_restoration(frame_list)
        # restoration_path = self.write_images_restoration(frame_list)
        

        # abs_wavpath = "./demo/test_wav/output.wav"
        restoration_path = self.concat_video_RGB(restoration_frame_list, abs_wavpath, restoration_video_path)
        # video_path = self.concat_video_RGB(frame_list, abs_wavpath, save_video_path)
        print("frame counts", str(len(frame_list)) + "\n")
        end_time = time.time()
        print(f"<==================== Pipeline finished ====================>")
        print(f"Cost {end_time - start_time} \n")
        
        import gc; gc.collect()
        del frame_list  # 回收张量
        del infer_model # 将推理模型回收掉
        # self.torch_gc() # 回收显存
        return video_path, restoration_path, chat_input
        
    
            
if __name__=="__main__":
    emotype = "sad"
    start_time = time.time()
    my_config_file_path = "./config/my_yaml_wholepipeline_gptsovits.yaml"
    model = metahuman(my_config_file_path)
    end_time = time.time()
    print(f"Metahuman init total cost {end_time - start_time} \n")
    
    start_time = time.time()
    
    audio_path = "./demo/video_processed/W015_neu_1_002/W015_neu_1_002.wav"
    wav_path = "./audio_temp"
    img_path = "./demo/imgs/out.jpg"
    emotype = "hap"
    model(img_path, emotype, audio_path, wav_path)
    end_time = time.time()
    print(f"Inference Cost Time {end_time - start_time}")
    

