from GPT_SoVits.synthesize_audio_EN import GPT_SoVITS_inference
from GPT_SoVits.GPT_SoVITS.utils import * 
import soundfile as sf



SoVITS_model_path = './GPT_SoVits/weights/EN/SoVITS_weights/liwen_en_e25_s100.pth'
GPT_model_path = './GPT_SoVits/weights/EN/GPT_weights/liwen_en-e25.ckpt'  # 我自己克隆的TTS
ref_audio_path = "./GPT_SoVits/reference_audio/reference_EN.wav"  # a slice of my audio


prompt_text = 'Give it a try and see if you can say it quickly and correctly'
prompt_language  ="英文"
infere_model = GPT_SoVITS_inference(GPT_model_path, SoVITS_model_path, ref_audio_path, prompt_text, prompt_language)

text = 'Compared to inefficient end-to-end training for emotional talking-head'
text_language = '英文'

audio = infere_model.get_tts_wav(text, text_language)
# print(audio,  32000)
sf.write("./result.wav", audio,  32000, 'PCM_24')