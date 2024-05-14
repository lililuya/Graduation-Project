from modelscope.pipelines import pipeline as asr_pipeline
from modelscope.utils.constant import Tasks
import requests
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Test():
    def __init__(self, audio_in, audio_tmp_dir) -> None:
        self.audio_in = audio_in
        self.audio_tmp_dir = audio_tmp_dir

        self.asr_pipeline   = asr_pipeline(
                            task=Tasks.auto_speech_recognition,
                            model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                            model_revision="v2.0.4"
                        )
        from TTS.synthesize_all import SpeechSynthesis
        self.tts_pipeline   = SpeechSynthesis('./TTS/config/AISHELL3')

    def generate_audio(self):
        asr_result_map  = self.asr_pipeline(self.audio_in)
        asr_result      = asr_result_map[0]['text']

        print(asr_result)
        data            = {
                            "prompt": asr_result + ", \
                            简短回答", "history": []
                        }
        response        = requests.post("http://0.0.0.0:8000",  
                                    json=data,
                                    headers={"Content-Type": "application/json"})
        chat_result_dict = eval(response.text)             
        chat_input       = chat_result_dict["response"]
        self.tts_pipeline.text2speech(chat_input, self.audio_tmp_dir)

    def __call__(self):
          self.generate_audio()

if __name__=="__main__":
      test = Test(audio_in="./dataset/Obama/aud.wav", audio_tmp_dir="./audio_temp")
      test()