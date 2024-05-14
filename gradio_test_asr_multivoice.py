import gradio as gr


import webbrowser
from generate_txt_byglm import ChatGLMModel
from seafoam_theme import Seafoam
from image_matting import Matting
from whole_pipeline_GPTSOVITS_asr_en_gradio_multivoice import metahuman
import requests
import unicodedata

"""
message: 我要传入的信息
max_length: 设置的最大的长度
"""

text_generator = ChatGLMModel()
seafoam = Seafoam()
matting_model = Matting()
talking = metahuman()

"""
大致思路：
    1.将传入的音频保存在服务器的某个目录下，EAT模型读取这个目录即可
    2.传入的图片是RGB float numpy ，直接拿到EAT模型进行处理即可
"""
# def save_audio(self, file_obj):
#     save_folder = "./demo/test_asr/"
#     save_file = os.path.join(save_folder, "asr_audio.wav")
#     samplerate, data = file_obj
#     sf.write(save_file, data, samplerate = samplerate)  # 假设采样率为 44100
#     print("音频文件已保存到:", save_file)

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{line.split("```")[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "`")
                    line = line.replace("<", "<")
                    line = line.replace(">", ">")
                    line = line.replace(" ", " ")
                    line = line.replace("*", "*")
                    line = line.replace("_", "_")
                    line = line.replace("-", "-")
                    line = line.replace(".", ".")
                    line = line.replace("!", "!")
                    line = line.replace("(", "(")
                    line = line.replace(")", ")")
                    line = line.replace("$", "$")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def normalize_string(text):
    normalized_text = unicodedata.normalize("NFKC", text)
    # 将非ASCII字符转换为ASCII字符
    normalized_text = ''.join(char for char in normalized_text if unicodedata.category(char)[0] in ('L', 'M', 'N', 'P', 'Z', 'S'))
    return normalized_text
    

# send_request, [user_input, chatbot, max_length, top_p, temperature, history
def send_request(prompt, max_length, top_p, temperature, history):
    url = 'http://localhost:8000/'  # FastAPI 服务器地址
    data = {
        "prompt": prompt,
        "history": history,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature
    }
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    result = response.json()
    
    if not result['history']:  # 检查历史记录是否为空
        result['history'] = [["", ""]]  # 如果为空，返回一个包含空对话对的列表
    
    updated_history = result['history']
    return updated_history  # 直接返回更新后的历史记录

def clear_history():
    return [["", ""]]   # 其中每个对话对由两个空列表组成,即使没有实际的对话内容，chatbot 组件也可以正确地处理并显示空的对话界面


with gr.Blocks(theme=seafoam) as demo:
    with gr.Tabs(): # 保存选项卡的状态，切换时不会重载选项卡
        with gr.TabItem("情感对话虚拟人生成"):
            with gr.Row():
                    with gr.Tabs(elem_id="eat_source_image"):
                        with gr.Row():
                            with gr.TabItem('选择虚拟人交互的语言'):
                                languageChoice  = ["中文", "英文"]
                                language_type = gr.Dropdown(label="选择语言", choices=languageChoice, value="中文")
                        with gr.Row():
                            with gr.TabItem('录制或上传您的指定语种的参考音频'):
                                audio_file = gr.Audio(label="参考音频") # 录音
                        # 这个地方怎么触发这个自动ASR
                        with gr.Row():
                            with gr.TabItem("ASR转录结果展示"):
                                message_ASR= gr.Textbox(label="ASR结果", placeholder="此处将展示ASR结果", lines=4)
                                show_Btn   = gr.Button("显示ASR结果", variant="primary")
                                show_Btn.click(talking.ASR_Linguage, [audio_file, language_type], [message_ASR])
                                
                        with gr.Row():
                            with gr.TabItem("大模型回复结果展示"):
                                message_LLM  = gr.Textbox(label="回复文本", placeholder="此处将展示回复的文本信息", lines=6)
                                
                
                    with gr.Tabs(elem_id="eat_source_image"):
                        with gr.Row():
                            with gr.TabItem('上传您的参考图片'):
                                source_image = gr.Image(label="选择人物图像", type="numpy")
                            
                        with gr.Row():
                            with gr.TabItem('选择指定的参数'):
                                pose_of_image   = gr.Radio(['驱动动作1', '驱动动作2', '驱动动作3'], label='选择驱动的动作', info="") # 
                                preprocess_type = gr.Radio(['生气', '满足', '厌恶', '恐惧', '开心', '中性', '伤心', '惊喜'], label='选择表情标签', info="")
                                voice_type = gr.Radio(['李文', '芙芙', '雷电将军', '流萤'], label='选择角色声音', info="")
                                genVideoBtn = gr.Button("生成情感说话人视频", variant="primary")
                    with gr.Tabs(elem_id="eat_source_image"):
                        with gr.Row():
                            with gr.TabItem('虚拟人视频生成原模型效果'):
                                # numpy float32 RBG
                                gen_video = gr.Video(label="Generated video", format="mp4", include_audio=True, autoplay=True)
                        with gr.Row():
                            with gr.TabItem('虚拟人视频生成超分效果'):
                                restoration_video = gr.Video(label="Restoration video", format="mp4", include_audio=True, autoplay=True)
                            genVideoBtn.click(talking, [source_image, preprocess_type, audio_file, language_type, pose_of_image, voice_type], [gen_video, restoration_video, message_LLM])

                        # with gr.Row(scale=6):  # numpy float32 RBG
                        #     with gr.TabItem('虚拟人视频生成超分模型效果'):
                        #         gen_video = gr.Video(label="Generated video", format="mp4", audio=True, autoplay=True)
                        #         genVideoBtn.click(talking, [source_image, preprocess_type, audio_file, language_type], [gen_video, message_LLM])
                                    
        
        with gr.TabItem("ChatGLM2-6B量化大模型"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(scale=12):
                        chatbot = gr.Chatbot()
                    with gr.Column(min_width=32, scale=1):
                        user_input = gr.Textbox(show_label=False, placeholder="请您问出您想要问的问题...", lines=10)
                        submit_button = gr.Button("提交您的输入", variant="primary")
                with gr.Column(scale=1):
                    with gr.Row():
                        clear_button = gr.Button("清除历史记录")
                        max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                    with gr.Row():
                        gr.Image(value = "./logo/font.png")
                    with gr.Row():
                        gr.Image(value = "./logo/logo.png")
            history = gr.State([])
            # past_key_values = gr.State(None)
            submit_button.click(
                send_request,
                inputs=[user_input, max_length, top_p, temperature, chatbot],
                outputs=[chatbot]
                )
            
            clear_button.click(
                clear_history,
                outputs=[chatbot]
                )


        # 测试完成
        with gr.TabItem("抠图测试"):
            with gr.Row():
                with gr.Column():
                    source_image = gr.Image(label="参考图像", type="numpy", format="png")
                    
                with gr.Column():
                    matting_image = gr.Image(label="抠图结果", type="numpy", format="png")
                
                with gr.Column():
                    color_choices  = ["红色", "蓝色", "绿色", "白色"]
                    color_dropdown = gr.Dropdown(label="选择纯色背景", choices=color_choices)
                    configBtn      = gr.Button("开始抠图")
                    configBtn.click(matting_model, [source_image, color_dropdown], [matting_image])
                    
        with gr.TabItem("中英文ASR测试"):
            with gr.Row():
                with gr.Column():
                    language_options = ["中文", "英文"]
                    language    = gr.Radio(language_options, label="选择语言")
                    asr_audio   = gr.Audio(label="输入待识别音频")
                    asr_result  = gr.Textbox(label="ASR识别结果", lines=8)
                    ASR_btn     = gr.Button("开始识别")
                    ASR_btn.click(talking.ASR_Linguage, [asr_audio, language], [asr_result])
                    
                # with gr.Column():
                #     with gr.TabItem('英文ASR测试'):
                #         en_asr_audio = gr.Audio(label="输入待识别中文音频", show_wave = True, interactive = True, editable = True)
                #         en_asr_result = gr.Textbox(label="英文ASR识别结果")
                #         en_ASR_btn = gr.Button("英文识别")
                #         en_ASR_btn.click(talking.ASR_EN, [en_asr_audio], [en_asr_result])
        
        with gr.TabItem("中英文TTS测试"):
            with gr.Row():
                with gr.Column():
                    language_options = ["中文", "英文"]
                    language    = gr.Radio(language_options, label="选择语言")
                    msg         = gr.Textbox(label="输入对应文本信息", lines=8)
                    voice_type  = gr.Radio(['李文', '芙芙', '雷电将军', '流萤'], label='选择角色声音', info="")
                    audio       = gr.Audio(label="生成的TTS音频")
                    TTS_btn     = gr.Button("语音合成")
                    TTS_btn.click(talking.specialTTS, [msg, language, voice_type], [audio])
                    
                    
                # with gr.Column():
                #     with gr.TabItem('英文TTS测试'):
                #         language    = "英文"
                #         msg_en = gr.Textbox(label="输入英文文本")
                #         en_audio    = gr.Audio(label="生成的英文音频")
                #         en_TTS_btn = gr.Button("英文语音合成")
                #         en_TTS_btn.click(talking.specialTTS, [msg_en, language], [en_audio])
            
        
# 这可以防止资源争用和服务器过载
demo.queue().launch(share=False, inbrowser=True)
