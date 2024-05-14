import os
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
from typing import Dict, Tuple, Union, Optional

from torch.nn import Module

def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 1,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().to(device)
    else:
        from accelerate import dispatch_model
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()
        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)
        model = dispatch_model(model, device_map=device_map)
    return model

class ChatGLMModel:
    def __init__(self, model_name="THUDM/chatglm2-6b-int4", num_gpus=1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.eval()

    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert(message),
                None if response is None else mdtex2html.convert(response),
            )
        return y

    def parse_text(self, text):
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

    def predict(self, input, chatbot, max_length, top_p, temperature, history, past_key_values):
        chatbot.append((self.parse_text(input), ""))
        for response, history, past_key_values in self.model.stream_chat(self.tokenizer, input, history, past_key_values=past_key_values,
                                                                        return_past_key_values=True,
                                                                        max_length=max_length, top_p=top_p,
                                                                        temperature=temperature):
            chatbot[-1] = (self.parse_text(input), self.parse_text(response))
            yield chatbot, history, past_key_values

    def reset_user_input(self):
        return gr.update(value='')

    def reset_state(self):
        return [], [], None