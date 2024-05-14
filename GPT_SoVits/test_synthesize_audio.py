import torch
import os
import re
import numpy as np
from time import time as ttime
from .GPT_SoVITS.tools.i18n.i18n import I18nAuto
from .GPT_SoVITS.my_utils import load_audio
from .GPT_SoVITS.feature_extractor import cnhubert
import librosa
from .GPT_SoVITS.module.models import SynthesizerTrn
from .GPT_SoVITS.text.cleaner import clean_text
from .GPT_SoVITS.text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .GPT_SoVITS.module.mel_processing import spectrogram_torch
import soundfile as sf

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
cnhubert.cnhubert_base_path = cnhubert_base_path
i18n = I18nAuto()
dict_language = {
    i18n("中文"): "all_zh",      #全部按中文识别
    i18n("英文"): "en",          #全部按英文识别#######不变
    i18n("日文"): "all_ja",      #全部按日文识别
    i18n("中英混合"): "zh",      #按中英混合识别####不变
    i18n("日英混合"): "ja",      #按日英混合识别####不变
    i18n("多语种混合"): "auto",  #多语种启动切分识别语种
}
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
is_half = eval(os.environ.get("is_half", "True"))

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def init_sovits_weights(sovits_path, device):
    global hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    return vq_model

def init_gpt_weights(gpt_path, device):
    global hz, max_sec, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    return t2s_model

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])

def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language.replace("all_",""))
    # print(phones, word2ph, norm_text)
    # print(phones, word2ph, norm_text)
    # phones = ['.', 'T', 'IY1','T', 'IY1', '.']
    # print(1)
    phones = cleaned_text_to_sequence(phones)
    # print(2)
    # print(phones, word2ph, norm_text)
    return phones, word2ph, norm_text

def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text

def get_cleaned_text_final(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


class GPT_SoVITS_inference:
    def __init__(self, GPT_model_path, SoVITS_model_path, ref_audio_path, prompt_text, device="cuda"):
        self.GPT_model_path = GPT_model_path
        self.SoVITS_model_path = SoVITS_model_path
        self.ref_audio_path = ref_audio_path
        language_combobox = i18n('中文')
        self.prompt_language = dict_language[language_combobox]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to(self.device)
        ssl_model = cnhubert.get_model().to(self.device)
        if is_half == True:
            ssl_model = ssl_model.half().to(device)
        else:
            ssl_model = ssl_model.to(device)
        
        self.ssl_model = ssl_model
        self.t2s_model = init_gpt_weights(self.GPT_model_path, self.device)
        self.vq_model = init_sovits_weights(self.SoVITS_model_path, self.device)
        

        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if self.prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)

        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half == True else np.float32,
        )

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_audio_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])

            # print('wav16k', wav16k)
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(
                    1, 2
                )  # .float()
            # print('ssl_content', ssl_content)
            codes = self.vq_model.extract_latent(ssl_content)
            # print('codes', codes)
            self.prompt_semantic = codes[0, 0]
        
        dtype=torch.float16 if is_half == True else torch.float32
        self.phones1, word2ph1, norm_text1=get_cleaned_text_final(prompt_text, self.prompt_language)
        self.bert1=self.get_bert_final(self.phones1, word2ph1, norm_text1, self.prompt_language).to(dtype)

        refer = get_spepc(hps, self.ref_audio_path)  # .to(device)
        if is_half == True:
            self.refer = refer.half().to(self.device)
        else:
            self.refer = refer.to(self.device)
        # print('refer', refer)

    def get_bert_final(self, phones, word2ph, text,language):
        if language == "en":
            bert = get_bert_inf(phones, word2ph, text, language)
        elif language in {"zh", "ja","auto"}:
            bert = nonen_get_bert_inf(text, language)
        elif language == "all_zh":
            bert = self.get_bert_feature(text, word2ph).to(self.device)
        else:
            bert = torch.zeros((1024, len(phones))).to(self.device)
        return bert

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T    
    
    def get_tts_wav(self, text, text_language, how_to_cut=i18n("凑四句一切"), top_k=5, top_p=1, temperature=1, ref_free = False):
        t1 = ttime()
        text_language = i18n(text_language)
        text_language = dict_language[text_language]
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        print(i18n("实际输入的目标文本:"), text)

        if (how_to_cut == i18n("凑四句一切")):
            text = cut1(text)
        elif (how_to_cut == i18n("凑50字一切")):
            text = cut2(text)
        elif (how_to_cut == i18n("按中文句号。切")):
            text = cut3(text)
        elif (how_to_cut == i18n("按英文句号.切")):
            text = cut4(text)
        elif (how_to_cut == i18n("按标点符号切")):
            text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print(i18n("实际输入的目标文本(切句后):"), text)
        texts = text.split("\n")
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []

        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half == True else np.float32,
        )

        dtype=torch.float16 if is_half == True else torch.float32
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2, word2ph2, norm_text2 = get_cleaned_text_final(text, text_language)
            bert2 = self.get_bert_final(phones2, word2ph2, norm_text2, text_language).to(dtype)
            #bert = torch.cat([bert1, bert2], 1)
            # print('bert2', bert2)

            if not ref_free:
                bert = torch.cat([self.bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(self.phones1+phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            # print('bert', bert)
            # all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = self.prompt_semantic.unsqueeze(0).to(self.device)
            t2 = ttime()
            # print('prompt', prompt)
            # print('all_phoneme_ids', all_phoneme_ids)
            # print('all_phoneme_len', all_phoneme_len)
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
            t3 = ttime()
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            
            # print('pred_semantic', pred_semantic)
            # print('idx', idx)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), self.refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            print('phones2', phones2)
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            # print(audio)
        result_audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
        t4 = ttime()
        print("%.3f\t%.3f\t%.3f" % (t2 - t1, t3 - t2, t4 - t3))
        return result_audio

    # def test_simple(self, text, text_language="中文", top_k=5, top_p=1, temperature=1, ref_free = False):
    #     text_language = i18n(text_language)
    #     text_language = dict_language[text_language]
    #     text = text.strip("\n")
    #     # print("tts text:", text)
    #     zero_wav = np.zeros(
    #         int(hps.data.sampling_rate * 0.3),
    #         dtype=np.float16 if is_half == True else np.float32,
    #     )

    #     dtype=torch.float16 if is_half == True else torch.float32
    #     # for text in [texts]:
    #     # # 解决输入目标文本的空行导致报错的问题
    #     # if (len(text.strip()) == 0):
    #     #     continue
    #     # if (text[-1] not in splits): text += "。" if text_language != "en" else "."
    #     # print(i18n("实际输入的目标文本(每句):"), text)
    #     phones2, word2ph2, norm_text2 = get_cleaned_text_final(text, text_language)
    #     bert2 = self.get_bert_final(phones2, word2ph2, norm_text2, text_language).to(dtype)

    #     if not ref_free:
    #         bert = torch.cat([self.bert1, bert2], 1)
    #         all_phoneme_ids = torch.LongTensor(self.phones1+phones2).to(self.device).unsqueeze(0)
    #     else:
    #         bert = bert2
    #         all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
    #     # all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    #     bert = bert.to(self.device).unsqueeze(0)
    #     all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
    #     prompt = self.prompt_semantic.unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         # pred_semantic = t2s_model.model.infer(
    #         pred_semantic, idx = self.t2s_model.model.infer_panel(
    #             all_phoneme_ids,
    #             all_phoneme_len,
    #             None if ref_free else prompt,
    #             bert,
    #             # prompt_phone_len=ph_offset,
    #             top_k=top_k,
    #             top_p=top_p,
    #             temperature=temperature,
    #             early_stop_num=hz * max_sec,
    #         )
    #     pred_semantic = pred_semantic[:, -idx:].unsqueeze(
    #         0
    #     )  # .unsqueeze(0)#mq要多unsqueeze一次
    #     audio = (
    #         self.vq_model.decode(
    #             pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), self.refer
    #         )
    #             .detach()
    #             .cpu()
    #             .numpy()[0, 0]
    #     )  ###试试重建不带上prompt部分
    #     max_audio=np.abs(audio).max()#简单防止16bit爆音
    #     if max_audio>1:audio/=max_audio
    #     # print(audio)
    #     result_audio = (np.concatenate((audio, zero_wav), 0) * 32768).astype(np.int16)
    #     return result_audio



if __name__ == '__main__':
    SoVITS_model_path = '/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/weights/EN/SoVITS_weights/liwen_en_e25_s100.pth'
    GPT_model_path = '/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/weights/EN/GPT_weights/liwen_en-e25.ckpt'  # 我自己克隆的TTS
    ref_audio_path = "/mnt/sdb/cxh/liwen/EAT_code/GPT_SoVits/reference_audio/reference_EN.wav"  # a slice of my audio
    
    
    prompt_text = 'Give it a try and see if you can say it quickly and correctly'
    infere_model = GPT_SoVITS_inference(GPT_model_path, SoVITS_model_path, ref_audio_path, prompt_text)

    text = 'Compared to inefficient end-to-end training for emotional talking-head'
    text_language = '英文'

    audio = infere_model.get_tts_wav(text, text_language)
    print(audio, hps.data.sampling_rate)
    sf.write("./result.wav", audio, hps.data.sampling_rate, 'PCM_24')