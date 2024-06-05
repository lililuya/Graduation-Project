# 基于大模型的高质量情感虚拟人系统
+ 系统流程图
![image](https://github.com/lililuya/Graduation-Project/assets/141640497/0e9b4630-979e-4dde-b66c-32bf3880566e)

## 1. 一些测试结果
### 1.1 测试卡通人像
|driven image|original| gfpgan|
| :----: |:--------------------: |:--------------------: |
| <img src='./img/comic.jpg' width='380'> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/c4ff0ceb-2bc4-419a-aa62-20d78793f2b2" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/bba70b1e-b3a3-4918-bc86-e0bf77ae5c26" type="video/mp4" > </video> 

### 1.2 合成人物测试
|driven image|original| gfpgan|
| :----: |:--------------------: |:--------------------: |
| <img src='./img/diffusion_gen.jpg' width='380'> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/50802b55-b0c1-4cd4-ad82-158b85adb476" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/91081951-bcd9-4df6-bf5b-2cb84e53ee04" type="video/mp4" > </video>

### 1.3 不同表情测试
|driven image|happy|scared|neural|
| :----: |:--------------------: |:--------------------: |:--------------------: |
| <img src='./img/jiege.jpg' width='380'> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/7b155a42-6c87-4855-a5dd-70ac3b8c6686" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/5c63af56-2900-45f8-95db-045c951bccc0" type="video/mp4" > </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/38fd3123-21ee-4666-b5b3-e97a1a7f5853" type="video/mp4" > 

### 1.4 不同的声音测试
|liwen|fufu|liuying|
| :----: |:--------------------: |:--------------------: |
| <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/4692307e-c437-4cf1-a26d-a23791ea0b45" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/b76d61e8-8a92-4d1a-bb55-6240ff8fc166" type="video/mp4" > </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/91081951-bcd9-4df6-bf5b-2cb84e53ee04" type="video/mp4" > </video>| | <video  src="" type="video/mp4" > </video>

### 1.5 不同语言测试
|Chinese|English|
|:--------------------: |:--------------------: |
|<video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/6016ddbe-97d0-4c23-9cd9-c26e84326311" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/a415cdb4-cab0-4eae-b73e-705f4c425eef" type="video/mp4" > </video>|

### 1.6 不同的动作测试
|pose1|pose2|pose3|
|:--------------------: |:--------------------: |:--------------------: |
| <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/418a8299-b570-4c47-8c02-556b34a42b15" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/1de09d69-00d6-48cf-832b-662acfc4ac58" type="video/mp4" > </video>| <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/74dbf7a6-4c4f-40aa-8a60-d72e16d77149" type="video/mp4" > </video>|

## 2. 环境准备
### 2.1 准备EAT环境和GPT-SOVITS环境
+ 系统环境
  + Python 3.9.19
  + Ubuntu 20.04.1
  + Graphics-Card 2-4090
```bash
git clone https://github.com/lililuya/Graduation-Project.git
cd env
```
+ Use conda or pip 
```bash
# if use conda, modify the prefix of environment.yml or delete it to use the default location
conda env create -f environment.yml
```
```bash
# if use pip, delete some local package index.
pip install -r requirements.txt
```
### 2.2 ModelScope和GPT-SOVISTS的环境问题，以ModelScope的为准
```bash
pip install funasr==1.0.22
pip install modelscope==1.13.3
```
### 2.3 tensorrt安装
参考[tensorrt安装笔记](https://github.com/lililuya/break-stones/blob/main/library%20problem/Tensorrt%20install%20and%20usage.md)

### 2.4 一些问题
+ 主要可能出现的问题是numba版本的问题，出现后更新numba版本即可
```bash
pip install -U numba
```
## 3.权重文件
+ [EAT权重文件](https://drive.google.com/file/d/1KK15n2fOdfLECWN5wvX54mVyDt18IZCo/view?usp=drive_link)
    + 下载后放在根目录下的`ckpt`下
+ [GFPGAN](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo)
    + 下载后放到根目录的`restoration`下面
+ [GPTSOVIT权重](https://huggingface.co/kaze-mio/so-vits-genshin/tree/main)
    + 下载后放到根目录下的`GPT_SoVits/weights`下面
+ [MODNET权重](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing)
    + 下载后放在`pretrain`下面
+ [DeepSpeech](https://github.com/ashawkey/RAD-NeRF)
    + 参考RADNERF

## 4.运行
### 4.1 本地运行
```python
python whole_pipeline_GPTSOVITS_asr_en_gradio_multivoice.py
```
### 4.2 使用Gradio自带内网穿透
```bash
# Modify launch=True
```
+ 一些配置参考[Gradio Network Traversal](https://github.com/lililuya/Meta_Doctor)
### 4.3界面
+ 情感虚拟人生成模块
![test_record](https://github.com/lililuya/Graduation-Project/assets/141640497/a4d331ff-d060-47e7-924a-bb35fd6004b8)
+ 中英文TTS
![test_TTS_en](https://github.com/lililuya/Graduation-Project/assets/141640497/f4e3180b-92c1-4740-aa23-27c5e81c1fbb)
+ 中英文ASR
![page4](https://github.com/lililuya/Graduation-Project/assets/141640497/00deaa77-58f8-46f4-ae69-19013efbe520)
+ 抠图
![page3](https://github.com/lililuya/Graduation-Project/assets/141640497/69b37296-6d36-45ae-9b13-67c0b1c2f48d)

## 引用文献
```txt
@InProceedings{Gan_2023_ICCV,
    author    = {Gan, Yuan and Yang, Zongxin and Yue, Xihang and Sun, Lingyun and Yang, Yi},
    title     = {Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22634-22645}
}

@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}

@inproceedings{gao22b_interspeech,
  author={Zhifu Gao and ShiLiang Zhang and Ian McLoughlin and Zhijie Yan},
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2063--2067},
  doi={10.21437/Interspeech.2022-9996}
}

@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
## 相关仓库
+ [FUN-ASR-ZH](https://www.modelscope.cn/models/iic/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)
+ [FUN-ASR-EN](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary)
+ [TTS-Sovits](https://github.com/RVC-Boss/GPT-SoVITS)
+ [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
+ [ModNet](https://github.com/ZHKKKe/MODNet)
+ [DeepSpeech](https://github.com/mozilla/DeepSpeech)
+ [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)
+ [GFPGAN](https://github.com/TencentARC/GFPGAN)
+ [EAT](https://github.com/yuangan/EAT_code)

## 目前存在的问题
+ 同步问题，参考[issue](https://github.com/yuangan/EAT_code/issues/28)
+ 显存需要6G+10G才可以跑起来，现存占用过大。
+ 目前展示的结果效果不太好，因为选择的初始图片不太清晰，并且onnx下损失了超分模型的部分精度。
+ 头拼合进身体，[EAT作者建议](https://github.com/yuangan/EAT_code/issues/16)。
+ 背景抖动，[EAT作者建议](https://github.com/yuangan/EAT_code/issues/27)，本仓库采取MODNet方案。
+ Deepspeech加速，目前提取音频特征需要时间特别久，使用的deepspeech-0.1版本。
+ GPT-SOVITS模型自定义载，资源换时间，每个模型大约1.8G左右，可以写入配置文件自定义加载。
## 声明
本项目以EAT为核心模型，主要做一个实验探究，不存在任何其他用途。
