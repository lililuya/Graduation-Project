# 基于大模型的高质量情感虚拟人系统
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
| <img src='./img/jiege.jpg' width='380'> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/7b155a42-6c87-4855-a5dd-70ac3b8c6686" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/5c63af56-2900-45f8-95db-045c951bccc0" type="video/mp4" > </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/91081951-bcd9-4df6-bf5b-2cb84e53ee04" type="video/mp4" > </video>| <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/21541e9a-c15f-4ce6-887f-fdb712e60a0e" type="video/mp4" > </video>|

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



































## 环境准备

## 一些环境上的问题

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
1. [FUN-ASR-ZH](https://www.modelscope.cn/models/iic/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)
2. [FUN-ASR-EN](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary)
3. [TTS-Sovits](https://github.com/RVC-Boss/GPT-SoVITS)
4. [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
5. [ModNet](https://github.com/ZHKKKe/MODNet)
6. [EAT](https://github.com/yuangan/EAT_code)
7. [DeepSpeech](https://github.com/mozilla/DeepSpeech)
8. [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)
9. [GFPGAN](https://github.com/TencentARC/GFPGAN)
10. [EAT](https://github.com/yuangan/EAT_code)
## 一些参考
1. 头拼合进身体，[EAT作者建议](https://github.com/yuangan/EAT_code/issues/16)。
2. 背景抖动，[EAT作者建议](https://github.com/yuangan/EAT_code/issues/27)，本仓库采取MODNet方案。
3. Deepspeech加速，目前提取音频特征需要时间特别久，使用的deepspeech-0.1版本。
4. GPT-SOVITS模型自定义载，资源换时间，每个模型大约1.8G左右，可以写入配置文件自定义加载。

## 目前的总占用显存情况
1. ChatGLM2-6B量化大模型---6G
2. EAT初始化化类中所有模型---13G（优化GPT-Sovits权重加载可以到10G）

## 声明
1. 仅供个人项目参考使用功能
