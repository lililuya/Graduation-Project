# 基于大模型的高质量情感虚拟人系统
## 一些测试结果
### 测试卡通人像

| original                 | gfpgan      |   driven image |
|:--------------------: |:--------------------: | :----: |
| <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/c4ff0ceb-2bc4-419a-aa62-20d78793f2b2" type="video/mp4"> </video> | <video  src="https://github.com/lililuya/Graduation-Project/assets/141640497/bba70b1e-b3a3-4918-bc86-e0bf77ae5c26" type="video/mp4" > </video>  | <img src='./img/comic.jpg' width='380'> 











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
