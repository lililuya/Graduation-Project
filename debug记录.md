# 改进后的EAT模块出来的人脸嘴部非常不同步
## 推测问题
+ 1.由于修改过deepspeech特征提取模块，所以判断deepspeech特征提取错误
+ 2.由于对latent进行过截断，判断driving latent的影响
+ 3.由于poseimg的影响
+ 4.由于模型状态的影响

## 做法
### 模型状态
+ 做法
    + 打印extractor和许相关模型的状态
+ 结果
    + 模型处于eval态，正常

### deepspeech影响
+ 做法
    + 对照实验
        + 1.将原来的的EAT模型中的音频和对应的deepspeech的特征放入到改进后的EAT模块中看看结果
        + 2.将我现在的中文音频和提取的deepspeech特征放入到原来的EAT模块中进行测试
    + 结果
        + 1.原来的英文数据和对应deepspeech特征在改进后的EAT模型这儿是生效的
    + 推测
        + 针对上述实验现象，推测deepspeech模块对中文的是被好像有问题，提取出来的特征不对
    + 进一步做法
        + 对EAT模块生成的中文音频数据和特征放入到原始的EAT模型中作测试

### latent的影响
+ 做法
    + 1.将自定义的driving latent放到原来的ETA模块中进行测试，看看原来EAT模块会不会出现不同步的问题
    + 2.将原来的EAT模块使用的相关的driving latent和音频放到我自己的模块中看看会不会出现不同步的问题
    