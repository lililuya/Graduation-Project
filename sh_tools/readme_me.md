##  每次推理所需要的数据为deepfeature、image_evp、latent_evp、poseimgs、wav
|--name
    |--deepfeature32
        |--*.npy
    |--image_evp
        |--croped
            |--*.jpg
            |--...
    |--latent_evp
        |--*.npy
    |--poseimg
        |--*.npy.gz
|--name.wav



## tensorrt的安装与使用
1. unpack the tar file
    version="10.x.x.x"
    arch=$(uname -m)
    cuda="cuda-x.x"
    tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz 

2. Add the absolute path to the TensorRT lib directory to the environment variable LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH

3. Install the Python TensorRT wheel file
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl

4. Install the Python onnx-graphsurgeon wheel file.
cd TensorRT-${version}/onnx_graphsurgeon
python3 -m pip install onnx_graphsurgeon-0.5.0-py2.py3-none-any.whl



## 前台需要什么数据传过来
1. 用户自定义上传的音频
2. 用户自动义上传的头像
3. 给定pose，不需要用户自定义上传
4. 用户可以上传视频，选择将视频进行各种处理操作
