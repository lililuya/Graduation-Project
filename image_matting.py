"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image

import onnx
import onnxruntime

class Matting:
    def __init__(self, onnx_model_path = "./MODNet/pretrained/modnet.onnx"):
        if not os.path.exists(onnx_model_path):
            print('Cannot find the ONXX model: {0}'.format(onnx_model_path))
            exit()
        self.ref_size = 512
        self.session = onnxruntime.InferenceSession(onnx_model_path, None)
        
    # Get x_scale_factor & y_scale_factor to resize image
    def get_scale_factor(self, im_h, im_w, ref_size):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h
        return x_scale_factor, y_scale_factor

    """ get the matting image
        image: BGR  H W C"""
    def concat_matting_iamge(self, original_image, matte, color):
        # output = foreground * mask + background*(1 - mask)
        trasplant_bg = np.zeros((matte.shape[0], matte.shape[1], 4), dtype=np.uint8)
        color_R, color_G, color_B = color
        print(color_R, color_G, color_B)
        trasplant_bg[:, :, 0] = color_R  # 设置红色通道为255
        trasplant_bg[:, :, 1] = color_G  # 设置绿色通道为255
        trasplant_bg[:, :, 2] = color_B  # 设置蓝色通道为255
        trasplant_bg[:, :, 3] = 255
        
        rgba_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = original_image[:, :, :3]
        rgba_image[:, :, 3] = 255  # 第四个通道，完全不透明
        
        matte = matte.astype(np.float32) / 255.0
        matte = np.expand_dims(matte, axis=2)
        
        out = rgba_image * matte + (1 - matte) * trasplant_bg  # RGBA
        # cv2.imwrite("out.jpg", out)
        return out
    
    def __call__(self, image, color):
        if isinstance(image, np.ndarray):
            im = image
            # original_image = image[..., ::-1]
            original_image = image
        elif isinstance(image, str):
            im = cv2.imread(image)
            original_image = im.copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # unify image channels to 3
        
        # 颜色转16进制
        if color == "红色":
            color = (255, 0, 0)
        elif color == "蓝色":
            color = (0, 0, 255)
        elif color =="绿色":
            color = (0, 255, 0)     
        elif color =="白色":
            color = (255, 255, 255)
        else:
            pritn("no specify color! Error")
            exit()
            
            
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # normalize values to scale it between -1 to 1
        im = (im - 127.5) / 127.5   

        im_h, im_w, im_c = im.shape
        x, y = self.get_scale_factor(im_h, im_w, self.ref_size) 

        # resize image
        im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

        # prepare input shape
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis = 0).astype('float32')

        # Initialize session and get prediction
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: im})

        # refine matte
        matte = (np.squeeze(result[0]) * 255).astype('uint8')
        matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)
        # print(matte.shape)
        # print(matte[0])
        # print(original_image.shape)
        combine = self.concat_matting_iamge(original_image, matte, color)
        # cv2.imwrite(args.output_path, matte)
        combine = (combine/255.).astype(np.float32)
        return combine

if __name__=="__main__":
    onnx_model_path = "./MODNet/pretrained/modnet.onnx"
    image_path = "./demo/imgs/me.jpg"
    matting_model = Matting(onnx_model_path)
    matting_model(image_path)