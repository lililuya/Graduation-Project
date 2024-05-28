import os
import numpy as np

import torch
import onnxruntime

import datetime
import glob
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def cut_frame_use_cv2(out_frame_dir, vid_path = "/mnt/sdb/cxh/liwen/EAT_code/save_videos/talking_2024-05-03-05-14-59.mp4"):
    print("start cut frames....")
    if not os.path.exists(out_frame_dir):
        os.makedirs(out_frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(vid_path)
    frame_count = 0
    while cap.isOpened(): # 检查cap是否被初始化
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        cv2.imwrite(os.path.join(out_frame_dir, f"frame_{frame_count}.png"), frame)
    print("finish cut frames....")


def cut_frame_use_moviepy(out_frame_dir, vid_path="/mnt/sdb/cxh/liwen/EAT_code/demo/test_gpfgan/o"):
    if not os.path.exists(out_frame_dir):
        os.makedirs(out_frame_dir, exist_ok=True)
    clip = VideoFileClip(vid_path)
    for i, frame in enumerate(clip.iter_frames()):
        frame_path = os.path.join(out_frame_dir, f"frame_{i}.jpg")
        frame.save_frame(frame_path, format="png")
    clip.close()

def test_gfpgan_tensorrt(image_dir, res_path, onnx_path = "/mnt/sdb/cxh/liwen/EAT_code/restoration/GFPGANv1.4.onnx"):
    imgs        = glob.glob(f"{image_dir}/*.png")
    # print(imgs)
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    img_list    = []
    result_list = []
    for n_img in imgs:  # BGR
        img_temp = cv2.imread(n_img)      # 0 - 255                               # 读取
        img_temp = cv2.resize(img_temp, (512,512), interpolation=cv2.INTER_CUBIC) # 插值缩放图像
        img_temp = img_temp[:,:,[2,1,0]]                                          # 转BGR为RGB
        img_temp = img_temp.transpose((2, 0, 1))                                  # H W C --> C H W 
        img_temp = 2*(torch.from_numpy(img_temp)/255 - 0.5).unsqueeze(0)          # B C H W  # - 1 1
        # img_temp = F.interpolate(img_temp,scale_factor=2)
        img_list.append(img_temp) 
    
    print(".....binding start.....") 
    session = onnxruntime.InferenceSession(onnx_path, providers=["TensorrtExecutionProvider","CUDAExecutionProvider"])
    io_binding = session.io_binding() 
    print("binding end.....")
    
    
    temp = torch.div(torch.zeros((1,3,512,512)), 255).cuda()
    outpred = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
    
    # io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=temp.data_ptr())
    # io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outpred.data_ptr())

    # # Sync and run model
    # session.run_with_iobinding(io_binding)
    
    print("start to test")
    start_time = datetime.datetime.now()
    for i in tqdm(range(len(img_list))):
        temp = img_list[i].cuda()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=temp.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outpred.data_ptr())
        session.run_with_iobinding(io_binding)
        outpred1 = torch.squeeze(outpred)
        outpred1 = torch.clamp(outpred1, -1, 1)
        outpred1 = torch.add(outpred1, 1)
        outpred1 = torch.div(outpred1, 2)
        outpred1 = torch.mul(outpred1, 255)[[2,1,0],:,:].permute(1,2,0).cpu().numpy()
        result_list.append(outpred1)
    end_time = datetime.datetime.now()
    print("time cost:", end_time-start_time)
    # Format back to cxHxW @ 255
    # outpred = torch.squeeze(outpred) 

    for index in range(len(result_list)):
        cv2.imwrite(os.path.join(res_path,  f"frame_{index+1}.png"), result_list[index])
    


if __name__=="__main__":
    out_frame_dir = "/mnt/sdb/cxh/liwen/EAT_code/demo/test_gpfgan/o1"
    res_path = "/mnt/sdb/cxh/liwen/EAT_code/demo/test_gpfgan/p1"
    cut_frame_use_cv2(out_frame_dir)
    test_gfpgan_tensorrt(out_frame_dir, res_path)
