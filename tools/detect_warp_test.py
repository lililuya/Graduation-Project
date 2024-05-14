
import cv2, os
import torch
import torch.nn.functional as F
import numpy as np
from   tqdm import tqdm
from   insightface_func.face_detect_crop_single import Face_detect_crop

def cv2totensor(cv2_img):
    """
    cv2_img: an image read by cv2, H*W*C
    return: an 1*C*H*W tensor
    """
    # cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
    cv2_img = torch.from_numpy(cv2_img[:,:,[2,1,0]])

    cv2_img = cv2_img.permute(2,0,1)
    temp    = cv2_img / 255.0
    # temp    -= self.imagenet_mean
    # temp    /= self.imagenet_std
    return temp.unsqueeze(0)

path = "./inputs/kim_7s_raw.mp4"
crop_size = 512
mode="none"
device = 'cuda'
tg_format = "png"
temp_dir  = "wocao3/"
os.makedirs(temp_dir, exist_ok=True)
detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
detect.prepare(ctx_id = 0, det_thresh=0.6,\
                        det_size=(640,640),mode = mode)
image111 = cv2.imread("J:/makeup_raw/1 (17).jpg")


bboxes, mat, mat_inv = detect.get_detailed_parameters(image111, crop_size)
mat[0][0]= 1/mat[0][0]
mat[1][1]= 1/mat[1][1]
mat[0][2]= -mat[0][2]/crop_size
mat[1][2]= -mat[1][2]/crop_size
mat 	= torch.from_numpy(mat).float().unsqueeze(0)
mat 	= F.affine_grid(mat, [1, 3, crop_size, crop_size], align_corners=True)
print(mat)
image111 = cv2totensor(image111)
print(image111)
warp_img = F.grid_sample(image111, mat, align_corners=True).squeeze().permute(1, 2, 0).numpy()
print(warp_img)
warp_img = np.clip(warp_img*255.0, 0, 255).astype(np.uint8)
f_path =os.path.join(temp_dir, str(1).zfill(6)+".%s"%(tg_format))
cv2.imencode('.png',warp_img)[1].tofile(f_path)
        # else:
        #     print("No face detected!")
