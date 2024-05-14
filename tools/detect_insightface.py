
import cv2, os
import numpy as np
from   tqdm import tqdm
from    insightface_func.face_detect_crop_single import Face_detect_crop

path = "./inputs/kim_7s_raw.mp4"
crop_size = 512
mode="none"
device = 'cuda'
tg_format = "png"
temp_dir  = "wocao1/"
os.makedirs(temp_dir, exist_ok=True)
detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
detect.prepare(ctx_id = 0, det_thresh=0.6,\
                        det_size=(640,640),mode = mode)

video = cv2.VideoCapture(path)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)

for frame_index in tqdm(range(frame_count)):
    ret, frame = video.read()
        
    if  ret:
        align_img, M = detect.get(frame,crop_size)
        align_img = align_img[0]
        align_img = cv2.copyMakeBorder(align_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0,0,0))
        bboxes = detect.det_model.detect(align_img,
                                             threshold=detect.det_thresh,
                                             max_num=0,
                                             metric='default')[0][0]
        print(bboxes)
        bboxes = [int(x) for x in bboxes]
        y1 = max(0, bboxes[1])
        y2 = min(frame.shape[0], bboxes[3])
        x1 = max(0, bboxes[0])
        x2 = min(frame.shape[1], bboxes[2])
        save = align_img[y1: y2, x1:x2]
        f_path =os.path.join(temp_dir, str(frame_index).zfill(6)+".%s"%(tg_format))
        cv2.imencode('.png',save)[1].tofile(f_path)
        # else:
        #     print("No face detected!")
