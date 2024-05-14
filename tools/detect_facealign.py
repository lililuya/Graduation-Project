import face_detection
import cv2, os
import numpy as np
from   tqdm import tqdm

path = "./inputs/kim_7s_raw.mp4"

device = 'cuda'
tg_format = "png"
temp_dir  = "wocao/"
os.makedirs(temp_dir, exist_ok=True)
detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

video = cv2.VideoCapture(path)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)

for frame_index in tqdm(range(frame_count)):
    ret, frame = video.read()
        
    if  ret:
        detect_results = detector.get_detections_for_batch(np.array([frame]))[0]
        y1 = max(0, detect_results[1])
        y2 = min(frame.shape[0], detect_results[3])
        x1 = max(0, detect_results[0])
        x2 = min(frame.shape[1], detect_results[2])
        save = frame[y1: y2, x1:x2]
        f_path =os.path.join(temp_dir, str(frame_index).zfill(6)+".%s"%(tg_format))
        cv2.imencode('.png',save)[1].tofile(f_path)
        # else:
        #     print("No face detected!")
