import numpy as np
import gzip

latent_evp_25  ="/mnt/sdb/cxh/liwen/EAT_code/demo/video_processed/W009_sad_3_003/latent_evp_25/W009_sad_3_003.npy"

latent_evp_25 = np.load(latent_evp_25, allow_pickle = True)
# print(latent_evp_25[0].shape)# 关键点

for x in latent_evp_25[1].keys():
    print(x, latent_evp_25[1][x].shape)

"""
(1, 15, 3)
(46, 66)
(46, 66)
(46, 66)
(46, 3)
(46, 45)
"""


pose_gz = gzip.GzipFile("/mnt/sdb/cxh/liwen/EAT_code/demo/video_processed/obama/poseimg/obama.npy.gz", 'r')
poseimg = np.load(pose_gz) # (N, 1, 64, 64)
print(poseimg.shape)