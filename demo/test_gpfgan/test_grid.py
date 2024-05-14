import os
import cv2
import numpy as np

def load_images_from_directory(directory, target_size, start_frame=1):
    images = []
    frame_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_count += 1
            if frame_count < start_frame:
                continue
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size)
            images.append(img)
    return images

def combine_images(image_list1, image_list2, num_cols):
    max_height = max(img.shape[0] for img in image_list1)
    max_width = max(img.shape[1] for img in image_list1)
    combined_image = np.zeros((max_height * 2, max_width * num_cols, 3), dtype=np.uint8)
    
    # 合并第一个目录的图片
    for idx, img in enumerate(image_list1[:num_cols]):
        x_start = idx * max_width
        combined_image[0:max_height, x_start:x_start+max_width] = img
    
    # 合并第二个目录的图片
    for idx, img in enumerate(image_list2[:num_cols]):
        x_start = idx * max_width
        combined_image[max_height:, x_start:x_start+max_width] = img

    return combined_image

# 目录路径
directory1 = '/mnt/sdb/cxh/liwen/EAT_code/demo/test_gpfgan/o1'  # 第一个目录路径
directory2 = '/mnt/sdb/cxh/liwen/EAT_code/demo/test_gpfgan/p1'  # 第二个目录路径

# 目标尺寸
target_size = (512, 512)

# 从第10帧开始取
start_frame = 10

# 加载图片
images1 = load_images_from_directory(directory1, target_size, start_frame)
images2 = load_images_from_directory(directory2, target_size, start_frame)

# 合并图片
combined_image = combine_images(images1, images2, num_cols=5)

# 保存拼接后的图片
output_path = 'output_combined_image.jpg'  # 输出图片的文件路径
cv2.imwrite(output_path, combined_image)

print("Combined image saved successfully.")
