from PIL import Image
import numpy as np

L_path='/root/disk1/zzj7/dataset/After/train_masks_gray/IMG_0251_1728_864_2592_1728.png'
L_image=Image.open(L_path)
img=np.array(L_image)

print(L_image.size)
print(img.shape)#高 宽 三原色分为三个二维矩阵
print(img)