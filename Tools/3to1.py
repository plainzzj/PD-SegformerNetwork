import os

from PIL import Image
import io

input_dir = '/root/disk1/zzj7/dataset/After/val_masks/'
out_dir = '/root/disk1/zzj7/dataset/After/val_masks_gray/'
a = os.listdir(input_dir)

for i in a:
    print(i)
    I = Image.open(input_dir + i)
    L = I.convert('L')
    L.save(out_dir + i)
