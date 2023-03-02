import os

path = '/root/disk1/zzj7/dataset/After/train_masks_gray/'
txts = os.listdir(path)
for txt in txts:
    temp = os.path.join(path, txt)
    print(temp)
