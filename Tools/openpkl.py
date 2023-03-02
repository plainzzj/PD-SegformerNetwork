import pickle
import os
import numpy as np

path = '/mmsegmentation/results_pkl'
filename = "pkl"

file_list = os.listdir(path)
for file in file_list:
    cur_path = os.path.join(path, file)
    if os.path.isdir(cur_path):
        continue
    else:
        if filename in file:
            fr = open(cur_path, 'rb')
            inf = pickle.load(fr)
            doc = open('1.txt', 'a')
            print(inf, file=doc)
            np.set_printoptions(threshold=np.inf)
            np.set_printoptions(linewidth=1000)