import numpy as np
import os
import random

thing = "dmg"

files = os.listdir(f'{thing}_pics')
num_files = len(files)
print("loading")
labels = np.load(f"{thing}_labels.npy")
imgs = np.array([np.load(f'{thing}_pics/pic_{i}.npy') for i in range(num_files)])


def shuffle(array,labels):
    old_array = list(array)
    old_labels = list(labels)
    length = len(old_array)
    new_array = []
    new_labels = []
    num = random.randint(0,len(old_array)-1)
    #num_lst = []
    for i in range(length):
        #while num in num_lst:
        num = random.randint(0,len(old_array)-1)
        #num_lst.append(num)
        new_array.append(old_array.pop(num))
        new_labels.append(old_labels.pop(num))
        print(i,end="\r")
    new_array = np.array(new_array)
    new_labels = np.array(new_labels)
    return new_array,new_labels

print("shuffling")
new_ims,new_labels = shuffle(imgs,labels)
print(new_ims.shape)
print(new_labels.shape)
print("saving")
np.save(f"{thing}_labels.npy",new_labels)
for i in range(len(new_ims)):
    np.save(f"{thing}_pics/pic_{i}",new_ims[i])