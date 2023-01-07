import numpy as np
labels = np.load("dmg_labels.npy")
labels = labels[0:len(labels)-1]
np.save("dmg_labels.npy",labels)