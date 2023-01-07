import numpy as np
import matplotlib.pyplot as plt
import os
import random

thing = "dmg"

files = os.listdir(f'{thing}_pics')
num_files = len(files)

labels = np.load(f"{thing}_labels.npy")
num_lst = []
num = 10
image = np.load(f"{thing}_pics/pic_{num}.npy")

# Print the label
print(f"{labels[num]} {num}")

# Display the image
plt.imshow(image)
plt.show()

for i in range(10):
    #while num in num_lst:
    #    num = random.randint(0,num_files)
    #num_lst.append(num)
    # Load the image from the file
    image = np.load(f"{thing}_pics/pic_{i}.npy")

    # Print the label
    print(f"{labels[i]} {i}",end="\r")

    # Display the image
    plt.imshow(image)
    plt.show()