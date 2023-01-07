import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

files = os.listdir('dmg_pics')
num_files = len(files)

thing = "dmg"
model_num = 35
num_arr = []

print("loading files")

# Load the numpy arrays from file "dmg_pics"
pics = [np.load(f"{thing}_pics/pic_{i}.npy") for i in range(num_files)]

# Load the numpy array of labels from "dmg_labels.npy"
labels = [np.load(f"{thing}_labels.npy")]

print("loading model")

# Load the neural network model using tensorflow from "dmg_model.h5"
model = tf.keras.models.load_model(f"{thing}_models/{thing}_model{model_num}.h5")

# Initialize the list for unmatched predictions
unmatched_lst = []

# Loop through each numpy array in "dmg_pics"
for i, pic in enumerate(pics):
    print(i)
    # Predict an outcome using the loaded model
    prediction = model.predict(np.array([pic]))

    # Get the index of the max of the prediction
    max_index = np.argmax(prediction[0])

    # Get the index of the number "1" in the corresponding label
    label_index = np.argmax(labels[0][i])

    # Compare the two indexes
    if max_index != label_index:
    # If they don't match, add the number of the prediction to the list
        unmatched_lst.append(i)
print(unmatched_lst)
print(len(unmatched_lst))
# The list "unmatched_lst" now contains the numbers of the predictions that did not match the corresponding labels
for i in unmatched_lst:
    image = pics[i]

    # Print the label
    print(f"{labels[0][i]} {i}")

    # Display the image
    plt.imshow(image)
    plt.show()

    num = int(input("Enter correct number or enter 4 for delete: "))
    if num == 4:
        num_arr.append(i)
    elif num == 0:
        labels[0][i] = [1,0,0,0]
    elif num == 1:
        labels[0][i] = [0,1,0,0]
    elif num == 2:
        labels[0][i] = [0,0,1,0]
    elif num == 3:
        labels[0][i] = [0,0,0,1]
    # Print the label
    print(f"{labels[0][i]} {i}")
    print("-------------")

offset = 0
labels = list(labels[0])

if len(num_arr) > 0:
    for i in num_arr:
        labels.pop(i-offset)
        pics.pop(i-offset)
        offset += 1

np.save(f"{thing}_labels.npy",np.array(labels))
for i in range(len(pics)):
    np.save(f"{thing}_pics/pic_{i}",pics[i])