import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(time.time()))
print(formatted_time)

print("loading arrays...")

# Get the number of files
files = os.listdir('screenshot_files')
num_files = len(files)

#load files

train_ratio = 0.8
train_num = int(num_files*train_ratio)

imgchunks_train = [np.load(f'screenshot_files/screenshots_{i}.npy') for i in range(train_num)]
imgchunks_val = [np.load(f'screenshot_files/screenshots_{i}.npy') for i in range(train_num,num_files)]

outputchunks_train = [np.load(f'output_files/outputs_{i}.npy') for i in range(train_num)]
outputchunks_val = [np.load(f'output_files/outputs_{i}.npy') for i in range(train_num,num_files)]

train_len = len(imgchunks_train)
val_len = len(imgchunks_val)

class My_Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self,batch_type,num_of_batches):
        self.batch_type = batch_type
        self.num_of_batches = num_of_batches
    
    def __len__(self):
        return self.num_of_batches

    def __getitem__(self,idx):
        if self.batch_type == "train":
            x_batch = np.reshape(imgchunks_train[idx],(-1,720,1280,4))
            y_batch = np.array(outputchunks_train[idx])
            return [x_batch,y_batch]
        if self.batch_type == "valid":
            x_batch = np.reshape(imgchunks_val[idx],(-1,720,1280,4))
            y_batch = np.array(outputchunks_val[idx])
            return [x_batch,y_batch]
print("creating model")
# Set up the input layer
'''
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(720,1280,4)))
model.add(tf.keras.Conv2D(32, 5, activation='relu'))
model.add(tf.keras.MaxPooling2D((2,2)))
model.add(tf.keras.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.MaxPooling2D((2,2)))
model.add(tf.keras.Flatten())
model.add(tf.keras.Dense(128, activation='relu'))
model.add(tf.keras.Dropout(0.2))
model.add(tf.keras.Dense(64, activation='relu'))
model.add(tf.keras.Dropout(0.2))
model.add(tf.keras.Dense(16, activation='softmax', name='keyboard'))
model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
'''
input_layer = tf.keras.layers.Input(shape=(720,1280,4))
print("input done")

# Add some convolutional layers
x = tf.keras.layers.Conv2D(16, 5, activation='tanh')(input_layer)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, 5, activation='tanh')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(64, 3, activation='tanh')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
print("conv done")

# Flatten the output of the convolutional layers
x = tf.keras.layers.Flatten()(x)
print("flatten done")

# Add some dense layers
#x = tf.keras.layers.Dense(128, activation='tanh')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='tanh')(x)
x = tf.keras.layers.Dropout(0.2)(x)
print("dense done")

# Output layer for keyboard actions
output_layer = tf.keras.layers.Dense(12, activation='tanh', name='output')(x)

print("creating model")
# Create the model
model = tf.keras.Model(input_layer, output_layer)

# Compile the model
print("compiling model")
model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
print("model created")

# Train the model
print("training model")
#trained = model.fit(imgchunks, [boolean_values, linear_values], epochs=10, batch_size=2, validation_split=0.2)
my_training_batch_generator = My_Custom_Generator("train",train_len)
my_validation_batch_generator = My_Custom_Generator("valid",val_len)
tf.compat.v1.enable_eager_execution()
trained = model.fit(my_training_batch_generator, epochs=10, verbose=1, validation_data=my_validation_batch_generator)

# Save the trained model
print("saving model")
model.save("model.h5")

formatted_time2 = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(time.time()))
print(formatted_time2)

# Plot the training and validation loss
print("plotting")
plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot the training and validation accuracy
plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Keyboard accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
#