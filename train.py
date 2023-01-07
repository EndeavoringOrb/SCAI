import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.init(project="SCAI_Health")

files = os.listdir('health_pics')
num_files = len(files)

#files = os.listdir('dmg_models')
#num_files2 = len(files)

input_layer = tf.keras.layers.Input(shape=(19,12,1))
#x = tf.keras.layers.Conv2D(config.conv1_filters, config.conv1_kernelsize, activation=config.activation_1)(input_layer)
#x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Flatten()(input_layer)
x = tf.keras.layers.Dense(wandb.config.dense1_size, activation=wandb.config.activation_3)(x)
x = tf.keras.layers.Dropout(wandb.config.dropout)(x)
x = tf.keras.layers.Dense(wandb.config.dense2_size, activation=wandb.config.activation_4)(x)
x = tf.keras.layers.Dropout(wandb.config.dropout)(x)
output_layer = tf.keras.layers.Dense(11, activation='softmax', name='number')(x)

model = tf.keras.Model(input_layer, output_layer)
'''
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=wandb.config.initial_learning_rate,  # initial learning rate
    decay_steps=wandb.config.decay_steps,  # number of epochs over which to decay the learning rate
    decay_rate=wandb.config.decay_rate,  # decay rate
    staircase=wandb.config.staircase  # whether to apply decay in a staircase fashion
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
'''
#model.compile(optimizer=optimizer, loss=wandb.config.loss, metrics=wandb.config.metric)

model.compile(optimizer='sgd', loss=wandb.config.loss, metrics=wandb.config.metric)

#load data
images = np.array([np.load(f'health_pics/pic_{i}.npy') for i in range(num_files)])
labels = np.array(np.load('health_labels.npy'))

# Define a callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=wandb.config.min_delta, patience=10)

# Set the random seed
tf.random.set_seed(42)

trained = model.fit(x=images, y=labels, epochs=wandb.config.epoch, batch_size=wandb.config.batch_size, verbose=wandb.config.verbose, validation_split=wandb.config.validation_split, callbacks=[WandbMetricsLogger(log_freq=wandb.config.log_freq),WandbModelCheckpoint("models"),early_stopping], shuffle=True)

model.save(f"dmg_mode0.h5")

wandb.finish()