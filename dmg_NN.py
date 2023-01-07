import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.init(
    project="SCAI_Damage",
    # track hyperparameters and run metadata with wandb.config
    config={
        "conv1_filters": 32,
        "conv1_kernelsize": 3,
        "activation_1": "linear",

        "dense1_size": 128,
        "activation_3": "linear",

        "dense2_size": 64,
        "activation_4": "linear",

        "dense3_size": 32,
        "activation_5": "sigmoid",

        #learning rate
        "initial_learning_rate": 0.001,  # initial learning rate
        "decay_steps": 100, # number of epochs over which to decay the learning rate
        "decay_rate": 0.95,  # decay rate
        "staircase": False,

        "dropout": 0.1,
        "optimizer": "sgd",
        "loss": "categorical_crossentropy", # mean_squared_error
        "metric": "accuracy",
        "epoch": 10000,
        "batch_size": 512,
        "verbose": 1, # 0,1,2
        "validation_split": 0.2,
        #early stopping
        "min_delta": 0.001,
        "patience": 100,
        #logging
        "log_freq": 100 # logs every _ batches
    }
)
'''
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
    }
}  
'''
#entity = endeavoringorb
#project = SCAI_Damage
#sweep_ID = return value from wandb.sweep()
config = wandb.config
#sweep_id = wandb.sweep(sweep=sweep_configuration, project="SCAI_Damage")
#wandb.agent(sweep_id, count=10)

files = os.listdir('dmg_pics')
num_files = len(files)

files = os.listdir('dmg_models')
num_files2 = len(files)

input_layer = tf.keras.layers.Input(shape=(70,70,4))
x = tf.keras.layers.Conv2D(config.conv1_filters, config.conv1_kernelsize, activation=config.activation_1)(input_layer)
x = tf.keras.layers.MaxPooling2D((2,2))(x) 
x = tf.keras.layers.Flatten()(input_layer)
x = tf.keras.layers.Dense(config.dense1_size, activation=config.activation_3)(x)
x = tf.keras.layers.Dropout(config.dropout)(x)
x = tf.keras.layers.Dense(config.dense2_size, activation=config.activation_4)(x)
x = tf.keras.layers.Dropout(config.dropout)(x)
x = tf.keras.layers.Dense(config.dense3_size, activation=config.activation_5)(x)
x = tf.keras.layers.Dropout(config.dropout)(x)
output_layer = tf.keras.layers.Dense(4, activation='softmax', name='number')(x)

model = tf.keras.Model(input_layer, output_layer)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.initial_learning_rate,  # initial learning rate
    decay_steps=config.decay_steps,  # number of epochs over which to decay the learning rate
    decay_rate=config.decay_rate,  # decay rate
    staircase=config.staircase  # whether to apply decay in a staircase fashion
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss=config.loss, metrics=config.metric)

#model.compile(optimizer='sgd', loss=config.loss, metrics=config.metric)

#load data
images = np.array([np.load(f'dmg_pics/pic_{i}.npy') for i in range(num_files)])
labels = np.array(np.load('dmg_labels.npy'))

# Define a callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=config.min_delta, patience=config.patience)

# Set the random seed
tf.random.set_seed(42)

trained = model.fit(x=images, y=labels, epochs=config.epoch, batch_size=config.batch_size, verbose=config.verbose, validation_split=config.validation_split, callbacks=[WandbMetricsLogger(log_freq=config.log_freq),early_stopping], shuffle=False) # WandbMetricsLogger(log_freq=config.log_freq),WandbModelCheckpoint("models"),

model.save(f"dmg_models/dmg_model{num_files2+34}.h5")


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
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
#


wandb.finish()