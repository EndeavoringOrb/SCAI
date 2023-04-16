from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keyboard
import mouse

# Set up the environment
print("making env...")
save_num = int(input("Enter the save number of the model: "))
max_steps = int(input("Enter max env steps: "))
health = 255

dmgmodel = tf.keras.models.load_model('dmg_model.h5') # 2,519,396 params for RGBA, 1,892,196 params for RGB, ONLY MODEL NUMBER 37 AND LATER IS COMPATIBLE
healthmodel = tf.keras.models.load_model('health_model.h5') #105,119

def convert_to_int(array):
    array = list(array) #array[0]
    largest_val = max(array)
    number = array.index(largest_val)
    if number == 10:
        return "N"
    return number

def mouse_callback(e):
    mouse_events.append(e)

def keyboard_callback(e):
    keyboard_events.append(e)

# Define the hyperparameters
input_shape = (720, 1280, 3)
action_size = 15
learning_rate = 1e-3
discount_factor = 0.95
num_episodes = 1000
batch_size = 1

load_model = input("Load model? [Y/n]: ").lower() == 'y'

if load_model == False:
    # Define the CNN architecture
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    #x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu')(x)
    #x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(action_size, activation='sigmoid')(x)

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Compile the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['mae','accuracy'])
else:
    # Load the saved model from the file path
    model_path = input("Enter relative model path: ")  # Replace with the file path to your saved model
    model = tf.keras.models.load_model(model_path)

    network_shape = []
    for layer in model.layers:
        try:
            network_shape.append(layer.output_shape[1])
        except IndexError:
            network_shape.append(layer.output_shape[0][1])
    network_shape = network_shape[1:-1]

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

model.summary()

def nice_list_to_str(lst):
    ret_str = str(lst[0])
    for item in lst[1:]:
        ret_str += f", {item}"
    return ret_str

def show_graph(key):
    global show_flag
    show_flag = True

def convert_action_to_str(action):
    print(action)
    ret_str = ""
    if action[0] > action[1]:
        ret_str += "W "
    if action[2] > action[3]:
        ret_str += "A "
    if action[4] > action[5]:
        ret_str += "S "
    if action[6] > action[7]:
        ret_str += "D "
    if action[8] > action[9]:
        ret_str += "R "
    if action[10] > action[11]:
        ret_str += "LMouse "
    ret_str += f"X:{action[12]} "
    ret_str += f"Y:{action[13]} "
    ret_str += f"Duration:{action[14]}"
    return ret_str


def load_experience(mod_num,batch_size=1, save_chunk_size=10, test_num=-1):
    # Compute the range of experiences to load based on the batch 
    num = 0
    while True:
        if test_num != -1:
            num,test_num = test_num,num
        load_nums = []
        files_to_load = []
        for i in range(num%mod_num-(num%mod_num)%save_chunk_size,(num%mod_num+batch_size)-(num%mod_num+batch_size)%save_chunk_size+1,10):
            files_to_load.append(f"experience/{i}-{i+save_chunk_size}.npy")
            load_nums.append((i,i+save_chunk_size))
        
        # Load the numpy arrays from disk
        exp_data = []
        for file in files_to_load:
            try:
                exp_data.extend(np.load(file,allow_pickle=True))
            except FileNotFoundError:
                for i in range(9):
                    file = f"experience/{load_nums[-1][0]}-{load_nums[-1][0]+i+1}.npy"
                    try:
                        exp_data.extend(np.load(file,allow_pickle=True))
                        break
                    except FileNotFoundError:
                        continue

        exp_data = np.array(exp_data[num%mod_num-load_nums[0][0]:num%mod_num+batch_size-load_nums[0][0]], dtype=object)
        
        # Return the experience data as a list of numpy arrays
        x_data = np.array([exp_data[0,0]])
        y_data = np.array([exp_data[0,1]])
        ret_val = (x_data, y_data)
        if test_num != -1:
            num,test_num = test_num,num
        else:
            num += 1
        if y_data.shape[1] == 15:
            #print(y_data)
            yield ret_val
        else:
            print(f"\nERROR: WRONG ACTION SHAPE {y_data.shape} PASSED")
            print(convert_action_to_str(y_data[0]))
            print("")

show_flag = False
continue_flag = True
keyboard.hook_key('`', show_graph)
keyboard_events = []
mouse_events = []
#load_experience(182,3,10)
keyboard.hook(keyboard_callback)
mouse.hook(mouse_callback)

find_mse = lambda x,y: np.mean((x - y) ** 2)
mses = []
moving_averages = []
num_epochs = 1000
train_gen = load_experience(max_steps)

for i in range(num_epochs):
    a = model.fit(train_gen,
          steps_per_epoch=max_steps,
          epochs=1)
    print("epochs:",i+1,"done.")
    history = a.history

    with open(f"models/custom_models/model_{save_num}info.txt","w",encoding="utf-8") as f:
        f.write(f"Epochs Ran: {i+1}\nSteps Ran: {(i+1)*max_steps}\nLoss = {history['loss'][0]:.4f}\nMean Absolute Error = {history['mae'][0]:.4f}\nAccuracy = {history['accuracy'][0]:.4f}")
    model.save(f"models/custom_models/model_{save_num}.h5")


'''
mse = find_mse(action,np.array(predictions))
print("mse: ", mse)
mses.append(mse)
mses = mses[-40:]
moving_avg = np.mean(mses)
print("40 moving average mse:", moving_avg)
moving_averages.append(moving_avg)

if show_flag:
    # clear previous points
    plt.clf()

    # plot the points
    plt.plot([i for i in range(len(moving_averages))], moving_averages, 'ro', label="total reward") # total reward
    #plt.plot([i for i in range(len(mses))], mses, 'ro', label="total reward") # total reward
    # add axis labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Stats')

    # add a legend
    plt.legend()

    # show the plot
    plt.show()
    show_flag = False'''