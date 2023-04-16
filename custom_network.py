from keras.layers import Conv2D, Dense, Flatten, Input
import tensorflow as tf
import matplotlib.pyplot as plt
print("loading custom environment...")
from custom_env import OW2Env
import pygame
import numpy as np
import keyboard

# Set up the environment
print("making env...")
save_num = int(input("Enter the save number of the model: "))
max_steps = int(input("Enter max env steps: "))


env = OW2Env(0,3,6,10,0.05,enable_movement=False) # enter all reward values as positive. health and miss rewards will be subtracted, all others will be added

action_size = env.action_space.shape[0]
input_shape = env.observation_space.shape

# Define the hyperparameters
learning_rate = 5e-4
discount_factor = 0.95
num_episodes = 1000

load_model = input("Load model? [Y/n]: ").lower() == 'y'

if load_model == False:
    # Define the CNN architecture
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(action_size, activation='sigmoid')(x)

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Compile the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['mae'])
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

def get_action_indices(actions):
    ret_arr = [0 for i in range(len(actions))]
    if action[0] >= action[1]:
        ret_arr[0] = 1
    else:
        ret_arr[1] = 1
    if action[2] >= action[3]:
        ret_arr[2] = 1
    else:
        ret_arr[3] = 1
    if action[4] >= action[5]:
        ret_arr[4] = 1
    else:
        ret_arr[5] = 1
    if action[6] >= action[7]:
        ret_arr[6] = 1
    else:
        ret_arr[7] = 1
    if action[8] >= action[9]: #14,15
        ret_arr[8] = 1
    else:
        ret_arr[9] = 1
    if action[10] >= action[11]: #16,17
        ret_arr[10] = 1
    else:
        ret_arr[11] = 1
    ret_arr[12] = 1
    ret_arr[13] = 1
    ret_arr[14] = 1
    return ret_arr

show_flag = False
continue_flag = True
keyboard.hook_key('`', show_graph)

# Train the Q-network using Q-learning
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        # Increment step counter
        steps += 1

        # Get the Q-values for the current state
        q_values = model.predict_on_batch(np.array([state]))[0]
        
        # Choose an action using an epsilon-greedy policy
        epsilon = 1.0 / ((episode / 50) + 10)
        for i in range(len(q_values[:-1])):
            if i%2 == 0:
                if np.random.uniform() < epsilon:
                    q_values[i],q_values[i+1] = q_values[i+1],q_values[i]
        #if np.random.uniform() < epsilon:
        #    action = env.action_space.sample()
        #else:
        #    action = np.argmax(q_values)
        action = q_values
            
        # Take the chosen action and observe the new state and reward
        next_state, reward, done, _ = env.step(action)
        print(steps)
        
        # add current reward to total
        total_reward += reward
        
        # Update the Q-value(s) for the current state and action
        q_values_target = np.copy(q_values)
        current_pred = model.predict(np.array([next_state]))[0]
        update_arr = get_action_indices(q_values)
        for i in range(len(action)):
            if update_arr[i] == 1:
                q_values_target[i] = reward + discount_factor * np.max(current_pred)

        # only for one q-value at a time
        #q_values_target[action] = reward + discount_factor * np.max(model.predict(np.array([next_state]))[0])
        
        # Train the Q-network using the Q-learning update rule
        with tf.GradientTape() as tape:
            predictions = model(np.array([state]))[0]
            loss = loss_fn(q_values_target, predictions)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        state = next_state

        if show_flag:
            # clear previous points
            plt.clf()

            # plot the points
            try:
                plt.plot(plot_data[:, 0], plot_data[:, 1], 'ro', label="total reward") # total reward
            except NameError:
                continue_flag = False
            if continue_flag:
                show_steps = True if input("Show steps? [Y/n]: ").strip().lower() == 'y' else False
                if show_steps:
                    plt.plot(plot_data[:, 0], plot_data[:, 2], 'go', label="steps ") # steps
                #plt.plot(plot_data[:, 0], plot_data[:, 3], 'bo', label="average reward") # average reward

                # add axis labels and title
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Training Stats')

                # add a legend
                plt.legend()

                # show the plot
                plt.show()
            show_flag = False
            continue_flag = True

    # Plot stuff
    try:
        plot_data = np.append(plot_data,np.array([[episode, total_reward, steps, total_reward/steps]]),axis=0)
    except Exception as e:
        plot_data = np.array([[episode, total_reward, steps, total_reward/steps]])

    # Print the episode score
    print(f"Episode {episode} | Total Reward = {total_reward} | Steps = {steps}")
    with open(f"models/custom_models/model_{save_num}info.txt","w",encoding="utf-8") as f:
        f.write(f"Episode {episode}\nTotal Reward = {total_reward}\nSteps = {steps}\nAverage Reward = {total_reward/steps:.3f}")
    model.save(f"models/custom_models/model_{save_num}.h5")
    
env.close()
