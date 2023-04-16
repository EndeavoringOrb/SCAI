from stable_baselines3 import PPO, DQN
import os
from custom_env import OW2Env
import time
from time import sleep

models_dir = f"SB3models/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

print("making env...")
sleep(5)
env = OW2Env()
steps = 2
model = PPO('CnnPolicy', env, verbose=2, n_steps=steps, batch_size=steps)

TIMESTEPS = 1000
iters = 0
while iters < 1:
	print(iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
	print("saving")
	model.save(f"ppo_model/{TIMESTEPS*iters}.h5")