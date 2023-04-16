# SCAI

Models are currently not available as they are too large to upload, once they are ready I will add a link in this README to the models.
Right now model_info.txt files contain information on the models, so if you are interested in the progress you can read those.

get_experience.py gets (screen, action) pairs and saves them to numpy files in the experience folder.
replay.py uses the numpy files saved in the experience folder to learn from a human player's actions.
these two are the first step in the training process

custom_env.py provides custom_network.py with a way to interact with the game (Overwatch 2).
custom_network.py creates and trains the model using custom_env.py to interact with the game in real time.

dmg_model.h5 and health_model.h5 are used to get rewards for the RL process.
