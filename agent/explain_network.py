import os
import numpy as np
import torch
from dqn_agent import DQN_Agent
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
from env.sunburstmaze_discrete import SunburstMazeDiscrete
from utils.state_preprocess import state_preprocess
import math

device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Was faster with cpu??? Loading between cpu and mps is slow maybe


def generate_q_values(env:SunburstMazeDiscrete, model):

    orientation_range = [0, 1, 2, 3] # north, east, south, west

    x_range = env.initial_map.shape[1]
    y_range = env.initial_map.shape[0]
    

    q_val_list_to_position = []

    for x in range(x_range):
        for y in range(y_range):
            if env.env_map[y, x] == 1:
                continue
            env.position = (y, x)
            q_list = []
            q_value_list = np.zeros(4)
            for orientation in orientation_range:
                env.orientation = orientation
                state = env._get_observation()
                state = state_preprocess(state, device=device)
                q_values = model(state).detach().numpy()
                q_list.append(q_values)
            for orientation in orientation_range:
                forward = q_list[orientation][0]
                left = q_list[(orientation + 1) % 4][1] /2
                right = q_list[orientation - 1][2] / 2
                q_value_sum = forward + left + right
                q_value_list[orientation] = q_value_sum

            q_value_list = softmax(q_value_list)
            dicti = {(y, x): q_value_list}
            q_val_list_to_position.append(dicti)


    return q_val_list_to_position



def compare_model_q_values(agent: DQN_Agent, env: SunburstMazeDiscrete):
    # iterate through each model in the folder "model" and compare the q-values of the models
    # for each position in the maze
    # save the q-values in a dictionary with the position as the key and the q-values as the value
    # return the dictionary
  
    # Load the models
    for model in os.listdir('model/feed_forward/lunar-darkness-740'):
        agent.model.load_state_dict(torch.load(model))
        agent.model.eval()
        q_values = generate_q_values(agent.env, agent.model)
        # Save the q-values in a dictionary
        # return the dictionary


# Grad-SAM
def grad_sam(attention_weights, gradients, block=0):

    # Apply ReLU to the gradients
    fig, axes = plt.subplots(3, math.ceil(len(attention_weights)/3), figsize=(30, 10))
    fig.suptitle(f"Grad-SAM for block {block}")
    for i in range(len(attention_weights)):
        grad = torch.relu(gradients[i])
        att_w = attention_weights[i].squeeze(0)
        grad = grad.squeeze(0)
        print("Att_w:", att_w.shape, type(att_w))
        print(grad.shape, type(grad))
        # Multiply the gradients with the attention weights
        grad_sam = torch.tensordot(att_w, grad)
        # Show the grad-sam as a heatmap
        sns.heatmap(grad_sam, ax=axes[math.floor(i/3), i%3])
        axes[math.floor(i/3), i%3].set_title(f"Attention head {i}")

    
    plt.show()
    
    

# concepts 

'''
- Toril sees goal
- Toril sees wall
- Toril sees open space
- Toril is spinning
- Toril moves away from goal
- Toril moves towards goal

'''

