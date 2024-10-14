import numpy as np
import torch
from scipy.special import softmax

from agent.dtqn_agent import DTQN_Agent
from env.sunburstmaze_discrete import SunburstMazeDiscrete
from utils.state_preprocess import state_preprocess

device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Was faster with cpu??? Loading between cpu and mps is slow maybe


def generate_q_values(env: SunburstMazeDiscrete, model):

    orientation_range = [0, 1, 2, 3]  # north, east, south, west

    x_range = env.initial_map.shape[1]
    y_range = env.initial_map.shape[0]

    q_val_list_to_position = []
    print("y-range", y_range)

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
                left = q_list[(orientation + 1) % 4][1] / 2
                right = q_list[orientation - 1][2] / 2
                q_value_sum = forward + left + right
                q_value_list[orientation] = q_value_sum
            # print('q_value_list', q_value_list)
            q_value_list = softmax(q_value_list)
            dicti = {(y, x): q_value_list}
            q_val_list_to_position.append(dicti)
            # print('q_value_list', q_value_list)

            # print('q_list', q_list)

    return q_val_list_to_position
