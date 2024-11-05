import json
import os
import pprint
import random as rd

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_pt_files(root, ids):
    
    filenames = [root + i + '.pt' for i in ids]

    combined_data = []

    for file in filenames:
        try:
            data = torch.load(file, map_location=device)
            combined_data.extend(data)
        except:
            FileNotFoundError

    torch.save(combined_data, f'{root}_combined.pt')

def read_pt_file(data_path):
    data = torch.load(data_path, map_location=device)
    return data

def create_mock_dataset():
    episode_list = []
    steps_list = []
    step_dicti = {
            "step": 0,
            "position": None,
            "tensors": None,
            "is_stuck": False,
        }

    episode_list.append(steps_list)
    steps_list.append(step_dicti)

    data = read_pt_file()

    print(data[0]['tensors'][0])
    print(data[0]['step'])
    print(data[0]['position'])
    print(data[0]['is_stuck'])


    '''for i in range(len(data)):
        step_dicti['tensors'] = data[i][0]
        step_dicti['gradients'] = data[i][1]
    '''

def sample(data, splitting_factor=0.1):
    # sample n_samples from the data
    d_sample = rd.sample(data, int(splitting_factor*len(data)))
    return d_sample

def split_on_label(data, label=None):

    pos_label = []
    neg_label = []

    for episode in data:
        for step_data in episode:
            if step_data[label]:
                pos_label.append(step_data)
            else:
                neg_label.append(step_data)
    
    return pos_label, neg_label
    

def plot_movement():
    # draw the maze and plot the counts of times the agent has been stuck at each position as a heatmap
    pass

def cosine_similarity(tensor_list):
    num_heads = len(tensor_list[0])
    sim = {head : [] for head in range(num_heads)}
    print(sim.keys())

    cosi = torch.nn.CosineSimilarity(dim=0) 

    for head, idx in enumerate(range(num_heads)):
        head_tensors = [tensor[head].flatten() for tensor in tensor_list]
        print(f'length of head tensor {idx}', len(head_tensors))
        for i in range(len(head_tensors)):
            for j in range(i+1, len(head_tensors)):
                sim[head].append(cosi(head_tensors[i], head_tensors[j]))
    
    # print('head_tensor:', head_tensors[0])
    # print(len(head_tensors))
    # print(len(sim[0]))

    # compute the average cosine similarity for each head
    avg_sim = {head : sum(sim[head])/len(sim[head]) for head in sim.keys()}
    print('average similarity:', avg_sim)
    # compute the overall average cosine similarity
    # avg_sim = sum([avg_sim[head] for head in avg_sim.keys()])/len(avg_sim.keys())
    # print('average overall similarity:', avg_sim)

    return avg_sim


def main():

    block = "block_3"
    root = f'./grad_sam/map_open_doors_90_degrees/{block}/visionary-hill-816'
    ids = ['_10', '_20', '_30', '_40', '_50', '_60', '_70', '_80', '_90', '_100']
    data_path = f'{root}_combined.pt'

    # agent/grad_sam/map_open_doors_90_degrees/block_2

    if not os.path.exists(data_path):
        data_path = combine_pt_files(root, ids)

    data = read_pt_file(f'./grad_sam/map_open_doors_90_degrees/{block}/visionary-hill-816_combined.pt')
    
    # divide the data into stuck and not stuck
    pos_label, neg_label = split_on_label(data, label='is_stuck')
    print('positives:', len(pos_label), 'negatives:', len(neg_label))

    # make the number of positive and negative samples equal
    if len(pos_label) > len(neg_label):
        pos_label = sample(pos_label, splitting_factor=len(neg_label)/len(pos_label))
    elif len(neg_label) > len(pos_label):
        neg_label = sample(neg_label, splitting_factor=len(pos_label)/len(neg_label))

    print('positives:', len(pos_label), 'negatives:', len(neg_label))

    # sample 10% of the data
    pos_label = sample(pos_label, splitting_factor=0.1)
    neg_label = sample(neg_label, splitting_factor=0.1)

    # get the tensors from the samples
    pos_tensors = [step_data['tensors'] for step_data in pos_label]
    neg_tensors = [step_data['tensors'] for step_data in neg_label]
    all_tensors = pos_tensors + neg_tensors

    print('positive samples:', len(pos_tensors), 'negative samples:', len(neg_tensors))

    # cosine_similarity(pos_tensors)

    avg_pos_sim = cosine_similarity(pos_tensors)
    avg_neg_sim = cosine_similarity(neg_tensors)
    avg_all_sim = cosine_similarity(all_tensors)

    print('average positive similarity:', avg_pos_sim)
    print('average negative similarity:', avg_neg_sim)
    print('average all similarity:', avg_all_sim)

    # write to csv file
    with open(f'{root}_cosine_similarity.csv', 'w') as f:
        f.write('average positive similarity, average negative similarity, average all similarity\n')
        f.write(f'{avg_pos_sim}, {avg_neg_sim}, {avg_all_sim}')

if __name__ == '__main__':
    main()