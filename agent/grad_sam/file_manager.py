import json
import os
import pprint
import random as rd

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_pt_files(path, ids):
    filenames = [path + i + '.pt' for i in ids]

    combined_data = []

    for file in filenames:
        try:
            data = torch.load(file, map_location=device)
            combined_data.extend(data)
        except:
            FileNotFoundError

    torch.save(combined_data, f'{path.split('/')[-2]}_combined.pt')

def get_combined_files(path):
    combined_files = []

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_combined.pt'):
                # print(os.path.join(subdir, file))
                combined_files.append(os.path.join(subdir, file))
            
    
    # sort the files by name
    combined_files.sort()
    
    return combined_files

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

def cosine_similarity(tensor_list):
    num_heads = len(tensor_list[0])
    sim = {head : [] for head in range(num_heads)}
    # print(sim.keys())

    cosi = torch.nn.CosineSimilarity(dim=0) 

    for head, idx in enumerate(range(num_heads)):
        head_tensors = [tensor[head].flatten() for tensor in tensor_list]
        # print(f'Head {head}, shape after flattening:', head_tensors[0].shape)  # Debugging shape
    
        #print(f'length of head tensor {idx}', len(head_tensors))
        for i in range(len(head_tensors)):
            for j in range(i+1, len(head_tensors)):
                sim[head].append(cosi(head_tensors[i], head_tensors[j]))
    
    # print('head_tensor:', head_tensors[0])
    # print(len(head_tensors))
    # print(len(sim[0]))

    # compute the average cosine similarity for each head
    avg_sim = {head : sum(sim[head])/len(sim[head]) for head in sim.keys()}
    # print('average similarity:', avg_sim)
    # compute the overall average cosine similarity
    # avg_sim = sum([avg_sim[head] for head in avg_sim.keys()])/len(avg_sim.keys())
    # print('average overall similarity:', avg_sim)

    return avg_sim

def cosine_similarity_for_both_classes(pos_data, neg_data):
    num_heads = len(pos_data[0])
    sim = {head : [] for head in range(num_heads)}
    # print(sim.keys())

    cosi = torch.nn.CosineSimilarity(dim=0) 

    for head, idx in enumerate(range(num_heads)):
        pos_head_tensors = [tensor[head].flatten() for tensor in pos_data]
        neg_head_tensors = [tensor[head].flatten() for tensor in neg_data]
        # print(f'Head {head}, shape after flattening:', head_tensors[0].shape)  # Debugging shape
    
        #print(f'length of head tensor {idx}', len(head_tensors))
        for i in range(len(pos_head_tensors)):
            for j in range(len(neg_head_tensors)):
                sim[head].append(cosi(pos_head_tensors[i], neg_head_tensors[j]))
    
    # print('head_tensor:', head_tensors[0])
    # print(len(head_tensors))
    # print(len(sim[0]))

    # compute the average cosine similarity for each head
    avg_sim = {head : sum(sim[head])/len(sim[head]) for head in sim.keys()}
    # print('average similarity:', avg_sim)
    # compute the overall average cosine similarity
    # avg_sim = sum([avg_sim[head] for head in avg_sim.keys()])/len(avg_sim.keys())
    # print('average overall similarity:', avg_sim)

    return avg_sim

def pre_process(data):
    pos_label, neg_label = split_on_label(data, label='is_stuck')

    print('Positives (before sampling):', len(pos_label))
    print('Negatives (before sampling):', len(neg_label))

    # make the number of positive and negative samples equal
    if len(pos_label) > len(neg_label):
        pos_label = sample(pos_label, splitting_factor=len(neg_label)/len(pos_label))
        print('Reduced positives to:', len(pos_label))
    elif len(neg_label) > len(pos_label):
        neg_label = sample(neg_label, splitting_factor=len(pos_label)/len(neg_label))
        print('Reduced negatives to:', len(neg_label))

    # sample 10% of the data
    pos_label = sample(pos_label, splitting_factor=0.1)
    neg_label = sample(neg_label, splitting_factor=0.1)

    print('Positives (after sampling):', len(pos_label))
    print('Negatives (after sampling):', len(neg_label))

    return pos_label, neg_label

def plot_similarity(block_nr, avg_pos_sim, avg_neg_sim, avg_all_sim):


    # Extract the number of heads dynamically (8 in this case)
    n_groups = len(avg_pos_sim)

    # Convert tensors to list of values for plotting
    pos = [tensor.item() for tensor in avg_pos_sim.values()]
    neg = [tensor.item() for tensor in avg_neg_sim.values()]
    all_ = [tensor.item() for tensor in avg_all_sim.values()]

    # Create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)  # Index based on number of heads
    bar_width = 0.25
    opacity = 0.8

    # Plot each bar group
    rects1 = plt.bar(index, pos, bar_width, alpha=opacity, color='b', label='Positives')
    rects2 = plt.bar(index + bar_width, neg, bar_width, alpha=opacity, color='g', label='Negatives')
    rects3 = plt.bar(index + 2 * bar_width, all_, bar_width, alpha=opacity, color='r', label='All')

    # Labels and Title
    plt.xlabel('Heads')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity of Grad-SAM for {block_nr}')
    plt.xticks(index + bar_width, [f'Head {i}' for i in range(n_groups)])  # Dynamic head labels
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_line_graph(block_nr, avg_pos_sim, avg_neg_sim, avg_all_sim):
        # Extract the number of heads dynamically (8 in this case)
    n_groups = len(avg_pos_sim)

    # Convert tensors to list of values for plotting
    pos = [tensor.item() for tensor in avg_pos_sim.values()]
    neg = [tensor.item() for tensor in avg_neg_sim.values()]
    all_ = [tensor.item() for tensor in avg_all_sim.values()]

    # Create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)  # Index based on number of heads

    # Plot each line with markers
    plt.plot(index, pos, marker='o', linestyle='-', color='b', label='Positives')
    plt.plot(index, neg, marker='o', linestyle='-', color='g', label='Negatives')
    plt.plot(index, all_, marker='o', linestyle='-', color='black', label='All')

    # Labels and Title
    plt.xlabel('Heads')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity of Grad-SAM for {block_nr}')
    plt.xticks(index, [f'Head {i}' for i in range(n_groups)])  # Dynamic head labels
    plt.xlim(0, n_groups - 1)  # Adjust x-axis limits
    plt.ylim(0.2, 0.8)
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    return pos, neg, all_

def plot_clusters(idx, data_for_clustering, labels, n_clusters):

    # Use UMAP for dimensionality reduction to 2D for plotting
    umap_reducer = umap.UMAP(n_components=2, random_state=0)
    reduced_data = umap_reducer.fit_transform(data_for_clustering)

    # Convert data to a format suitable for Plotly
    fig = px.scatter(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        color=labels.astype(str),
        title=f'K-Means Clusters for Head {idx} using UMAP',
        labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Customize hover information
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                        selector=dict(mode='markers'),
                        hovertemplate='<b>Cluster</b>: %{marker.color}<br>' +
                                    'UMAP 1: %{x}<br>' +
                                    'UMAP 2: %{y}<extra></extra>')

    # Show the interactive plot
    fig.show()


def cluster(data, k=2):
    tensors = [steps['tensors'] for step_data in data for steps in step_data ]
    
    # cluster the data into k clusters with k-means
    
    num_heads = len(tensors[0])

    clustering_results = {}
    
    for head, idx in enumerate(range(num_heads)):
        flattened_tensors = [tensor[head].flatten() for tensor in tensors]
        # print(f'Head {head}, shape after flattening:', head_tensors[0].shape)  # Debugging shape

        data_for_clustering = np.stack(flattened_tensors)
    
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data_for_clustering)
        
         # Store clustering results: cluster labels and cluster centers
        clustering_results[head] = {
            'labels': kmeans.labels_,
            'centers': kmeans.cluster_centers_
        }

        # Plot clusters using PCA for dimensionality reduction
        plot_clusters(idx, data_for_clustering, kmeans.labels_, k)


    return kmeans.labels_