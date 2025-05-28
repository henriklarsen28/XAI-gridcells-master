import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVR
from vector_field import VectorField, create_vector_field
import seaborn as sns


# for measuring rotation pattern in vector field
def rotation_vectors(vector_field: VectorField, grid_length: int):

    center_point = (-grid_length / 2, grid_length / 2)
    print("Center point", center_point)
    length_dist = []

    for coordinate, (vector, _) in vector_field.vectors.items():

        vector_length = np.linalg.norm(vector)

        center_of_vector = (
            coordinate[0] + vector[0] / 2,
            coordinate[1] + vector[1] / 2,
        )
        dist_from_center = np.linalg.norm(
            np.array(center_of_vector) - np.array(center_point)
        )
        dicti = {
            "length": vector_length,
            "dist_from_center": dist_from_center,
        }

        length_dist.append(dicti)

    # Linear regression
    X = np.array([d["dist_from_center"] for d in length_dist]).reshape(-1, 1)
    y = np.array([d["length"] for d in length_dist])

    try:
        correlation, pval = pearsonr(X.flatten(), y)
        print("Correlation coefficient:", correlation)
        print("P-value:", pval)

        return correlation, pval
    except ValueError:
        print("Not enough data to calculate correlation.")
        return 0, 0

    

def nearest_neighbour(
    coordinate: np.ndarray, vector: np.ndarray, target_vectors, k: int = 3
) -> np.ndarray:
    # Compute Euclidean distances to all vectors in V
    target_vectors_keys = np.array(list(target_vectors.keys()))
    # target_vectors_keys = np.array(np.array(key) for key in target_vectors_keys.item())
    distances = np.linalg.norm(target_vectors_keys - coordinate, axis=1)

    # Get indices of the 3 smallest distances
    nearest_indices = np.argsort(distances)[:k]

    # Get the nearest vectors
    nearest_coordinates = target_vectors_keys[nearest_indices]
    nearest_vectors = np.array(
        [target_vectors[(key[0], key[1])][0] for key in nearest_coordinates]
    )
    return nearest_vectors


def compare_to_target(
    vector_field: VectorField, target_vector_field: VectorField, grid_length: int
) -> dict:

    # match the dictionary keys to ensure they are the same and sort the dictionaries, keep the values
    vector_field_dict = dict(sorted(vector_field.vectors.items()))
    target_vector_field_dict = dict(sorted(target_vector_field.vectors.items()))
    # check if the keys are the same

    # drop the keys that are not in the target vector field
    # vector_field_dict = {k: vector_field_dict[k] for k in vector_field_dict.keys() if k in target_vector_field_dict}
    # second method is to find the nearest neighbour

    # if vector_field_dict.keys() != target_vector_field_dict.keys():
    #    raise ValueError("The keys of the two dictionaries are not the same.")

    # Calculate the cosine similarity between the two vector fields
    # vector_field_values = np.array(list(vector_field_dict.values()))
    # target_vector_field_values = np.array(list(target_vector_field_dict.values()))
    cos_sim_list = []
    vector_length_diff = []
    for coordinate, (vector, _) in vector_field_dict.items():

        nearest_target_vectors = nearest_neighbour(
            coordinate, vector, target_vector_field_dict, k=3
        )

        # Calculate the cos sim for all the nearest target vectors
        cos_sim_vectors = cosine_similarity(
            vector.reshape(1, -1), nearest_target_vectors
        )
        cos_sim = np.max(cos_sim_vectors, axis=1)[0]
        # select the max cosine similarity
        target_vector_field_index = np.argmax(cos_sim)
        target_vector = nearest_target_vectors[target_vector_field_index]

        # cos_sim = cosine_similarity(vector.reshape(1, -1), target_vector.reshape(1, -1))
        cos_sim_list.append(cos_sim)
        vector_length_diff.append(
            np.linalg.norm(vector) - np.linalg.norm(target_vector)
        )

        # print(f"Vector: {vector}")
        print(f"Target vector: {target_vector}")
        print(f"Cosine similarity: {cos_sim}")
        print(
            f"Difference in length: {np.linalg.norm(vector) - np.linalg.norm(target_vector)}"
        )

    # Calculate the mean and standard deviation of the cosine similarity
    mean_cos_sim = np.mean(cos_sim_list)
    std_cos_sim = np.std(cos_sim_list)
    mean_vector_length_diff = np.mean(vector_length_diff)
    std_vector_length_diff = np.std(vector_length_diff)
    print(f"Mean cosine similarity: {mean_cos_sim}, std: {std_cos_sim}")
    print(
        f"Mean difference in length: {mean_vector_length_diff}, std: {std_vector_length_diff}"
    )
    dicti = {
        "cosine_similarity": cos_sim_list,
        "vector_length_difference": vector_length_diff,
    }
    return dicti

def plot_vector_diff(target_dicti: dict):
    """
    Box and whisker plot of the cosine similarity and vector length difference
    """
    """
    # Color palette
    palette = sns.color_palette([ "#6C8EBF", "#82B366","#D6B656"])
    sns.set_palette(palette)

    print(target_dicti)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    df = pd.DataFrame.from_dict(target_dicti, orient="columns")

    df = df.reset_index(names="metric")

    # Flatten it to long format
    long_df = df.melt(id_vars='metric', var_name='category', value_name='value')
    long_df = long_df.explode('value', ignore_index=True)  # now safe
    print(long_df)
    cos_sim_df = long_df[long_df["metric"] == "cosine_similarity"]
    vector_length_df = long_df[long_df["metric"] == "vector_length_difference"]



    sns.boxplot(data=cos_sim_df, x='category', y='value', ax=axes[0], hue='category')
    axes[0].set_title('Cosine similarity')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Cosine similarity')
    axes[0].set_ylim(-1, 1)
    axes[0].grid()

    # Vector length difference subplot
    sns.boxplot(data=vector_length_df, x='category', y='value', ax=axes[1], hue='category')
    axes[1].set_title('Vector length difference')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Length difference')
    axes[1].set_ylim(-12, 12)
    axes[1].grid()

    plt.tight_layout()
    plt.show()
    """
    # Prepare data into long format
    records = []

    for (method, episode), metrics in target_dicti.items():
        for metric_name, values in metrics.items():
            for value in values:
                records.append({
                    "method": method,
                    "episode": episode,
                    "metric": metric_name,
                    "value": value
                })

    df = pd.DataFrame(records)

    # Set color palette
    palette = sns.color_palette(["#6C8EBF", "#82B366", "#D6B656"])
    sns.set_palette(palette)

    # Setup subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot cosine similarity
    cos_sim_df = df[df["metric"] == "cosine_similarity"]
    sns.boxplot(data=cos_sim_df, x='episode', y='value', hue='method', ax=axes[0])
    axes[0].set_title('Cosine similarity')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cosine similarity')
    axes[0].set_ylim(-1, 1)
    axes[0].grid()
    axes[0].legend(loc='lower right')

    # Plot vector length difference
    vec_len_df = df[df["metric"] == "vector_length_difference"]
    sns.boxplot(data=vec_len_df, x='episode', y='value', hue='method', ax=axes[1])
    axes[1].set_title('Vector length difference')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length difference')
    axes[1].set_ylim(-6, 6)
    axes[1].grid()
    axes[1].legend(loc='lower right')
    # Adjust layout


    plt.tight_layout()
    plt.show()

def compare_vector_direction(
    vector_field: VectorField, vertical: bool = False, range: list = [45, 90, 180]
):
    """
    See how many of the vectors are in the same direction as expected, within a certain range.
    """
    # vector_field.plot_field_centered()
    percentages = {}
    for i, deg in enumerate(range):
        count = 0

        if vertical:
            expected_range = (270 - deg / 2, 270 + deg / 2)

        else:
            expected_range = (0 + deg / 2, 360 - deg / 2)

        print("Expected range", expected_range)

        for coordinate, (vector, _) in vector_field.vectors.items():
            angle = np.degrees(np.arctan2(vector[0], vector[1]))
            if angle < 0:
                angle += 360

            if (
                angle < expected_range[0] or angle > expected_range[1]
            ) and not vertical:
                count += 1

            elif angle > expected_range[0] and angle < expected_range[1] and vertical:
                count += 1

        
        try:
            percentage = count / len(vector_field.vectors) * 100
        except ZeroDivisionError:
            percentage = 0
        
        percentages[deg] = (percentage, f"{count} of {len(vector_field.vectors)}")

        print(
            f"Number of vectors in the expected direction: {count}",
            "out of",
            len(vector_field.vectors),
            " in deg:",
            deg,
        )
        print(
            "Percentage of vectors in the expected direction: ",
            percentage,
            "%",
        )

        # Plot the vectors with the expected direction
        angle_rad = np.deg2rad(expected_range[0])

        # Length of vector (e.g., 1 unit)
        length = 100
        x = length * np.cos(angle_rad)
        y = length * np.sin(angle_rad)

        angle_rad2 = np.deg2rad(expected_range[1])
        x2 = length * np.cos(angle_rad2)
        y2 = length * np.sin(angle_rad2)

        """if i == 0:
            plt.plot(
                [0, x], [0, y], color="#d62728", linestyle="--", linewidth=2
            )  # red dashed line
            plt.plot([0, x2], [0, y2], color="#d62728", linestyle="--", linewidth=2)

        elif i == 1:
            plt.plot([0, x], [0, y], color="#d62728", linestyle="-.", linewidth=2)
            plt.plot([0, x2], [0, y2], color="#d62728", linestyle="-.", linewidth=2)

        elif i == 2:
            plt.plot([0, x], [0, y], color="#d62728", linestyle=":", linewidth=3)
            plt.plot([0, x2], [0, y2], color="#d62728", linestyle=":", linewidth=3)"""

    return percentages

def plot_vector_direction(percentages: dict, methods: list):
    """
    Plot the percentages of vectors in the expected direction.
    """
    print(percentages)
    # Extract all methods and episodes
    #methods = sorted(set([k[0] for k in percentages.keys()]))
    episodes = sorted(set([k[1] for k in percentages.keys()]))
    angles = [45, 90, 180]
    # Prepare data for plotting
    for angle in angles:
        plot_data = {method: [] for method in methods}

        for episode in episodes:
            for method in methods:
                key = (method, episode)
                if key in percentages and angle in percentages[key]:
                    value = percentages[key][angle][0]
                    print(value)
                else:
                    value = np.nan
                plot_data[method].append(value)
                print(plot_data)

        # Plot
        x = np.arange(len(episodes))  # label locations
        width = 0.2

        palette = sns.color_palette(["#6C8EBF", "#82B366", "#D6B656"])
        sns.set_palette(palette)

        fig, ax = plt.subplots(figsize=(10, 6))

        

        for i, method in enumerate(methods):
            ax.bar(x + i*width, plot_data[method], width, label=method)

        # Labels and formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Percentage')
        ax.set_title(f'{angle}°')
        ax.set_xticks(x + width)
        ax.set_xticklabels(episodes)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y')
        plt.tight_layout()
        plt.show()

def count_equal_end_points(vector_field: VectorField, grid_length: int):
    """
    Count the number of vectors that have the same end point.
    """

    count = 0

    vector_end_points = []
    for coordinate, (vector, _) in vector_field.vectors.items():
        # check if the end point is in the vector field
        end_point = (
            coordinate[0] + vector[0],
            coordinate[1] + vector[1],
        )
        vector_end_points.append(end_point)

    # check for duplicates
    vector_end_points = pd.Series(vector_end_points).value_counts()
    vector_end_points = vector_end_points[vector_end_points > 1]
    print(vector_end_points)
    # sum up the number of duplicates
    for end_point, value in vector_end_points.items():
        count += value

    if count == 0:
        print("No vectors with the same end point")
        return
    print(
        f"Number of vectors with the same end point: {count} of {len(vector_field.vectors)}"
    )
    print(
        "Percentage of vectors with the same end point: ",
        count / len(vector_field.vectors) * 100,
        "%",
    )
    percentage = count / len(vector_field.vectors) * 100

    return percentage

def plot_same_end_points(percentages : dict, target_percentages: dict, methods: list, number_of_vectors: dict, number_of_target_vectors: dict):
    episodes = sorted(set([k[1] for k in percentages.keys()]))

    # Prepare data for plotting
    plot_data = {method: [] for method in methods}


    for episode in episodes:
        for method in methods:
            key = (method, episode)
            if key in percentages:
                value = percentages[key]
                num_vec = number_of_vectors[key]
                print(value)
                if value is None:
                    value = 0

            else:
                value = np.nan
            plot_data[method].append(value)
            
            #print(plot_data)

    # Plot
    x = np.arange(len(episodes))  # label locations
    width = 0.2

    palette = sns.color_palette(["#6C8EBF", "#82B366", "#D6B656"])
    sns.set_palette(palette)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    

    for i, method in enumerate(methods):
        bars = ax[0].bar(x + i*width, plot_data[method], width, label=method)
        

    # Labels and formatting
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel(f'Percentage')
    ax[0].set_title(f'Percentage of vectors with the same end point')
    ax[0].set_xticks(x + width)
    ax[0].set_xticklabels(episodes)
    ax[0].legend(loc='upper right')
    ax[0].set_ylim(0, 100)
    ax[0].grid(axis='y')

    target_plot_data = {method: [] for method in methods}

    for episode in episodes:
        for method in methods:
            key = (method, episode)
            if key in percentages:
                value = target_percentages[key]
                #num_vec = number_of_target_vectors[key]
                print(value)
                if value is None or num_vec == 0:
                    value = 0
                    #num_vec = ""
                
            else:
                value = np.nan
                #num_vec = np.nan
            target_plot_data[method].append(value)
            #print(target_plot_data)

    # Plot

    for i, method in enumerate(methods):
        bars = ax[1].bar(x + i*width, target_plot_data[method], width, label=method)
        

    # Labels and formatting
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel(f'Percentage')
    ax[1].set_title(f'Percentage of target vectors with the same end point')
    ax[1].set_xticks(x + width)
    ax[1].set_xticklabels(episodes)
    ax[1].legend(loc='upper right')
    ax[1].set_ylim(0, 100)
    ax[1].grid(axis='y')



    plt.tight_layout()
    plt.show()

def plot_number_of_vectors(number_of_vectors: dict, target_number_of_vectors: dict, methods: list):
    episodes = sorted(set([k[1] for k in number_of_vectors.keys()]))

    # Prepare data for plotting
    label_data = {method: [] for method in methods}


    for episode in episodes:
        for method in methods:
            key = (method, episode)
            if key in number_of_vectors:
                num_vec = number_of_vectors[key]

            else:
                num_vec = np.nan
            label_data[method].append(num_vec)
            
            #print(plot_data)

    # Plot
    x = np.arange(len(episodes))  # label locations
    width = 0.2

    palette = sns.color_palette(["#6C8EBF", "#82B366", "#D6B656"])
    sns.set_palette(palette)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    

    for i, method in enumerate(methods):
        bars = ax[0].bar(x + i*width, label_data[method], width, label=method)

    # Labels and formatting
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel(f'Number of vectors')
    ax[0].set_title(f'Number of vectors in the vector field')
    ax[0].set_xticks(x + width)
    ax[0].set_xticklabels(episodes)
    ax[0].legend(loc='upper right')
    ax[0].set_ylim(0, 40)
    ax[0].grid(axis='y')

    target_label_data = {method: [] for method in methods}

    for episode in episodes:
        for method in methods:
            key = (method, episode)
            if key in target_number_of_vectors:
                num_vec = target_number_of_vectors[key]
                
                
            else:
                num_vec = np.nan
            target_label_data[method].append(num_vec)
            #print(target_plot_data)

    # Plot

    for i, method in enumerate(methods):
        bars = ax[1].bar(x + i*width, target_label_data[method], width, label=method)

    # Labels and formatting
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel(f'Number of vectors')
    ax[1].set_title(f'Number of vectors in the target vector field')
    ax[1].set_xticks(x + width)
    ax[1].set_xticklabels(episodes)
    ax[1].legend(loc='upper right')
    ax[1].set_ylim(0, 40)
    ax[1].grid(axis='y')



    plt.tight_layout()
    plt.show()

def plot_correlation(correlation: dict, pvalue: dict, methods: list):

    episodes = sorted(set([k[1] for k in correlation.keys()]))

    x = np.arange(len(episodes))  # label locations
    width = 0.2


    correlation_data = {method: [] for method in methods}
    pvalue_data = {method: [] for method in methods}


    for episode in episodes:
        for method in methods:
            key = (method, episode)
            if key in correlation:
                value = correlation[key]
                print(value)
                if value is None:
                    value = 0
            else:
                value = np.nan


            if key in pvalue:
                pval = pvalue[key]
                print(pval)
                if pval is None:
                    pval = 0
            else:  
                pval = np.nan
            
            correlation_data[method].append(value)
            pvalue_data[method].append(pval)
            

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

    for i, method in enumerate(methods):
        ax[0].bar(x + i*width, correlation_data[method], width, label=method)
        ax[1].bar(x + i*width, pvalue_data[method], width, label=method)

    ax[0].set_title('Correlation Coefficient')
    ax[0].set_ylabel('Correlation')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylim(-1, 1)
    ax[0].legend(loc='upper right')
    ax[0].grid(axis='y')
    ax[0].set_xticks(x + width)
    ax[0].set_xticklabels(episodes)
    
    ax[1].set_title('P-value')
    ax[1].set_ylabel('P-value')
    ax[1].set_xlabel('Episode')
    ax[1].axhline(y=0.05, color="#B85450", linestyle='--', label='Significance Level (0.05)')

    ax[1].set_xticks(x + width)
    ax[1].set_xticklabels(episodes)
    ax[1].legend(loc='upper right')
    ax[1].grid(axis='y')
    ax[1].set_yscale('log')  # Log scale for better visibility of small p-values
    ax[1].set_ylim(0.001, 1)


    plt.xlabel("Episode")
    #plt.suptitle("Correlation between distance from center and vector length", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    source_map = "map_circular_4_19"
    #target_map = "map_circular_vertically_8_19"
    target_map = "map_circular_rot90_19_16"
    
    """source_map = "map_two_rooms_18_19"
    #target_map = "map_circular_vertically_8_19"
    target_map = "map_two_rooms_rot90_19_2"
"""
    methods = ["CAV", "CAV (Cos-sim)","CAR"]  #["CAV"]

    vertical = (True if "vertical" in target_map else False)

    episodes = [1000, 1025, 1200, 1700]
    #episodes = [1000]
    exclude = True

    target_metrics = {}
    ranges_metrics = {}
    endpoints = {}
    number_of_vectors = {}
    target_endpoints = {}
    number_of_target_vectors = {}

    correlation = {}
    pvalue = {}

    target_correlation = {}
    target_pvalue = {}
    

    for method in methods:

        for episode in episodes:

        

            vector_field = create_vector_field(method, exclude, episode, source_map, target_map, target=False)
            ranges_metrics[method, episode] = compare_vector_direction(vector_field, vertical=vertical, range=[45, 90, 180])
            
            correlation[method, episode], pvalue[method, episode] = rotation_vectors(vector_field, vector_field.grid_length)
            endpoints[method, episode] = count_equal_end_points(vector_field, vector_field.grid_length)
            number_of_vectors[method, episode] = len(vector_field.vectors)

            target_vector_field = create_vector_field(
                method, exclude, episode, source_map, target_map, target=True
            )
            target_correlation[method, episode], target_pvalue[method, episode] = rotation_vectors(target_vector_field, vector_field.grid_length)

            target_metrics[method, episode] = compare_to_target(vector_field, target_vector_field, vector_field.grid_length)
            target_endpoints[method, episode] = count_equal_end_points(target_vector_field, target_vector_field.grid_length)
            number_of_target_vectors[method, episode] = len(target_vector_field.vectors)

    #plot_vector_direction(ranges_metrics, methods)

    plot_vector_diff(target_metrics)

    plot_number_of_vectors(number_of_vectors, number_of_target_vectors, methods)

    plot_same_end_points(endpoints, target_endpoints, methods, number_of_vectors, number_of_target_vectors)

    plot_correlation(correlation, pvalue, methods)
    plot_correlation(target_correlation, target_pvalue, methods)




if __name__ == "__main__":
    main()
