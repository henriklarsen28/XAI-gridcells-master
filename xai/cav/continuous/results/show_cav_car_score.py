import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def read_concept_scores(path: str):

    data = torch.load(path, weights_only=False)

    return data


def process_concept_scores(data, episode):

    # Split into blocks and sort out episodes that are not in the range
    data = [d for d in data if d[1] == str(episode)]

    block0 = [d[2] for d in data if d[0] == 0]
    block1 = [d[2] for d in data if d[0] == 1]
    block2 = [d[2] for d in data if d[0] == 2]
    print(f"Block0: {block0}, Block1: {block1}, Block2: {block2}")
    return block0[0], block1[0], block2[0]


def get_concept_scores(path, episode):

    dicti = {"block0": [], "block1": [], "block2": []}
    for file in os.listdir(path):
        if not file.endswith(".pt"):
            continue
        data = read_concept_scores(os.path.join(path, file))
        print(f"Data: {data}")
        block0, block1, block2 = process_concept_scores(data, episode)

        dicti["block0"].append(block0)
        dicti["block1"].append(block1)
        dicti["block2"].append(block2)
    return dicti


def plot_concept_scores(cav_data: dict, car_data: dict, episode: int):

    metric_dict = {
        "map_circular_4_19": "CIRCULAR",
        "map_circular_horizontally_4_40": "CIRCULAR-H",
        "map_circular_vertically_8_19": "CIRCULAR-V",
        "map_circular_rot90_19_16": "CIRCULAR-ROT",
        "map_two_rooms_18_19": "SQUARE",
        "map_two_rooms_horizontally_18_40": "SQUARE-H",
        "map_two_rooms_vertically_36_19": "SQUARE-V",
        "map_two_rooms_rot90_19_2": "SQUARE-ROT"
    }

    palette = sns.color_palette(["#6C8EBF", "#D6B656"])
    sns.set_palette(palette)

    # Helper to convert nested dicts into DataFrame
    def flatten_dict(data_dict, label, metric):
        rows = []
        for block, values in data_dict.items():
            for val in values:
                rows.append(
                    {"Block": block, "Value": val, "Source": label, "Metric": metric}
                )
        return rows

    # Prepare full DataFrame
    all_rows = []
    for metric in cav_data.keys():
        all_rows += flatten_dict(cav_data[metric], "CAV", metric)
        all_rows += flatten_dict(car_data[metric], "CAR", metric)

    df = pd.DataFrame(all_rows)

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 10), sharey=True)
    axes = axes.flatten()  # Flatten to 1D for easier indexing

    for i, metric in enumerate(cav_data.keys()):
        ax = axes[i]
        subset = df[df["Metric"] == metric]
        sns.boxplot(x="Block", y="Value", hue="Source", data=subset, ax=ax)
        ax.set_title(metric_dict[metric])
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.legend_.remove()  # Hide legend in subplots for clarity

        # Show X-axis labels only on the bottom row
        if i < 4:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Block")

    # Add legend to last subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(
        f"CAV vs CAR per environment at episode: {episode}", fontsize=16
    )
    plt.show()


def main():

    episode = 1700
    cav_data = {}
    car_data = {}
    maps = [
        "map_circular_4_19",
        "map_circular_horizontally_4_40",
        "map_circular_vertically_8_19",
        "map_two_rooms_18_19",
        "map_two_rooms_horizontally_18_40",
        "map_two_rooms_vertically_36_19",
    ]

    maps = [
        "map_circular_4_19",
        "map_circular_horizontally_4_40",
        "map_circular_vertically_8_19",

        "map_circular_rot90_19_16",
        "map_two_rooms_18_19",
        "map_two_rooms_horizontally_18_40",
        "map_two_rooms_vertically_36_19",

        "map_two_rooms_rot90_19_2",
    ]

    for map_name in maps:
        path_cav = f"helpful-bush-1369/{map_name}/grid_length_6/plot_data/"
        cav_data[map_name] = get_concept_scores(path_cav, episode)

        path_car = f"helpful-bush-1369/{map_name}/grid_length_6/plot_data_car/"
        car_data[map_name] = get_concept_scores(path_car, episode)

    plot_concept_scores(cav_data, car_data, episode)


if __name__ == "__main__":
    main()
