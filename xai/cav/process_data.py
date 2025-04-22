import ast
import copy
import json
import os
import random as rd
from collections import deque

import pandas as pd
import torch


def save_to_csv(dataset: deque, file_name: str, path: str):

    if not os.path.exists(path):
        os.makedirs(path)

    print("dataset", type(dataset[0]))
    # Convert the sequences 
    #dataset = [state.tolist() for state in dataset]

    dataset = [[state.tolist() for state in sequence] for sequence in dataset]
    
    # Convert from list of tensors to list of numpy arrays
    df = pd.DataFrame(dataset)

    df.to_csv(os.path.join(path, file_name + ".csv"), index=False)


def shuffle_and_trim_datasets(dataset: deque, max_length: int):
    # shuffle the dataset
    data = list(dataset)
    rd.shuffle(data)
    # trim the dataset
    if len(data) >= max_length:
        data = data[:max_length]
    return data


def split_dataset_into_train_test(
    dataset_path: str, dataset_subfolder="", ratio: float = 0.8
):
    # check if the folder 'train' and 'test' exists in the dataset path, if not create them
    train_dir = os.path.join(
        dataset_path, "train" if not dataset_subfolder == "" else dataset_subfolder
    )
    test_dir = os.path.join(
        dataset_path, "test" if not dataset_subfolder == "" else dataset_subfolder
    )

    raw_data_dir = os.path.join(dataset_path, dataset_subfolder)

    print("Splitting dataset into training and test set")
    # walk through the dataset path directory
    for file in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file)
        # check if the file is a csv file
        if not file.endswith(".csv"):
            continue
        # If it already is a test or train file, skip it
        if file.__contains__("train") or file.__contains__("test"):
            continue
        dataset = pd.read_csv(file_path)

        # Split the dataset into a training and test set
        train_size = int(len(dataset) * ratio)
        train_dataset = dataset[:train_size]
        print(
            "Length of dataset for file ",
            file,
            " is ",
            int(len(dataset)),
            "train size is ",
            train_size,
        )
        test_dataset = dataset[train_size:]

        train = [
            [torch.tensor(ast.literal_eval(state)) for state in states]
            for _, states in train_dataset.iterrows()
        ]
        test = [
            [torch.tensor(ast.literal_eval(state)) for state in states]
            for _, states in test_dataset.iterrows()
        ]

        filename = os.path.splitext(file)[0]
        save_to_csv(train, f"{filename}_train", train_dir)
        save_to_csv(test, f"{filename}_test", test_dir)


def save_config(dataset_path: str, config: dict):

    if os.path.exists(dataset_path) == False:
        os.makedirs(dataset_path)
    with open(os.path.join(dataset_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def find_model_files(base_path: str, ep_ids: list):

    # find the model file that ends with a specific number followed by '.pth'
    try:
        files = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Directory {base_path} does not exist.")
        return None

    ep_ids = sorted(ep_ids, reverse=False)

    model_files = []

    for num in ep_ids:
        for file in files:
            if file.endswith(f"_{num}.pth"):
                path = os.path.join(base_path, file)
                model_files.append(path)
    return model_files


def build_random_dataset(dataset_path: str, dataset_subfolder=""):

    file_path = os.path.join(dataset_path, dataset_subfolder)
    # load and concatenate all CSV files
    print("Building random dataset")
    files = [
        os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".csv")
    ]
    dataset = pd.concat([pd.read_csv(f) for f in files])

    dataset_length = len(dataset)
    sample_size = min(1500, dataset_length)
    # shuffle and sample the dataset
    random_sample = dataset.sample(
        n=sample_size, frac=None, random_state=42
    ).reset_index(drop=True)

    # Split into positive and negative
    half = len(random_sample) // 2  # Use integer division directly
    random_positive = random_sample.iloc[:half]
    random_negative = random_sample.iloc[half:]

    random_positive.to_csv(
        f"{os.path.join(dataset_path, 'random_positive.csv')}", index=False
    )
    random_negative.to_csv(
        f"{os.path.join(dataset_path, 'random_negative.csv')}", index=False
    )


def get_positive_negative_data(concept: str, datapath: str):
    negative_files = []
    positive_file = None

    print('Concept:', concept)
    print('Datapath:', datapath)
    for file in os.listdir(datapath):
        file_path = os.path.join(datapath, file)
        base_name, extension = os.path.splitext(file)
        if base_name.__contains__("negative") or base_name.__contains__("positive"):
            continue
        print("Looking for concept", concept)
        # print("Base name:", base_name)
        # print("Extension:")
        if base_name.startswith(concept) and (
            base_name == concept or base_name[len(concept) : len(concept) + 1] == "_"
        ):
            positive_file = file_path
            print("Positive file:", positive_file)
        else:
            negative_files.append(file_path)
            print("Negative file:", file_path)

    if positive_file is None:
        return None, None
    
    positive_df = pd.read_csv(positive_file)
    
    # Determine sample size: at least 3000 lines or the length of the positive file content, whichever is greater
    sample_size = min(1500, len(positive_df))

    # Aggregate negative file content and then sample
    neg_dfs = []
    for neg_file in negative_files:
        print("Reading negative file ", neg_file)
        neg_df = pd.read_csv(neg_file)
        neg_dfs.append(neg_df)

    negative_df = pd.concat(neg_dfs)
    negative_df = negative_df.sample(sample_size)

    return positive_df, negative_df


def grid_observation_dataset(dataset_path, grid_size: int):
    for i in range(grid_size):
        concept = "grid_observations_" + str(i)
        negative_file_test = os.path.join(
            dataset_path, "test", f"{concept}_negative_test.csv"
        )
        negative_file_train = os.path.join(
            dataset_path, "train", f"{concept}_negative_train.csv"
        )

        if not os.path.exists(negative_file_test):
            positive_file_test, negative_file_test = get_positive_negative_data(
                concept, os.path.join(dataset_path, "test")
            )
            if positive_file_test is None:
                continue  # Skipping concept
            positive_file_test.to_csv(
                os.path.join(dataset_path, "test", f"{concept}_positive_test.csv"),
                index=False,
            )
            negative_file_test.to_csv(
                os.path.join(dataset_path, "test", f"{concept}_negative_test.csv"),
                index=False,
            )

        if not os.path.exists(negative_file_train):
            positive_file_train, negative_file_train = get_positive_negative_data(
                concept, os.path.join(dataset_path, "train")
            )
            if positive_file_train is None:
                continue
            positive_file_train.to_csv(
                os.path.join(dataset_path, "train", f"{concept}_positive_train.csv"),
                index=False,
            )
            negative_file_train.to_csv(
                os.path.join(dataset_path, "train", f"{concept}_negative_train.csv"),
                index=False,
            )
