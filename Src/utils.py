import os
import torch
import numpy as np
import json
import pickle


def set_seeds(seed_val=3047):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def save_txt_single_column(data, save_dir='./', filename='save_test.txt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, filename), 'w') as f:
        for i in data:
            f.write(str(i) + '\n')


def keep_nonNaN_values(arr, n, random_seed=42):
    np.random.seed(random_seed)

    mask = np.full(arr.shape, False)

    for i, row in enumerate(arr):
        non_nan_indices = np.where(~np.isnan(row))[0]

        if len(non_nan_indices) > n:
            indices_to_keep = np.random.choice(non_nan_indices, n, replace=False)
            indices_to_replace = np.setdiff1d(non_nan_indices, indices_to_keep)
            arr[i, indices_to_replace] = np.nan
            mask[i, indices_to_replace] = True

    return arr, mask


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_dict_2_json(data, filename, save_dir):
    with open(os.path.join(save_dir, filename), 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_tokenizer(tokenizer, filename='tokenizer.pkl', save_dir='./'):
    with open(os.path.join(save_dir, filename), 'wb') as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(path):
    with open(path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer
