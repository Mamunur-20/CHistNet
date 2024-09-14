# -*- coding: utf-8 -*-
"""
TFRecord Generation for Histopathology Image Classification
This script preprocesses images and creates TFRecord files for training a deep learning model.
"""

import os
import re
import random
import warnings
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from PIL import Image
import imagehash
import cv2


# Set random seed for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()

# Define directories (Update these paths as needed)
val_dir = '/path/to/patches_checked/train/'
output_dir = '/path/to/train_images/'
output_csv = 'train.csv'
database_base_path = '/path/to/patches_checked/'
IMAGES_DIR = os.path.join(database_base_path, 'train_images/')
HEIGHT, WIDTH = 512, 512
IMG_QUALITY = 100
N_FILES = 10  # Split into N TFRecord files

# Function to copy images to a specified directory
def copy_images_to_dir(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.png'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dest_dir, file)
                shutil.copy(src_path, dst_path)

# Create pandas DataFrame with image file names and labels
def create_image_dataframe(val_dir, output_csv):
    image_files, labels = [], []
    for root, _, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.png'):
                image_files.append(file)
                labels.append(os.path.basename(root))
    df = pd.DataFrame({'image_id': image_files, 'label': labels})
    df.to_csv(output_csv, index=False)

# Function to generate image hashes for deduplication
def generate_image_hashes(image_dir, funcs):
    image_ids, hashes = [], []
    for path in tqdm(glob.glob(os.path.join(image_dir, '*.png'))):
        image = Image.open(path)
        image_id = os.path.basename(path)
        image_ids.append(image_id)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
    hashes_all = np.array(hashes)
    return image_ids, torch.Tensor(hashes_all.astype(int))

# Function to remove duplicate images based on hashing
def remove_duplicates(image_ids, hashes_all, threshold=0.9):
    sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy() / 256 for i in range(hashes_all.shape[0])])
    indices1 = np.where(sims > threshold)
    indices2 = np.where(indices1[0] != indices1[1])
    image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
    image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
    dups = {tuple(sorted([image_id1, image_id2])): True for image_id1, image_id2 in zip(image_ids1, image_ids2)}
    duplicate_image_ids = sorted(list(dups))
    print(f'Found {len(duplicate_image_ids)} duplicates')
    return duplicate_image_ids

# TFRecord functions
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, target, image_name):
    feature = {
        'image': _bytes_feature(image),
        'target': _int64_feature(target),
        'image_name': _bytes_feature(image_name),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Write TFRecords from images
def write_tfrecords(train_df, path, num_files):
    for tfrec_num in range(num_files):
        print(f'Writing TFRecord {tfrec_num + 1} of {num_files}...')
        samples = train_df[train_df['file'] == tfrec_num]
        with tf.io.TFRecordWriter(f'Id_train{tfrec_num:02d}-{len(samples)}.tfrec') as writer:
            for row in samples.itertuples():
                img_path = os.path.join(path, row.image_id)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (HEIGHT, WIDTH))
                img = cv2.imencode('.png', img, (cv2.IMWRITE_PNG_COMPRESSION, IMG_QUALITY))[1].tostring()
                example = serialize_example(img, row.label, str.encode(row.image_id))
                writer.write(example)

# Load dataset from TFRecords
def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def read_tfrecord(example):
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    target = example['target']
    name = example['image_name']
    return image, target, name

def decode_image(image_data):
    image = tf.image.decode_png(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    return image

# Save image samples
def save_samples(ds, row, col, output_dir='output'):
    ds_iter = iter(ds)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for j in range(row * col):
        image, label, name = next(ds_iter)
        filename = f"{label[0]}_{name[0].numpy().decode('utf-8')}.png"
        output_path = os.path.join(output_dir, filename)
        img = tf.cast(image[0], tf.uint8).numpy()
        plt.imsave(output_path, img)

# Visualize data distribution
def visualize_data_distribution(train_df, n_folds):
    CLASSES = ['good', 'bad', 'moderate']
    label_count = train_df.groupby('label', as_index=False).count()
    label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
    label_count['Label'] = label_count['Label'].apply(lambda x: CLASSES[x])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis', ax=ax)
    ax.tick_params(labelsize=16)
    plt.savefig('All_label_count.png', bbox_inches='tight', dpi=300)

    for fold_n in range(n_folds):
        label_count_fold = train_df[train_df['file'] == fold_n].groupby('label', as_index=False).count()
        label_count_fold.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
        label_count_fold['Label'] = label_count_fold['Label'].apply(lambda x: CLASSES[x])

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f'File {fold_n + 1}', fontsize=22)
        sns.barplot(x=label_count_fold['Count'], y=label_count_fold['Label'], palette='viridis', ax=ax)
        ax.tick_params(labelsize=16)
        plt.savefig(f'plot_file_{fold_n + 1}.png', bbox_inches='tight', dpi=300)

# Main function to process data and create TFRecords
def main():
    # Copy images to the output directory
    copy_images_to_dir(val_dir, output_dir)
    
    # Create a DataFrame with image filenames and labels
    create_image_dataframe(val_dir, output_csv)

    # Generate image hashes for deduplication
    funcs = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]
    image_ids, hashes_all = generate_image_hashes(IMAGES_DIR, funcs)

    # Remove duplicate images
    duplicate_image_ids = remove_duplicates(image_ids, hashes_all)
    train_df = pd.read_csv(output_csv)
    train_df = train_df[~train_df['image_id'].isin(duplicate_image_ids)]
    train_df.reset_index(inplace=True)

    # Split dataset into folds
    folds = StratifiedKFold(n_splits=N_FILES, shuffle=True, random_state=seed)
    train_df['file'] = -1
    for fold_n, (_, val_idx) in enumerate(folds.split(train_df, train_df['label'])):
        train_df.loc[val_idx, 'file'] = fold_n

    # Write train DataFrame to CSV
    train_df.to_csv('train.csv', index=False)

    # Write TFRecords
    write_tfrecords(train_df, IMAGES_DIR, N_FILES)

    # Visualize data distribution
    visualize_data_distribution(train_df, N_FILES)

if __name__ == "__main__":
    main()
