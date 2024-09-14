# -*- coding: utf-8 -*-
"""
TFRecord Generation for Histopathology Image Classification
This script preprocesses images and creates TFRecord files for training a deep learning model.
"""

# CHistNet: Histopathology Image Classification Using Supervised Contrastive Deep Learning
# License: MIT License

import math
import os
import re
import warnings
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, losses, metrics, Model
from sklearn.manifold import TSNE

# Ensures reproducibility by setting a seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set seed for reproducibility
seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')

# Initialize TPU strategy if available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU {tpu.master()}')
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# Enable mixed precision to speed up training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)

# Enable XLA optimization
tf.config.optimizer.set_jit(True)

# --- Data preparation ---

# Paths to image data
val_dir = '/g/data/nk53/mr3328/bracs/binary/train/'
new_dir = '/g/data/nk53/mr3328/bracs/binary/train_images/'
output_path = 'train.csv'

# Create DataFrame with image file paths and labels
def create_image_dataframe(val_dir):
    image_files = []
    labels = []
    for root, _, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.png'):
                image_files.append(file)
                labels.append(os.path.basename(root))
    df = pd.DataFrame({'image_id': image_files, 'label': labels})
    df.to_csv(output_path, index=False)
    print(f"Image DataFrame saved to {output_path}")

# Copy images to a single directory
def copy_images(val_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for root, _, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.png'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(new_dir, file)
                shutil.copy(src_path, dst_path)
    print(f"Copied images to {new_dir}")

# Run the image preparation steps
create_image_dataframe(val_dir)
copy_images(val_dir, new_dir)

# --- Duplicate removal using image hashes ---

import glob, torch, imagehash
from tqdm.auto import tqdm
from PIL import Image

IMAGES_DIR = '/g/data/nk53/mr3328/bracs/binary/train_images/'

# Function to calculate image hashes for duplicate removal
def calculate_image_hashes(IMAGES_DIR):
    funcs = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]
    image_ids = []
    hashes = []

    for path in tqdm(glob.glob(IMAGES_DIR + '*.png')):
        image = Image.open(path)
        image_id = os.path.basename(path)
        image_ids.append(image_id)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

    return np.array(hashes), image_ids

# Calculate hashes for the images
hashes_all, image_ids = calculate_image_hashes(IMAGES_DIR)

# Convert hashes to torch tensors for further processing
hashes_all = torch.Tensor(hashes_all.astype(int))

# --- TensorFlow record creation and dataset loading ---

# Decode image from raw bytes
def decode_image(image_data, HEIGHT, WIDTH):
    image = tf.image.decode_png(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    return image

# Parse a TFRecord example
def read_tfrecord(example, HEIGHT, WIDTH):
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64), 
        'image_name': tf.io.FixedLenFeature([], tf.string), 
    }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'], HEIGHT, WIDTH)
    target = example['target']
    name = example['image_name']
    return image, target, name

# Load the dataset from TFRecord files
def load_dataset(filenames, HEIGHT, WIDTH, CHANNELS=3):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: read_tfrecord(x, HEIGHT, WIDTH), num_parallel_calls=AUTO)
    return dataset

# Function to save samples from the dataset as PNG images
def save_samples(ds, row, col, output_dir='output'):
    ds_iter = iter(ds)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for j in range(row * col):
        image, label, name = next(ds_iter)
        filename = f"{label[0]}_{name[0].numpy().decode('utf-8')}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save the image without displaying it
        img = tf.cast(image[0], tf.uint8)
        img_array = img.numpy()
        plt.imsave(output_path, img_array)
        
    print(f"Saved {row * col} images in the '{output_dir}' directory.")

# Function to count the number of data items in TFRecord filenames
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# Helper functions for creating TFRecords
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # Check if the value is a TensorFlow tensor
        value = value.numpy()  # Convert EagerTensor to numpy
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Serialize an example to be written as a TFRecord
def serialize_example(image, target, image_name):
    feature = {
        'image': _bytes_feature(image),
        'target': _int64_feature(target),
        'image_name': _bytes_feature(image_name),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Define paths and settings
database_base_path = '/g/data/nk53/mr3328/bracs/binary/'
PATH = f'{database_base_path}train_images/'
IMGS = os.listdir(PATH)
N_FILES = 5  # Number of files to split into
HEIGHT, WIDTH = (512, 512)
IMG_QUALITY = 100

print(f'Image samples: {len(IMGS)}')

# Compute image similarities for duplicate detection
sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy()/256 for i in range(hashes_all.shape[0])])
indices1 = np.where(sims > 0.9)
indices2 = np.where(indices1[0] != indices1[1])
image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([image_id1, image_id2])): True for image_id1, image_id2 in zip(image_ids1, image_ids2)}
duplicate_image_ids = sorted(list(dups))
print('Found %d duplicates' % len(duplicate_image_ids))

# Remove duplicate images from external data
imgs_to_remove = [x[1] for x in duplicate_image_ids]
remove_pd = []
for image in imgs_to_remove:
    remove_pd.append(image)

train = pd.read_csv(database_base_path + 'train.csv')

# Remove duplicates from the training data
train = train[~train['image_id'].isin(remove_pd)]
train.reset_index(inplace=True)
print('Train samples: %d' % len(train))

# Stratified K-Fold split for creating TFRecords
folds = StratifiedKFold(n_splits=N_FILES, shuffle=True, random_state=seed)
train['file'] = -1

for fold_n, (train_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print(f'File {fold_n + 1} has {len(val_idx)} samples')
    train['file'].loc[val_idx] = fold_n

train.to_csv('train.csv', index=False)

# Write TFRecords
for tfrec_num in range(N_FILES):
    print(f'\nWriting TFRecord {tfrec_num} of {N_FILES}...')
    samples = train[train['file'] == tfrec_num]
    n_samples = len(samples)
    print(f'{n_samples} samples')
    
    with tf.io.TFRecordWriter(f'Id_train{tfrec_num:02d}-{n_samples}.tfrec') as writer:
        for row in samples.itertuples():
            label = row.label
            image_name = row.image_id
            img_path = f'{PATH}{image_name}'
            
            img = cv2.imread(img_path)
            img = cv2.resize(img, (HEIGHT, WIDTH))
            img = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, IMG_QUALITY])[1].tobytes()
            
            # Serialize the example and write to the TFRecord file
            example = serialize_example(img, label, str.encode(image_name))
            writer.write(example)

# Load and verify TFRecords
AUTO = tf.data.experimental.AUTOTUNE
FILENAMES = tf.io.gfile.glob('Id_train*.tfrec')
print(f'TFRecords files: {FILENAMES}')
print(f'Created image samples: {count_data_items(FILENAMES)}')

# Save sample images from the dataset
save_samples(load_dataset(FILENAMES, HEIGHT, WIDTH).batch(1), 6, 6)

# Class labels
CLASSES = ['N', 'A']

# Plot and save the label distribution
label_count = train.groupby('label', as_index=False).count()
label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
label_count['Label'] = label_count['Label'].apply(lambda x: CLASSES[x])

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax = sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis')
ax.tick_params(labelsize=16)
plt.savefig('All_label_count.png', bbox_inches='tight', dpi=300)

# Save fold-specific label distributions
for fold_n in range(folds.n_splits):
    label_count = train[train['file'] == fold_n].groupby('label', as_index=False).count()
    label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
    
    # Ensure valid label indices
    label_count['Label'] = label_count['Label'].apply(lambda x: CLASSES[x] if 0 <= x < len(CLASSES) else f'Invalid index: {x}')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'File {fold_n + 1}', fontsize=22)
    ax = sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis')
    ax.tick_params(labelsize=16)
    
    # Save the plot for each fold
    plt.savefig(f'plot_file_{fold_n + 1}.png', bbox_inches='tight', dpi=300)