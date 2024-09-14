# Importing required libraries
import math, os, re, warnings, random, time
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
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa
from sklearn.manifold import TSNE

# Function to ensure reproducibility by setting random seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')

# Mixed precision and GPU detection
from tensorflow.keras import mixed_precision

# Detect and configure GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'Running on GPU {gpus}')
        strategy = tf.distribute.MirroredStrategy()
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# Set mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Model parameters
BATCH_SIZE = 32 * REPLICAS
LEARNING_RATE = 3e-5 * REPLICAS
EPOCHS_SCL = 60
EPOCHS = 50
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 2
N_FOLDS = 5
FOLDS_USED = 5

# Function to count data items in filenames
def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# Dataset paths
database_base_path = '/g/data/nk53/mr3328/bracs/mhist/'
train = pd.read_csv(f'{database_base_path}train.csv')
print(f'Train samples: {len(train)}')

IMAGES_PATH = os.path.join(database_base_path, "train_images/")
TF_RECORDS_PATH = os.path.join(database_base_path, "train_tfrecords/")
FILENAMES_COMP = [os.path.join(TF_RECORDS_PATH, file) for file in os.listdir(TF_RECORDS_PATH) if file.endswith('.tfrec')]

NUM_TRAINING_IMAGES = count_data_items(FILENAMES_COMP)
print(f'Number of training images: {NUM_TRAINING_IMAGES}')

CLASSES = ['HP', 'SSA']

# Data augmentation function
def data_augment(image, label):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    # Apply shear transformation
    if p_shear > .2:
        shear_angle = 20. if p_shear > .6 else -20.
        image = transform_shear(image, HEIGHT, shear=shear_angle)
    
    # Apply rotation
    if p_rotation > .2:
        rotate_angle = 45. if p_rotation > .6 else -45.
        image = transform_rotation(image, HEIGHT, rotation=rotate_angle)
    
    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > .75:
        image = tf.image.transpose(image)

    # Apply rotation transformations
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # 90º
    
    # Pixel-level augmentations
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)
    
    # Crop augmentations
    if p_crop > .6:
        image = tf.image.central_crop(image, central_fraction=.5 + (p_crop - .6) * .5)
    elif p_crop > .3:
        crop_size = tf.random.uniform([], int(HEIGHT * .6), HEIGHT, dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, CHANNELS])
    
    image = tf.image.resize(image, size=[HEIGHT, WIDTH])

    if p_cutout > .5:
        image = data_augment_cutout(image)
    
    return image, label

# Function to apply rotation transformation
def transform_rotation(image, height, rotation):
    DIM = height
    XDIM = DIM % 2
    rotation = rotation * tf.random.uniform([1], dtype='float32')
    rotation = math.pi * rotation / 180.  # Convert to radians
    
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    rotation_matrix = tf.reshape(tf.concat([c1, s1, 0.0, -s1, c1, 0.0, 0.0, 0.0, 1.0], axis=0), [3, 3])

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    idx2 = K.dot(rotation_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    
    idx3 = tf.stack([DIM // 2 - idx2[0], DIM // 2 - 1 + idx2[1]])
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d, [DIM, DIM, 3])

# Function to apply shear transformation
def transform_shear(image, height, shear):
    DIM = height
    XDIM = DIM % 2
    shear = shear * tf.random.uniform([1], dtype='float32')
    shear = math.pi * shear / 180.
    
    shear_matrix = tf.reshape(tf.concat([1.0, tf.math.sin(shear), 0.0, 0.0, tf.math.cos(shear), 0.0, 0.0, 0.0, 1.0], axis=0), [3, 3])

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    idx2 = K.dot(shear_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    
    idx3 = tf.stack([DIM // 2 - idx2[0], DIM // 2 - 1 + idx2[1]])
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d, [DIM, DIM, 3])

# CutOut augmentation
def data_augment_cutout(image, min_mask_size=(int(HEIGHT * .1), int(HEIGHT * .1)), max_mask_size=(int(HEIGHT * .125), int(HEIGHT * .125))):
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if p_cutout > .85:
        n_cutout = tf.random.uniform([], 10, 15, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .6:
        n_cutout = tf.random.uniform([], 5, 10, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .25:
        n_cutout = tf.random.uniform([], 2, 5, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    else:
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=1)

    return image

# Helper function for cutout
def random_cutout(image, height, width, channels=3, min_mask_size=(10, 10), max_mask_size=(80, 80), k=1):
    assert height > min_mask_size[0]
    assert width > min_mask_size[1]
    assert height > max_mask_size[0]
    assert width > max_mask_size[1]

    for _ in range(k):
        mask_height = tf.random.uniform(shape=[], minval=min_mask_size[0], maxval=max_mask_size[0], dtype=tf.int32)
        mask_width = tf.random.uniform(shape=[], minval=min_mask_size[1], maxval=max_mask_size[1], dtype=tf.int32)

        pad_h = height - mask_height
        pad_w = width - mask_width

        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left

        cutout_area = tf.zeros(shape=[mask_height, mask_width, channels], dtype=tf.uint8)
        cutout_mask = tf.pad([cutout_area], [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=1)
        cutout_mask = tf.squeeze(cutout_mask, axis=0)
        image = tf.multiply(tf.cast(image, tf.float32), tf.cast(cutout_mask, tf.float32))

    return image

# Function to decode images
def decode_image(image_data):
    return tf.image.decode_png(image_data, channels=3)

# Function to scale images
def scale_image(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

# Function to prepare images for training
def prepare_image(image, label):
    image = tf.image.resize(image, [HEIGHT_RS, WIDTH_RS])
    image = tf.reshape(image, [HEIGHT_RS, WIDTH_RS, 3])
    return image, label

# Function to read TFRecord dataset
def read_tfrecord(example, labeled=True):
    if labeled:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'target': tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'image_name': tf.io.FixedLenFeature([], tf.string), 
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    label_or_name = tf.cast(example['target'], tf.int32) if labeled else example['image_name']
    return image, label_or_name




# TensorFlow dataset preparation
def get_dataset(FILENAMES, labeled=True, ordered=False, repeated=False, cached=False, augment=False):
    """
    Return a TensorFlow dataset ready for training or inference.
    
    Args:
        FILENAMES (list): List of TFRecord filenames.
        labeled (bool): Whether the dataset is labeled or not.
        ordered (bool): If True, maintain order of data, otherwise shuffle it.
        repeated (bool): If True, repeat the dataset indefinitely.
        cached (bool): If True, cache the dataset for faster access.
        augment (bool): If True, apply data augmentation.
    
    Returns:
        dataset (tf.data.Dataset): The prepared dataset.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        dataset = tf.data.Dataset.list_files(FILENAMES)
        dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
    else:
        dataset = tf.data.TFRecordDataset(FILENAMES, num_parallel_reads=AUTO)
    
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTO)
    
    if augment:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    
    dataset = dataset.map(scale_image, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_image, num_parallel_calls=AUTO)
    
    if not ordered:
        dataset = dataset.shuffle(2048)
    if repeated:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(BATCH_SIZE)
    
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    
    return dataset

# Visualization utility functions
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    """
    Convert a batch of images and labels to numpy arrays.
    
    Args:
        data (tuple): A batch of (images, labels).
    
    Returns:
        numpy_images (numpy array): Array of images.
        numpy_labels (numpy array): Array of labels or None if unlabeled.
    """
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:  # binary string for image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    """
    Generate a title from label and correct label for visualization.
    
    Args:
        label (int): Predicted label.
        correct_label (int): Ground truth label.
    
    Returns:
        title (str): Title for the plot.
        correct (bool): Whether the prediction is correct.
    """
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return f"{CLASSES[label]} [{'OK' if correct else 'NO'} {'→' if not correct else ''}{CLASSES[correct_label] if not correct else ''}]", correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    """
    Display a single image with title.
    
    Args:
        image (numpy array): Image to display.
        title (str): Title for the image.
        subplot (tuple): Subplot parameters.
        red (bool): If True, display the title in red.
        titlesize (int): Font size for the title.
    
    Returns:
        subplot (tuple): Updated subplot configuration.
    """
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=titlesize if not red else int(titlesize / 1.2), color='red' if red else 'black', pad=int(titlesize / 1.5))
    return (subplot[0], subplot[1], subplot[2] + 1)

def display_batch_of_images(databatch, filename="batch_of_images.png", predictions=None):
    """
    Display a batch of images, optionally with predictions.
    
    Args:
        databatch (tuple): Batch of (images, labels).
        filename (str): File name to save the image grid.
        predictions (list): Optional predictions for each image.
    """
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
    
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows
    
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))
    
    for i, (image, label) in enumerate(zip(images[:rows * cols], labels[:rows * cols])):
        title = CLASSES[label] if label is not None else ''
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.savefig(filename, dpi=300)
    plt.close()

# Model evaluation utility functions
def dataset_to_numpy_util(dataset, N):
    """
    Convert a dataset to numpy arrays.
    
    Args:
        dataset (tf.data.Dataset): TensorFlow dataset.
        N (int): Number of samples to return.
    
    Returns:
        numpy_images (numpy array): Numpy array of images.
        numpy_labels (numpy array): Numpy array of labels.
    """
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    """
    Generate a title from label and correct label for model evaluation.
    
    Args:
        label (int): Predicted label.
        correct_label (int): Ground truth label.
    
    Returns:
        title (str): Title for the plot.
        correct (bool): Whether the prediction is correct.
    """
    label = np.argmax(label, axis=-1)
    correct = (label == correct_label)
    return f"{label} [{'OK' if correct else 'NO'}{' → ' if not correct else ''}{correct_label if not correct else ''}]", correct

def display_one_flower_eval(image, title, subplot, red=False):
    """
    Display a single image with evaluation title.
    
    Args:
        image (numpy array): Image to display.
        title (str): Title for the image.
        subplot (int): Subplot number.
        red (bool): If True, display the title in red.
    
    Returns:
        subplot (int): Updated subplot number.
    """
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=14, color='red' if red else 'black')
    return subplot + 1

def display_9_images_with_predictions(images, predictions, labels):
    """
    Display 9 images with predictions and ground truth labels.
    
    Args:
        images (list): List of images.
        predictions (list): List of predicted labels.
        labels (list): List of ground truth labels.
    """
    subplot = 331
    plt.figure(figsize=(13, 13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i])
        subplot = display_one_flower_eval(image, title, subplot, not correct)
        if i >= 8:
            break
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("predictions.png", dpi=300)
    plt.close()

# Function to plot training metrics
def plot_metrics(history, filename="metrics.png"):
    """
    Plot the training and validation metrics from the history object.
    
    Args:
        history (dict): History object from the model training.
        filename (str): File name to save the plot.
    """
    fig, axes = plt.subplots(2, 1, sharex='col', figsize=(20, 8))
    axes = axes.flatten()
    
    axes[0].plot(history['loss'], label='Train loss')
    axes[0].plot(history['val_loss'], label='Validation loss')
    axes[0].legend(loc='best', fontsize=16)
    axes[0].set_title('Loss')
    axes[0].axvline(np.argmin(history['loss']), linestyle='dashed')
    axes[0].axvline(np.argmin(history['val_loss']), linestyle='dashed', color='orange')
    
    axes[1].plot(history['sparse_categorical_accuracy'], label='Train accuracy')
    axes[1].plot(history['val_sparse_categorical_accuracy'], label='Validation accuracy')
    axes[1].legend(loc='best', fontsize=16)
    axes[1].set_title('Accuracy')
    axes[1].axvline(np.argmax(history['sparse_categorical_accuracy']), linestyle='dashed')
    axes[1].axvline(np.argmax(history['val_sparse_categorical_accuracy']), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.savefig(filename, dpi=300)
    plt.close()



# Embedding Visualization
def visualize_embeddings(embeddings, labels, filename="embeddings.png", figsize=(16, 16)):
    """
    Visualizes the 2D embedding space using TSNE and saves the plot.
    
    Args:
        embeddings (numpy array): Embedding vectors.
        labels (list): Corresponding labels for the embeddings.
        filename (str): Filename for saving the visualization.
        figsize (tuple): Size of the figure for the plot.
    """
    # Extract TSNE values from embeddings
    embed2D = TSNE(n_components=2, n_jobs=-1, random_state=seed).fit_transform(embeddings)
    embed2D_x = embed2D[:, 0]
    embed2D_y = embed2D[:, 1]

    # Create dataframe with labels and TSNE values
    df_embed = pd.DataFrame({'labels': labels})
    df_embed = df_embed.assign(x=embed2D_x, y=embed2D_y)

    # Create dataframes for each class
    df_embed_pb = df_embed[df_embed['labels'] == 0]
    df_embed_dcic = df_embed[df_embed['labels'] == 1]

    # Plot embeddings
    plt.figure(figsize=figsize)
    plt.scatter(df_embed_pb['x'], df_embed_pb['y'], color='yellow', s=10, label='N')
    plt.scatter(df_embed_dcic['x'], df_embed_dcic['y'], color='blue', s=10, label='C')
    
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

# Dataset Visualization
train_dataset = get_dataset(FILENAMES_COMP, ordered=True, augment=True)
train_iter = iter(train_dataset.unbatch().batch(20))

# Display batches of images
display_batch_of_images(next(train_iter))
display_batch_of_images(next(train_iter))

# Distribution of dataset
ds_dist = get_dataset(FILENAMES_COMP)
labels_comp = [target.numpy() for img, target in iter(ds_dist.unbatch())]

fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax = sns.countplot(y=labels_comp, palette='viridis')
ax.tick_params(labelsize=16)

plt.savefig('dataset_distribution.png', dpi=300)
plt.close()

# Metrics and Learning Rate Scheduler Initialization
precisions, recalls, f1_scores, accuracies = [], [], [], []
all_train_acc, all_val_acc, all_train_loss, all_val_loss = [], [], [], []

lr_start = 1e-8
lr_min = 1e-8
lr_max = LEARNING_RATE
num_cycles = 1.0
warmup_epochs = 1
hold_max_epochs = 0
total_epochs = EPOCHS
warmup_steps = warmup_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)
hold_max_steps = hold_max_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)
total_steps = total_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)

# Learning Rate Scheduler Function
@tf.function
def cosine_schedule_with_warmup(step, total_steps, warmup_steps=0, hold_max_steps=0, 
                                lr_start=1e-4, lr_max=1e-3, lr_min=None, num_cycles=0.5):
    """
    Cosine learning rate scheduler with warmup and optional hold phase.
    
    Args:
        step (int): Current step.
        total_steps (int): Total number of training steps.
        warmup_steps (int): Number of steps for warmup.
        hold_max_steps (int): Number of steps to hold max learning rate.
        lr_start (float): Initial learning rate.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        num_cycles (float): Number of cycles within the total training steps.
        
    Returns:
        lr (float): Computed learning rate.
    """
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(np.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, float(lr))
    return lr

# Generate learning rate schedule
rng = [i for i in range(int(total_steps))]
y = [cosine_schedule_with_warmup(tf.cast(x, tf.float32), tf.cast(total_steps, tf.float32), 
                                 tf.cast(warmup_steps, tf.float32), hold_max_steps, 
                                 lr_start, lr_max, lr_min, num_cycles) for x in rng]

sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(20, 6))
plt.plot(rng, y)
plt.savefig("plot_learning.png", dpi=300)
plt.close()


#For ResNet:
#def encoder_fn(input_shape):
#    inputs = L.Input(shape=input_shape, name='inputs')
#    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
#    x = base_model(inputs)
#    model = Model(inputs=inputs, outputs=x)
#    return model




# Model Definitions
def encoder_fn(input_shape):
    """
    Creates an encoder model using EfficientNetB3.

    Args:
        input_shape (tuple): Shape of the input tensor.

    Returns:
        Model: Encoder model.
    """
    inputs = L.Input(shape=input_shape, name='inputs')
    base_model = efn.EfficientNetB3(include_top=False, 
                                    weights="/g/data/nk53/mr3328/bracs/new_exp/efficientnet-b3_noisy-student_notop.h5", 
                                    pooling='avg')
    x = base_model(inputs)
    return Model(inputs=inputs, outputs=x)

def classifier_fn(input_shape, N_CLASSES, encoder, trainable=True):
    """
    Creates a classifier model on top of an encoder.

    Args:
        input_shape (tuple): Shape of the input tensor.
        N_CLASSES (int): Number of classes for classification.
        encoder (Model): Pretrained encoder model.
        trainable (bool): Whether to keep the encoder layers trainable.

    Returns:
        Model: Classifier model.
    """
    for layer in encoder.layers:
        layer.trainable = trainable
    
    inputs = L.Input(shape=input_shape, name='inputs')
    features = encoder(inputs)
    features = L.Dropout(0.5)(features)
    features = L.Dense(1000, activation='relu')(features)
    features = L.Dropout(0.5)(features)
    outputs = L.Dense(N_CLASSES, activation='softmax', name='outputs', dtype='float32')(features)
    
    return Model(inputs=inputs, outputs=outputs)

# Supervised Contrastive Loss
temperature = 0.1

class SupervisedContrastiveLoss(losses.Loss):
    """
    Custom supervised contrastive loss function.
    """
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        logits = tf.divide(tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized)), temperature)
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(input_shape, encoder):
    """
    Adds a projection head on top of an encoder.

    Args:
        input_shape (tuple): Shape of the input tensor.
        encoder (Model): Pretrained encoder model.

    Returns:
        Model: Encoder model with projection head.
    """
    inputs = L.Input(shape=input_shape, name='inputs')
    features = encoder(inputs)
    outputs = L.Dense(128, activation='relu', name='projection_head', dtype='float32')(features)
    
    return Model(inputs=inputs, outputs=outputs)

# Error handling for missing data
if not os.path.exists(TF_RECORDS_PATH):
    print(f"Error: {TF_RECORDS_PATH} does not exist.")

# Data counting function
def count_data_items(filenames):
    """
    Counts the total number of data items in the given filenames.

    Args:
        filenames (list): List of filenames.

    Returns:
        int: Total number of data items.
    """
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# Dataset paths and initialization
database_base_path = "/g/data/nk53/mr3328/bracs/mhist/"
train = pd.read_csv(f'{database_base_path}train.csv')
print(f'Train samples: {len(train)}')

IMAGES_PATH = "/g/data/nk53/mr3328/bracs/mhist/train_images/"
TF_RECORDS_PATH = "/g/data/nk53/mr3328/bracs/mhist/train_tfrecords/"
FILENAMES_COMP = tf.io.gfile.glob(os.path.join(TF_RECORDS_PATH, '*.tfrec'))
NUM_TRAINING_IMAGES = count_data_items(FILENAMES_COMP)
print(f'Number of training images: {NUM_TRAINING_IMAGES}')


# K-Fold Cross Validation
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
oof_pred, oof_labels, oof_embed = [], [], []

# Training loop for each fold
for fold, (idxT, idxV) in enumerate(skf.split(np.arange(5))):
    if fold >= FOLDS_USED:
        break

    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {idxT} VALID: {idxV}')
    
    TRAIN_FILENAMES = tf.io.gfile.glob([os.path.join(TF_RECORDS_PATH, f'Id_train{i:02d}*.tfrec') for i in idxT])
    VALID_FILENAMES = tf.io.gfile.glob([os.path.join(TF_RECORDS_PATH, f'Id_train{i:02d}*.tfrec') for i in idxV])
    
    np.random.shuffle(TRAIN_FILENAMES)
    ct_train = count_data_items(TRAIN_FILENAMES)
    step_size = (ct_train // BATCH_SIZE)
    total_steps = EPOCHS * step_size

    # Pre-training the encoder using Supervised Contrastive Loss
    print('Pre-training the encoder using "Supervised Contrastive" Loss')
    with strategy.scope():
        encoder = encoder_fn((None, None, CHANNELS))
        encoder_proj = add_projection_head((None, None, CHANNELS), encoder)
        encoder_proj.summary()

        lr = lambda: cosine_schedule_with_warmup(tf.cast(optimizer.iterations, tf.float32), 
                                                 tf.cast(total_steps, tf.float32), 
                                                 tf.cast(warmup_steps, tf.float32), 
                                                 hold_max_steps, lr_start, lr_max, lr_min, num_cycles)
        
        optimizer = optimizers.Adam(learning_rate=lr)
        encoder_proj.compile(optimizer=optimizer, 
                             loss=SupervisedContrastiveLoss(temperature))
        
    history_enc = encoder_proj.fit(x=get_dataset(TRAIN_FILENAMES, repeated=True, augment=True), 
                                   validation_data=get_dataset(VALID_FILENAMES, ordered=True, cached=True), 
                                   steps_per_epoch=step_size, 
                                   batch_size=BATCH_SIZE, 
                                   epochs=EPOCHS_SCL,
                                   verbose=2).history

    # Training classifier with frozen encoder
    print('Training the classifier with the frozen encoder')
    with strategy.scope():
        model = classifier_fn((None, None, CHANNELS), N_CLASSES, encoder, trainable=False)
        model.summary()

        lr = lambda: cosine_schedule_with_warmup(tf.cast(optimizer.iterations, tf.float32), 
                                                 tf.cast(total_steps, tf.float32), 
                                                 tf.cast(warmup_steps, tf.float32), 
                                                 hold_max_steps, lr_start, lr_max, lr_min, num_cycles)
        optimizer = optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, 
                      loss=losses.SparseCategoricalCrossentropy(), 
                      metrics=[metrics.SparseCategoricalAccuracy()])

    history = model.fit(x=get_dataset(TRAIN_FILENAMES, repeated=True, augment=True), 
                        validation_data=get_dataset(VALID_FILENAMES, ordered=True, cached=True), 
                        steps_per_epoch=step_size, 
                        epochs=EPOCHS,  
                        verbose=2).history

    model.save_weights(f'model_scl_{fold}.h5')

    # Results and metrics
    print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")
    
    ds_valid = get_dataset(VALID_FILENAMES, ordered=True)
    oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
    x_oof = ds_valid.map(lambda image, target: image)
    oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))
    oof_embed.append(encoder.predict(x_oof))

    y_true_fold = oof_labels[-1]
    y_pred_fold = oof_pred[-1]

    print(f"Classification Report for Fold {fold+1}:")
    print(classification_report(y_true_fold, y_pred_fold, target_names=CLASSES))
    
    # Collect metrics for each fold
    report = classification_report(y_true_fold, y_pred_fold, target_names=CLASSES, output_dict=True)
    precisions.append(report['weighted avg']['precision'])
    recalls.append(report['weighted avg']['recall'])
    f1_scores.append(report['weighted avg']['f1-score'])
    accuracies.append(report['accuracy'])

    all_train_acc.append(history['sparse_categorical_accuracy'])
    all_val_acc.append(history['val_sparse_categorical_accuracy'])
    all_train_loss.append(history['loss'])
    all_val_loss.append(history['val_loss'])

# Calculate and print overall metrics across all folds
mean_precision, std_precision = np.mean(precisions), np.std(precisions)
mean_recall, std_recall = np.mean(recalls), np.std(recalls)
mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
mean_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)

print(f"\nOverall Metrics Across All Folds:")
print(f"Precision: {mean_precision:.5f} ± {std_precision:.5f}")
print(f"Recall: {mean_recall:.5f} ± {std_recall:.5f}")
print(f"F1 Score: {mean_f1:.5f} ± {std_f1:.5f}")
print(f"Accuracy: {mean_accuracy:.5f} ± {std_accuracy:.5f}")


# Plot training and validation metrics across all folds
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Color list for different folds

for fold in range(FOLDS_USED):
    color = colors[fold % len(colors)]  # Cycle through colors if more than available
    plt.plot(all_train_acc[fold], color, label=f'Train Acc Fold {fold+1}')
    plt.plot(all_val_acc[fold], color + '--', label=f'Valid Acc Fold {fold+1}')
    plt.plot(all_train_loss[fold], color + ':', label=f'Train Loss Fold {fold+1}')
    plt.plot(all_val_loss[fold], color + '-.', label=f'Valid Loss Fold {fold+1}')

plt.title('Training and Validation Metrics Across All Folds')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()

# Save the figure to file
plt.savefig('training_validation_metrics.png', dpi=300)
plt.close()

# Combine OOF predictions, true labels, and embeddings from all folds
y_true = np.concatenate(oof_labels)
y_pred = np.concatenate(oof_pred)
embeddings_scl = np.concatenate(oof_embed)

# Visualize the embeddings using TSNE
visualize_embeddings(embeddings_scl, y_true, filename="plot_scl")

# Print classification report
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix Visualization
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
cfn_matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
cfn_matrix = (cfn_matrix.T / cfn_matrix.sum(axis=1)).T  # Normalize by the true labels

# Create a DataFrame from the confusion matrix for better visualization
df_cm = pd.DataFrame(cfn_matrix, index=CLASSES, columns=CLASSES)

# Plot the confusion matrix heatmap
sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('OOF', fontsize=30)

# Save the confusion matrix plot to a file
plt.savefig("confusion_matrix_2.png", dpi=300)
plt.close()
