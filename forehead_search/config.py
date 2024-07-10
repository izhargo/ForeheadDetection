import torch
import os


DATA_FOLDER = os.path.join('data', 'Forehead')
TRAIN_IMAGES_FOLDER = os.path.join(DATA_FOLDER, 'train', 'images')
TRAIN_LABELS_FOLDER = os.path.join(DATA_FOLDER, 'train', 'labels')
TRAIN_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'train', 'masks')
TRAIN_BIN_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'train', 'bin_masks')


VAL_IMAGES_FOLDER = os.path.join(DATA_FOLDER, 'val', 'images')
VAL_LABELS_FOLDER = os.path.join(DATA_FOLDER, 'val', 'labels')
VAL_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'val', 'masks')
VAL_BIN_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'val', 'bin_masks')


TEST_IMAGES_FOLDER = os.path.join(DATA_FOLDER, 'test', 'images')
TEST_LABELS_FOLDER = os.path.join(DATA_FOLDER, 'test', 'labels')
TEST_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'test', 'masks')
TEST_BIN_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'test', 'bin_masks')


KEY_TO_FOLDER = {
    'train': {'images': TRAIN_IMAGES_FOLDER, 'labels': TRAIN_LABELS_FOLDER},
    'val': {'images': VAL_IMAGES_FOLDER, 'labels': VAL_LABELS_FOLDER},
    'test': {'images': TEST_IMAGES_FOLDER, 'labels': TEST_LABELS_FOLDER}, 
}

# determine the device to be used for training and evaluation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == 'cuda' else False

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 16
L_CLASSES  = ['R', 'G', 'B']
numCls = len(L_CLASSES)
# define the input image dimensions
INPUT_IMAGE_WIDTH = 640 
INPUT_IMAGE_HEIGHT = 480
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = 'output'
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, 'unet.pth')
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])

λ = 20.0 #<! Localization Loss
ϵ = 0.1 #<! Label Smoothing
SIG_THRESHOLD = 0.5

