from forehead_search.dataset import ForeheadDataset
from forehead_search.model import UNet, ObjLocLoss, ObjLocScore
from forehead_search import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

from DeepLearningPyTorch import TrainModel

# # load the image and mask filepaths in a sorted manner
# imagePaths = sorted(list(paths.list_images(config.TRAIN_IMAGES_FOLDER)))
# maskPaths = sorted(list(paths.list_images(config.TRAIN_MASKS_FOLDER)))
# # partition the data into training and testing splits using 85% of
# # the data for training and the remaining 15% for testing
# split = train_test_split(imagePaths, maskPaths,
# 	test_size=config.TEST_SPLIT, random_state=42)
# # unpack the data split
# (trainImages, testImages) = split[:2]
# (trainMasks, testMasks) = split[2:]
# # write the testing image paths to disk so that we can use then
# # when evaluating/testing our model

if __name__ == "__main__":
      
    trainImages = sorted(list(paths.list_images(config.TRAIN_IMAGES_FOLDER)))
    trainMasks = sorted(list(paths.list_images(config.TRAIN_MASKS_FOLDER)))

    valImages = sorted(list(paths.list_images(config.VAL_IMAGES_FOLDER)))
    valMasks = sorted(list(paths.list_images(config.VAL_MASKS_FOLDER)))

    testImages = sorted(list(paths.list_images(config.TEST_IMAGES_FOLDER)))
    testMasks = sorted(list(paths.list_images(config.TEST_MASKS_FOLDER)))

    # print("[INFO] saving testing image paths...")
    # f = open(config.TEST_PATHS, "w")
    # f.write("\n".join(testImages))
    # f.close()


    transforms = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])
    # create the train and test datasets
    trainDS = ForeheadDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms)
    valDS = ForeheadDataset(imagePaths=valImages, maskPaths=valMasks,
        transforms=transforms)
    testDS = ForeheadDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    # create the training and test data loaders
    # TODO: add num_workers=os.cpu_count()
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()
        )
    valLoader = DataLoader(valDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()
        )
    # testLoader = DataLoader(testDS, shuffle=False,
    #     batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()
    #     )


    hL = ObjLocLoss(config.numCls, config.λ)
    hL = hL.to(config.DEVICE)
    hS = ObjLocScore(config.numCls, config.SIG_THRESHOLD)
    hS = hS.to(config.DEVICE)

    trainSteps = len(trainDS) // config.BATCH_SIZE


    # Training Loop
    unet = UNet().to(config.DEVICE)
    oOpt = AdamW(unet.parameters(), lr = 1e-5, betas = (0.9, 0.99), weight_decay = 1e-5)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 5e-4, total_steps = trainSteps)
    _, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(
        unet, trainLoader, valLoader, oOpt, config.NUM_EPOCHS, hL, hS, oSch = oSch
        )
    torch.save(unet, config.MODEL_PATH)
