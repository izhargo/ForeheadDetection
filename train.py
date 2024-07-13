import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from DeepLearningPyTorch import TrainModel
import utils
from forehead_search import config
from forehead_search.dataset import ForeheadDataset
from forehead_search.model import UNet, ObjLocLoss, ObjLocScore, DivideBy255, ObjFocalLoss


if __name__ == "__main__":   
    
    images = sorted(list(paths.list_images(config.TRAIN_IMAGES_FOLDER)))
    masks = sorted(list(paths.list_files(config.TRAIN_BIN_MASKS_FOLDER, validExts='npy')))
    trainImages, valImages, trainMasks , valMasks = utils.train_val_split_by_lead_num(images, masks)
    # trainImages, valImages, trainMasks , valMasks = train_test_split(images, masks, test_size = 0.2, train_size = 0.8, shuffle = True, random_state=42)
    
    transforms = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(), DivideBy255()])
    # create the train and test datasets
    trainDS = ForeheadDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
    valDS = ForeheadDataset(imagePaths=valImages, maskPaths=valMasks, transforms=transforms)
    # testDS = ForeheadDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the training set...")
    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()
        )
    valLoader = DataLoader(valDS, shuffle=True,
        batch_size=config.BATCH_SIZE*2, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()
        )

    hL = ObjFocalLoss(config.numCls, config.λ, config.α)
    # hL = ObjLocLoss(config.numCls, config.λ)

    hL = hL.to(config.DEVICE)
    hS = ObjLocScore(config.numCls, config.SIG_THRESHOLD)
    hS = hS.to(config.DEVICE)

    trainSteps = len(trainDS) // config.BATCH_SIZE

    # Training Loop
    unet = UNet().to(config.DEVICE)
    oOpt = AdamW(unet.parameters(), lr = 1e-5, betas = (0.9, 0.99), weight_decay = 1e-5)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 5e-4, total_steps = trainSteps)
    _, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(unet, trainLoader, valLoader, oOpt, config.NUM_EPOCHS, hL, hS, oSch = oSch)
    torch.save(unet, config.MODEL_PATH)

    # Plot Training Phase

    hF, vHa = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 5))
    vHa = np.ravel(vHa)

    hA = vHa[0]
    hA.plot(lTrainLoss, lw = 2, label = 'Train')
    hA.plot(lValLoss, lw = 2, label = 'Validation')
    hA.set_title(f'Object Localization Loss (λ = {config.λ:0.1f})')
    hA.set_xlabel('Epoch')
    hA.set_ylabel('Loss')
    hA.legend()

    hA = vHa[1]
    hA.plot(lTrainScore, lw = 2, label = 'Train')
    hA.plot(lValScore, lw = 2, label = 'Validation')
    hA.set_title('Object Localization Score')
    hA.set_xlabel('Epoch')
    hA.set_ylabel('Score')
    hA.legend()

    hA = vHa[2]
    hA.plot(lLearnRate, lw = 2)
    hA.set_title('Learn Rate Scheduler')
    hA.set_xlabel('Epoch')
    hA.set_ylabel('Learn Rate')
    hF.savefig(config.PLOT_PATH)

