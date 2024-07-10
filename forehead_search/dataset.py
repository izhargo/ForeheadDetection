import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset
import cv2


class ForeheadDataset(Dataset):
	def __init__(self, imagePaths: List[str], maskPaths: List[str], transforms: List[str]=None):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		with open(self.maskPaths[idx], 'rb') as f:
			mask = np.load(f)
		mask = torch.from_numpy(mask).float() / 255
		mask = torch.unsqueeze(mask, 0)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
		# return a tuple of the image and its mask
		return (image, mask)
