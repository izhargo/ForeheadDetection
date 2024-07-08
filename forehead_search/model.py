from forehead_search import config
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU, MSELoss
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from typing import Tuple
import torch
import torchvision



class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))
	

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs
	

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures
	

class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
	
	def forward(self, x):
		# grab the features from the encoder
		encFeatures = self.encoder(x)
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		# return the segmentation map
		return map


# Object Localization Loss
class ObjLocLoss( Module ):
	def __init__( self, numCls: int, λ: float, ϵ: float = 0.0 ) -> None:
		super(ObjLocLoss, self).__init__()

		self.numCls     = numCls
		self.λ          = λ
		self.ϵ          = ϵ
		self.oMseLoss   = MSELoss()
	
	def forward(self, mYHat: torch.Tensor, mY: torch.Tensor ) -> torch.Tensor:

		mseLoss = self.oMseLoss(mYHat[:, self.numCls:], mY[:, 1:])

		lossVal = (self.λ * mseLoss)
		
		return lossVal


# Object Localization Score
class ObjLocScore(Module):
	def __init__(self, numCls: int) -> None:
		super(ObjLocScore, self).__init__()

		self.numCls = numCls
	
	def forward(self, mYHat: torch.Tensor, mY: torch.Tensor ) -> Tuple[float, float, float]:

		batchSize = mYHat.shape[0]
		
		vY, mBox = mY[:, 0].to(torch.long), mY[:, 1:]

		vIoU = torch.diag(torchvision.ops.box_iou(torchvision.ops.box_convert(mYHat[:, self.numCls:], 'cxcywh', 'xyxy'), torchvision.ops.box_convert(mBox, 'cxcywh', 'xyxy')))
		vCor = (vY == torch.argmax(mYHat[:, :self.numCls], dim = 1)).to(torch.float32) #<! Correct labels

		# valIoU      = torch.mean(vIoU).item()
		# valAcc      = torch.mean(vCor).item()
		valScore    = torch.inner(vIoU, vCor) / batchSize
		
		return valScore