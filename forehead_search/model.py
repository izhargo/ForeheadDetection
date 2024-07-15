from forehead_search import config
# from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU, MSELoss, BCEWithLogitsLoss
from torchvision.transforms import CenterCrop
from torchvision.ops import sigmoid_focal_loss
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import torch


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)
	
	
class DownBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DownBlock, self).__init__()
		self.double_conv = DoubleConv(in_channels, out_channels)
		self.down_sample = nn.MaxPool2d(2)

	def forward(self, x):
		skip_out = self.double_conv(x)
		down_out = self.down_sample(skip_out)
		return (down_out, skip_out)

	
class UpBlock(nn.Module):
	def __init__(self, in_channels, out_channels, up_sample_mode):
		super(UpBlock, self).__init__()
		if up_sample_mode == 'conv_transpose':
			self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
		elif up_sample_mode == 'bilinear':
			self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
		self.double_conv = DoubleConv(in_channels, out_channels)

	def forward(self, down_input, skip_input):
		x = self.up_sample(down_input)
		x = torch.cat([x, skip_input], dim=1)
		return self.double_conv(x)

	
class UNet(nn.Module):
	def __init__(self, out_classes=1, up_sample_mode='conv_transpose'):
		super(UNet, self).__init__()
		self.up_sample_mode = up_sample_mode
		# Downsampling Path
		self.down_conv1 = DownBlock(3, 64)
		self.down_conv2 = DownBlock(64, 128)
		self.down_conv3 = DownBlock(128, 256)
		self.down_conv4 = DownBlock(256, 512)
		# Bottleneck
		self.double_conv = DoubleConv(512, 1024)
		# Upsampling Path
		self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
		self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
		self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
		self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
		# Final Convolution
		self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

	def forward(self, x):
		x, skip1_out = self.down_conv1(x)
		x, skip2_out = self.down_conv2(x)
		x, skip3_out = self.down_conv3(x)
		x, skip4_out = self.down_conv4(x)
		x = self.double_conv(x)
		x = self.up_conv4(x, skip4_out)
		x = self.up_conv3(x, skip3_out)
		x = self.up_conv2(x, skip2_out)
		x = self.up_conv1(x, skip1_out)
		x = self.conv_last(x)
		return x


class ObjFocalLoss(nn.Module):
	def __init__(self, numCls: int, λ: float, α: float) -> None:
		super(ObjFocalLoss, self).__init__()
		self.numCls = numCls
		self.λ = λ
		self.α = α
		self.focal_loss = sigmoid_focal_loss
	 
	def forward(self, mYHat: torch.Tensor, mY: torch.Tensor ) -> torch.Tensor:
		return self.λ * self.focal_loss(mYHat[:, :self.numCls], mY, alpha=self.α, reduction='mean')


class ObjLocLoss(nn.Module):
	def __init__(self, numCls: int, λ: float) -> None:
		super(ObjLocLoss, self).__init__()
		self.numCls = numCls
		self.λ = λ
		self.bce = nn.BCEWithLogitsLoss()
	
	def forward(self, mYHat: torch.Tensor, mY: torch.Tensor ) -> torch.Tensor:
		return self.λ * self.bce(mYHat[:, :self.numCls], mY)
	

class ObjLocScore(nn.Module):
	def __init__(self, numCls: int, threshold: float) -> None:
		super(ObjLocScore, self).__init__()
		self.numCls = numCls
		self.threshold = threshold
	
	def forward(self, mYHat: torch.Tensor, mY: torch.Tensor ) -> Tuple[float, float, float]:
		batchSize = mYHat.shape[0]
		pred_sig = torch.sigmoid(mYHat[:, :self.numCls])
		pred_sig = (pred_sig > self.threshold).float()

		intersection = (pred_sig * mY).sum()
		union = pred_sig.sum() + mY.sum()
		dice = (2. * intersection + 1e-8) / (union + 1e-8)
		return dice / batchSize

class DivideBy255(object):
	def __call__(self, tensor):
		return tensor / 255.0
