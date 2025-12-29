import numpy as np
import torch
import logging
log = logging.getLogger('exp')
from torch import nn
from torchvision import models
from ..pipeline.transforms_pytorch import torch_onehot
from ..common.util_networks import Padder
import torch.nn.functional as F

from torch.nn import functional as torch_functional

# class Correlation(nn.Module):
#
# 	@staticmethod
# 	def operation(a, b):
# 		"""
# 		B x C x H x W
# 		"""
# 		return torch.sum(a * b, dim=1, keepdim=True)
#
# 	def forward(self, a, b):
# 		return self.operation(a, b)

class Correlation(nn.Module):
    @staticmethod
    def operation(a, b):
        """
        a, b: tensors of shape [B, C, H, W]
        Returns: cosine dissimilarity map [B, 1, H, W]
        """
        # Normalize along channel dimension
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)

        # Cosine similarity: sum over channel dimension
        cos_sim = torch.sum(a_norm * b_norm, dim=1, keepdim=True)  # [B, 1, H, W]

        # Cosine dissimilarity (anomaly map)
        return 1 - cos_sim  # [B, 1, H, W]

    def forward(self, a, b):
        return self.operation(a, b)


# VGG16 features
class VggFeatures(nn.Module):
	LAYERS_VGG16 = [3, 8, 15, 22, 29]

	def __init__(self, vgg_mod, layers_to_extract, freeze=True):
		super().__init__()

		vgg_features = vgg_mod.features

		ends = np.array(layers_to_extract, dtype=int) + 1
		starts = [0] + list(ends[:-1])

		# print(list(zip(starts, ends)))

		self.slices = nn.Sequential(*[
			nn.Sequential(*vgg_features[start:end])
			for (start, end) in zip(starts, ends)
		])

		if freeze:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, image, **_):
		results = []
		value = image
		for slice in self.slices:
			value = slice(value)
			results.append(value)

		return results


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.shortcut(x) + self.conv_block(x)

class CorrDifference01(nn.Module):

	# class UpBlock(nn.Sequential):
	# 	def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True):
	#
	# 		modules = [
	# 			nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
	# 			nn.SELU(inplace=True),
	# 			nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
	# 			nn.SELU(inplace=True),
	# 		]
	#
	# 		if b_upsample:
	# 			modules += [
	# 				nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
	# 			]
	#
	# 		super().__init__(*modules)

	class UpBlock(nn.Module):
		def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True):
			super().__init__()

			self.res_block = ResidualBlock(in_channels, middle_channels)

			if b_upsample:
				self.upsample = nn.Sequential(
					nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
					nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
				)
			else:
				self.upsample = nn.Identity()

		def forward(self, x):
			x = self.res_block(x)
			x = self.upsample(x)
			return x

	class CatMixCorr(nn.Module):
		def __init__(self, in_ch):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch*2, in_ch, kernel_size=1)

		def forward(self, prev, feats_img, feats_rec):
			channels = [prev] if prev is not None else []

			channels += [
				Correlation.operation(feats_img, feats_rec),
				self.conv_1x1(torch.cat([feats_img, feats_rec], 1)),
			]

			# print('cat', [ch.shape[1] for ch in channels])

			return torch.cat(channels, 1)


	def __init__(self, num_outputs=2, freeze=True):
		super().__init__()

		self.vgg_extractor = VggFeatures(
			vgg_mod = models.vgg16(pretrained=True),
			layers_to_extract = VggFeatures.LAYERS_VGG16[:4],
			freeze = freeze,
		)


		feat_channels = [512, 256, 128, 64]
		out_chans = [256, 256, 128, 64]
		prev_chans = [0] + out_chans[:-1]
		cmis = []
		decs = []

		for i, fc, oc, pc in zip(range(feat_channels.__len__(), 0, -1), feat_channels, out_chans, prev_chans):

			#print(i, fc)
			#print(i, fc+1+pc, oc, oc)

			cmi = self.CatMixCorr(fc)
			dec = self.UpBlock(fc+1+pc, oc, oc, b_upsample=(i != 1))

			cmis.append(cmi)
			decs.append(dec)

			# self.add_module('cmi_{i}'.format(i=i), cmi)
			# self.add_module('dec_{i}'.format(i=i), dec)

		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

	def forward(self, image, gen_image, **_):

		if gen_image.shape != image.shape:
			gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]

		if not self.training:
			padder = Padder(image.shape, 16)
			image, gen_image = (padder.pad(x) for x in (image, gen_image))

		vgg_feats_img = self.vgg_extractor(image)
		vgg_feats_gen = self.vgg_extractor(gen_image)

		value = None
		num_steps = self.cmis.__len__()

		for i in range(num_steps):
			i_inv = num_steps-(i+1)
			value = self.decs[i](
				self.cmis[i](value, vgg_feats_img[i_inv], vgg_feats_gen[i_inv])
			)

		result = self.final(value)

		if not self.training:
			result = padder.unpad(result)

		return result

# SoftPool2D 实现
class SoftPool2D(nn.Module):
	def __init__(self, kernel_size=2, stride=2):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride

	def forward(self, x):
		x_exp = torch.exp(x)
		x_pool = F.avg_pool2d(x * x_exp, self.kernel_size, self.stride)
		norm = F.avg_pool2d(x_exp, self.kernel_size, self.stride)
		return x_pool / (norm + 1e-6)

class ComparatorImageToLabels(nn.Module):

	class CatMix(nn.Module):
		def __init__(self, in_ch_img, in_ch_sem, out_ch):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch_img + in_ch_sem, out_ch, kernel_size=1)

		def forward(self, prev, feats_img, feats_sem):
			channels = [prev] if prev is not None else []

			channels += [
				self.conv_1x1(torch.cat([feats_img, feats_sem], 1)),
			]

			return torch.cat(channels, 1)

	# class SemFeatures(nn.Sequential):
	#
	# 	def __init__(self, num_sem_classes, feat_channels_num_sem):
	# 		self.num_sem_classes = num_sem_classes
	#
	# 		nonlinearily = nn.ReLU(True)
	#
	# 		layers = [nn.Sequential(
	# 			nn.ReflectionPad2d(3),
	# 			nn.Conv2d(num_sem_classes, feat_channels_num_sem[0], kernel_size=7, padding=0),
	# 			nonlinearily,
	# 		)]
	#
	# 		num_prev_ch = feat_channels_num_sem[0]
	# 		for num_ch in feat_channels_num_sem[1:]:
	# 			layers.append(nn.Sequential(
	# 				nn.Conv2d(num_prev_ch, num_ch, kernel_size=3, stride=2, padding=1),
	# 				nonlinearily,
	# 			))
	# 			num_prev_ch = num_ch
	#
	# 		super().__init__(*layers)
	# 	# 将离散的语义标签图（H × W） → 多尺度、连续语义特征图（每层 C × H' × W'）
	# 	def forward(self, labels):
	# 		results = []
	# 		value = torch_onehot(labels, self.num_sem_classes, dtype=torch.float32)
	# 		# log.debug(f'discrepancy-onehot labels {labels.shape} {value.shape}')
	# 		for slice in self:
	# 			value = slice(value)
	# 			results.append(value)
	#
	# 		return results

	class SemFeatures(nn.Sequential):
		def __init__(self, num_sem_classes, feat_channels_num_sem):
			self.num_sem_classes = num_sem_classes
			nonlinearity = nn.SiLU()  # 更平滑的激活函数

			layers = [nn.Sequential(
				nn.ReflectionPad2d(3),
				nn.Conv2d(num_sem_classes, feat_channels_num_sem[0], kernel_size=7, padding=0),
				nonlinearity,
			)]

			num_prev_ch = feat_channels_num_sem[0]
			for num_ch in feat_channels_num_sem[1:]:
				layers.append(nn.Sequential(
					nn.Conv2d(num_prev_ch, num_ch, kernel_size=3, stride=1, padding=1),
					SoftPool2D(kernel_size=2, stride=2),  # 替代 AvgPool2d
					nonlinearity,
				))
				num_prev_ch = num_ch
			super().__init__(*layers)

		def forward(self, labels):
			results = []
			value = torch_onehot(labels, self.num_sem_classes, dtype=torch.float32)
			for layer in self:
				value = layer(value)
				results.append(value)
			return results

	def __init__(self, num_outputs=1, num_sem_classes=19, freeze=True):
		super().__init__()

		self.num_sem_classes = num_sem_classes

		self.vgg_extractor = VggFeatures(
			vgg_mod = models.vgg16(pretrained=True),
			layers_to_extract = VggFeatures.LAYERS_VGG16[:4],
			freeze = freeze,
		)

		feat_channels_num_vgg = [512, 256, 128, 64]
		feat_channels_num_sem = [256, 128, 64, 32]

		self.sem_extractor = self.SemFeatures(num_sem_classes, feat_channels_num_sem[::-1])

		out_chans = [256, 256, 128, 64]
		prev_chans = [0] + out_chans[:-1]

		cmis = []
		decs = []

		for i, fc, sc, oc, pc in zip(
				range(feat_channels_num_vgg.__len__(), 0, -1),
				feat_channels_num_vgg,
				feat_channels_num_sem,
				out_chans,
				prev_chans
		):

			cmi = self.CatMix(fc, sc, fc)
			dec = CorrDifference01.UpBlock(fc+pc, oc, oc, b_upsample=(i != 1))

			cmis.append(cmi)
			decs.append(dec)

			# self.add_module('cmi_{i}'.format(i=i), cmi)
			# self.add_module('dec_{i}'.format(i=i), dec)

		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

	def forward(self, labels, image, **_):

		if not self.training:
			padder = Padder(image.shape, 16)
			image, labels = (padder.pad(x) for x in (image, labels))

		vgg_feats_img = self.vgg_extractor(image)
		feats_sem = self.sem_extractor(labels)

		value = None
		num_steps = self.cmis.__len__()

		for i in range(num_steps):
			i_inv = num_steps-(i+1)
			value = self.decs[i](
				self.cmis[i](value, vgg_feats_img[i_inv], feats_sem[i_inv])
			)

		result = self.final(value)

		if not self.training:
			result = padder.unpad(result)

		return result

# 差异网络架构：比较真实图像、生成图像和标签，并进行特征融合，以生成最终的输出
class ComparatorImageToGenAndLabels(nn.Module):

	class CatMixCorrWithSem(nn.Module):
		def __init__(self, in_ch_img, in_ch_sem, ch_out):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch_img*2 + in_ch_sem, ch_out, kernel_size=1)

		def forward(self, prev, feats_img, feats_rec, feats_sem):
			channels = [prev] if prev is not None else []

			channels += [
				Correlation.operation(feats_img, feats_rec),
				self.conv_1x1(torch.cat([feats_img, feats_rec, feats_sem], 1)),
			]

			return torch.cat(channels, 1)


	def __init__(self, num_outputs=2, num_sem_classes=19, freeze=True):
		super().__init__()

		self.num_sem_classes = num_sem_classes

		self.vgg_extractor = VggFeatures(
			vgg_mod = models.vgg16(pretrained=True),
			layers_to_extract = VggFeatures.LAYERS_VGG16[:4],
			freeze = freeze,
		)

		feat_channels_num_vgg = [512, 256, 128, 64]
		feat_channels_num_sem = [256, 128, 64, 32]

		self.sem_extractor = ComparatorImageToLabels.SemFeatures(num_sem_classes, feat_channels_num_sem[::-1])

		out_chans = [256, 256, 128, 64]
		prev_chans = [0] + out_chans[:-1]

		cmis = []
		decs = []

		for i, fc, sc, oc, pc in zip(
				range(feat_channels_num_vgg.__len__(), 0, -1),
				feat_channels_num_vgg,
				feat_channels_num_sem,
				out_chans,
				prev_chans
		):

			cmi = self.CatMixCorrWithSem(fc, sc, fc)
			dec = CorrDifference01.UpBlock(fc + 1 + pc, oc, oc, b_upsample=(i != 1))

			cmis.append(cmi)
			decs.append(dec)

			# self.add_module('cmi_{i}'.format(i=i), cmi)
			# self.add_module('dec_{i}'.format(i=i), dec)

		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

	def forward(self, image, gen_image, labels, **_):

		if gen_image.shape != image.shape:
			gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]
		# 对输入图像及标签进行 padding，以满足某些网络的输入要求，并确保在不同的数据格式之间保持一致性。
		if not self.training:
			padder = Padder(image.shape, 16)
			image, gen_image, labels = (padder.pad(x.float()).type(x.dtype) for x in (image, gen_image, labels))

		#print(f'img {tuple(image.shape)} | gen {tuple(gen_image.shape)} | labels {tuple(labels.shape)}')
		#print(f'Label range {torch.min(labels)} ... {torch.max(labels)}')

		vgg_feats_img = self.vgg_extractor(image)
		vgg_feats_gen = self.vgg_extractor(gen_image)

		feats_sem = self.sem_extractor(labels)

		value = None
		num_steps = self.cmis.__len__()

		for i in range(num_steps):
			i_inv = num_steps-(i+1)
			value = self.decs[i](
				self.cmis[i](value, vgg_feats_img[i_inv], vgg_feats_gen[i_inv], feats_sem[i_inv])
			)

		result = self.final(value)

		if not self.training:
			result = padder.unpad(result)

		return result
