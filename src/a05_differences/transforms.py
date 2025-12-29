import cv2
import numpy as np
from math import floor
from scipy import stats
from random import choices
from ..paths import DIR_DATA
from ..pipeline.transforms import TrsChain
from ..pipeline.transforms_imgproc import TrShow
from ..datasets.dataset import imread
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo
from ..a01_sem_seg.transforms import SemSegLabelsToColorImg
from ..a04_reconstruction.transforms import tr_instances_from_semantics

# 0.1 作用是统一标签格式，将原始标签映射为适合评估的标签，确保评估过程中的标签标准化。
def tr_label_to_validEval(labels, dset, **_):
	v = dset.label_info.valid_for_eval_trainId[labels.reshape(-1)].reshape(labels.shape)
	return dict(
		labels_validEval = v,
	)

# 通过比较预测标签与真实标签，输出一个表示预测错误的位置掩码（mask）
def tr_get_errors(labels, pred_labels, labels_validEval, **_):
	errs = (pred_labels != labels) & labels_validEval
	return dict(
		semseg_errors = errs,
	)

# 将语义分割的误差数组转换为有效的 ground truth 标签数组
def tr_errors_to_gt(semseg_errors, labels_validEval, **_):
	errs = semseg_errors.astype(np.int64)
	errs[np.logical_not(labels_validEval)] = 255  # 将无效标签区域的误差值设置为 255，标记为“无效”或“忽略”
	return dict(
		semseg_errors_label = errs,
	)

def tr_errors_to_gt_float(semseg_errors, labels_validEval, **_):
	errs = semseg_errors.astype(np.int64)
	errs[np.logical_not(labels_validEval)] = 255
	return dict(
		semseg_errors_label = errs,
	)


try:
	"""
	CTC_ROI is the region of interest associated with the Cityscapes vehicle/camera setup.
	For example, it excludes the ego vehicle from evaluation.
	Since Lost And Found does not provide that kind of ROI, but uses a similar vehicle, we reuse the Cityscapes ROI.

	By default it is stored in DIR_DATA/cityscapes/roi.png
	"""
	CTC_ROI_path = DIR_DATA/'cityscapes/roi.png'
	CTC_ROI = imread(CTC_ROI_path).astype(bool)
	# print(f'Cityscapes ROI loaded from {CTC_ROI_path}')
except Exception as e:
	print(f'Cityscapes ROI file is not present at {CTC_ROI_path}): {e}')
	CTC_ROI = np.ones((512, 1024), dtype=bool)

CTC_ROI_neg = ~CTC_ROI

DISAPPEAR_TRAINIDS = [CityscapesLabelInfo.name2trainId[n] for n in ['person', 'rider', 'car', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']]

FORBIDDEN_DISAPPEAR_ALTERNATIVE_IDS = [0, 1, 2, 3]
FORBIDDEN_DISAPPEAR_ALTERNATIVE_TRAINIDS = [CityscapesLabelInfo.trainId2label[n].id for n in FORBIDDEN_DISAPPEAR_ALTERNATIVE_IDS]


MORPH_KERNEL = np.ones((11, 11), np.uint8)

def tr_LAF_exclude_anomalies_from_difference_training(semseg_errors_label, labels_source, **_):
	anomaly_mask = labels_source > 1

	#print(np.count_nonzero(anomaly_mask), np.count_nonzero(semseg_errors_label == 255))


	# expand with a margin
	anomaly_mask = cv2.dilate(anomaly_mask.astype(np.uint8), MORPH_KERNEL).astype(np.bool)

	label_to_override = semseg_errors_label.copy()
	label_to_override[anomaly_mask] = 255

	#print(np.count_nonzero(anomaly_mask), np.count_nonzero(label_to_override == 255))

	return dict(
		semseg_errors_label = label_to_override,
	)

def tr_exclude_ROI_from_difference_training(semseg_errors_label, roi, **_):
	semseg_errors_label[~roi] = 255
	return dict(
		semseg_errors_label = semseg_errors_label,
	)

tr_show_errors = TrsChain(
	SemSegLabelsToColorImg([('labels', 'labels_colorimg'), ('pred_labels', 'pred_labels_colorimg')]),
	tr_label_to_validEval,
	tr_get_errors,
	TrShow(
		['labels_colorimg', 'pred_labels_colorimg'],
		['image', 'labels_validEval'],
		'semseg_errors',
	),
)


def tr_disappear_inst(labels_source, instances, inst_ids=None, clear_instance_map=False, only_objects=True, swap_fraction=0.5, **_):
	if inst_ids is None:
		inst_uniq = np.unique(instances)

		if only_objects:
			inst_uniq_objects = inst_uniq[inst_uniq >= 24000]
		else:
			inst_uniq_objects = inst_uniq[inst_uniq >= 1]

		if inst_uniq_objects.__len__() == 0:
			return dict(
				labels_fakeErr=labels_source.copy(),
			)

		inst_ids = np.random.choice(inst_uniq_objects, int(inst_uniq_objects.__len__() *swap_fraction), replace=False)
		#print(inst_uniq, 'remove', inst_ids)


	disappear_mask = np.any([instances == inst_id for inst_id in inst_ids], axis=0)

	obj_classes = np.unique(labels_source[disappear_mask])

	forbidden_classes = DISAPPEAR_TRAINIDS
	forbidden_class_mask = np.any([labels_source == cl for cl in forbidden_classes], axis=0)

	mask_dont_use_label = forbidden_class_mask | disappear_mask
	mask_use_label = np.logical_not(mask_dont_use_label)

	# 	show(forbidden_class_mask)

	# 	dis_mask_u8 = disappear_mask.astype(np.uint8)
	nearest_dst, nearest_labels = cv2.distanceTransformWithLabels(
		mask_dont_use_label.astype(np.uint8),
		distanceType=cv2.DIST_L2,
		maskSize=5,
		labelType=cv2.DIST_LABEL_PIXEL,
	)

	background_indices = nearest_labels[mask_use_label]
	background_labels = labels_source[mask_use_label]

	label_translation = np.zeros(labels_source.shape, dtype=np.uint8).reshape(-1)
	label_translation[background_indices] = background_labels

	label_reconstr = labels_source.copy()
	label_reconstr[disappear_mask] = label_translation[nearest_labels.reshape(labels_source.shape)[disappear_mask]]

	# 	label_reconstr = label_translation[nearest_labels].reshape(labels_source.shape)

	# 	show(label_reconstr)

	result = dict(
		# 		dist_lab = nearest_labels,
		labels_fakeErr=label_reconstr,
	)

	if clear_instance_map:
		inst_cleared = instances.copy()
		inst_cleared[disappear_mask] = 0
		result['instances'] = inst_cleared

	return result

# 通过指定的消失比例（disap_fraction）模拟标签中的物体消失现象
def tr_synthetic_disappear_objects(pred_labels_trainIds, instances = None, disap_fraction=0.5, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	if instances is None:
		instances_encode_class = False
		instances = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']
	else:
		instances_encode_class = True # gt instances

	labels_disap = tr_disappear_inst(
		pred_labels_trainIds,
		instances,
		only_objects = instances_encode_class,
		fraction = disap_fraction,
	)['labels_fakeErr']


	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_disap,
		semseg_errors=(pred_labels_trainIds != labels_disap) & CTC_ROI,
	)


def tr_synthetic_disappear_objects_onPred(pred_labels_trainIds, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	instance_map = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']

	labels_disap = tr_disappear_inst(pred_labels_trainIds, instance_map, only_objects=False)['labels_fakeErr']

	return dict(
		instances=instance_map,
		labels_fakeErr_trainIds=labels_disap,
		semseg_errors=(labels != labels_disap) & CTC_ROI,
	)


# # 用于生成带有标签错误的合成数据, 它模拟语义分割任务中的标签扰, 通过交换对象类别来创建合成错误标签
# def tr_swap_labels(labels_source, instances, inst_ids=None, only_objects=False, fraction=0.2, target_classes=np.arange(19), invalid_class=255, **_):
# 	if inst_ids is None:
# 		inst_uniq = np.unique(instances)  # 获取 instances 图像中所有的实例 ID
# 		# 实例(instances)是对图像中所有物体个体的精确描述，包括前景和特定的背景物体
# 		if only_objects: # 控制实例过滤
# 			inst_uniq_objects = inst_uniq[inst_uniq >= 24000]  # True: 仅扰动物体类（ID >= 24000），适合 Cityscapes 数据集（24000 以上通常是物体类）
# 		else:
# 			inst_uniq_objects = inst_uniq[inst_uniq >= 1]  # False: 扰动所有非背景类（ID >= 1）
#
# 		if inst_uniq_objects.__len__() == 0:  # 若无可选实例
# 			return dict(
# 				labels_fakeErr=labels_source.copy(),  # 返回原标签
# 			)
# 		# 按比例随机选取实例
# 		inst_ids = np.random.choice(inst_uniq_objects, floor(inst_uniq_objects.__len__() * fraction), replace=False)
#
# 	# print(inst_uniq, 'remove', inst_ids)
#
# 	labels = labels_source.copy()  # 创建标签副本，避免修改原数据
#
# 	for inst_id in inst_ids:
# 		inst_mask = instances == inst_id  # 获取当前实例的掩码
#
# 		inst_view = labels[inst_mask]  # 获取实例内的标签类别
# 		# 获取实例内标签的众数（即该实例的主要类别）
# 		obj_class = stats.mode(inst_view, axis=None).mode[0]
#
# 		if obj_class != invalid_class:  # 若实例不是无效类别
# 			tc = list(target_classes)
# 			try:
# 				tc.remove(obj_class)  # 移除原类别，避免自我替换
# 			except ValueError:
# 				print(f'Instance class {obj_class} not in set of classes {target_classes}')
# 			new_class = choices(tc)  ## 随机选取新类别
#
# 			labels[inst_mask] = new_class
#
# 	result = dict(
# 		# 		dist_lab = nearest_labels,
# 		labels_fakeErr=labels
# 	)
#
# 	return result

# 代码改进：结合类别分布进行替换
# Cityscapes 数据集类别及其数量
class19_distribution = {
    "road": 3616, "sidewalk": 8349, "building": 8301, "wall": 1979, "fence": 2839,
    "pole": 52748, "traffic light": 11898, "traffic sign": 24976, "vegetation": 17745,
    "terrain": 5070, "sky": 3421, "person": 21413, "rider": 2363, "car": 31822,
    "truck": 582, "bus": 483, "train": 194, "motorcycle": 888, "bicycle": 4904
}
def tr_swap_labels(labels_source, instances, inst_ids=None, only_objects=False, fraction=0.2, target_classes=np.arange(19), invalid_class=255, **_):
	if inst_ids is None:
		inst_uniq = np.unique(instances)  # 获取 instances 图像中所有的实例 ID
		# 实例(instances)是对图像中所有物体个体的精确描述，包括前景和特定的背景物体
		if only_objects:  # 控制实例过滤
			inst_uniq_objects = inst_uniq[inst_uniq >= 24000]  # True: 仅扰动物体类（ID >= 24000），适合 Cityscapes 数据集（24000 以上通常是物体类）
		else:
			inst_uniq_objects = inst_uniq[inst_uniq >= 1]  # False: 扰动所有非背景类（ID >= 1）

		if inst_uniq_objects.__len__() == 0:  # 若无可选实例
			return dict(
				labels_fakeErr=labels_source.copy(),  # 返回原标签
			)
		# 按比例随机选取实例
		inst_ids = np.random.choice(inst_uniq_objects, floor(inst_uniq_objects.__len__() * fraction), replace=False)

	# print(inst_uniq, 'remove', inst_ids)
	labels = labels_source.copy()  # 创建标签副本，避免修改原数据

	# 计算类别选择概率
	class_weights = 1 / np.maximum(np.array(list(class19_distribution.values())), 1e-6)
	class_weights /= class_weights.sum()  # 归一化

	for inst_id in inst_ids:
		inst_mask = instances == inst_id  # 获取当前实例的掩码

		inst_view = labels[inst_mask]  # 获取实例内的标签类别
		# 获取实例内标签的众数（即该实例的主要类别）
		obj_class = stats.mode(inst_view, axis=None).mode[0]

		if obj_class != invalid_class:  # 若实例不是无效类别
			tc = list(target_classes)
			try:
				tc.remove(obj_class)  # 移除原类别，避免自我替换
			except ValueError:
				print(f'Instance class {obj_class} not in set of classes {target_classes}')
			# 使用类别概率分布进行采样
			new_class = choices(tc, weights=list(class_weights[tc]), k=1)[0]

			labels[inst_mask] = new_class

	result = dict(
		# 		dist_lab = nearest_labels,
		labels_fakeErr=labels
	)

	return result


def tr_synthetic_swapAll_labels_onPred(pred_labels_trainIds, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	instance_map = tr_instances_from_semantics(labels, min_size=750)['instances']

	labels_swapped = tr_swap_labels(pred_labels_trainIds, instance_map, only_objects=False)['labels_fakeErr']

	return dict(
		instances=instance_map,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)

# import matplotlib.pyplot as plt
# def show_pred_labels(pred_labels_trainIds):
# 	plt.figure(figsize=(10, 5))
# 	plt.imshow(pred_labels_trainIds, cmap='gray')  # 灰度显示
# 	plt.axis('off')
# 	plt.savefig('pred_labels_trainIds_visualization.png')  # 保存到当前路径
#
# # 假设有19类
# trainId_colors = np.array([
#     (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
#     (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
#     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
#     (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
#     (  0, 80,100), (  0,  0,230), (119, 11, 32)
# ], dtype=np.uint8)  # (19, 3)
# def visualize_labels(pred_labels_trainIds):
# 	color_image = np.zeros((pred_labels_trainIds.shape[0], pred_labels_trainIds.shape[1], 3), dtype=np.uint8)
# 	for trainId in range(len(trainId_colors)):
# 		color_image[pred_labels_trainIds == trainId] = trainId_colors[trainId]
#
# 	plt.figure(figsize=(10, 5))
# 	plt.imshow(color_image)
# 	plt.title('Colored Predicted Labels')
# 	plt.axis('off')
# 	plt.savefig('pred_labels_trainIds_visualization_color.png')  # 保存到当前路径
#
# def overlay_semseg_errors(labels_swapped, semseg_errors):
#     labels_color = np.zeros((labels_swapped.shape[0], labels_swapped.shape[1], 3), dtype=np.uint8)
#     for trainId in range(len(trainId_colors)):
#         labels_color[labels_swapped == trainId] = trainId_colors[trainId]
#
#     semseg_errors = semseg_errors.astype(np.uint8)  # 转成0/1
#     contours, _ = cv2.findContours(semseg_errors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 第三步：在 labels_color 上画红色边界框
#     labels_with_errors = labels_color.copy()
#     cv2.drawContours(labels_with_errors, contours, -1, (255, 0, 0), 2)  # 红色，线宽1
#
#     plt.figure(figsize=(12, 6))
#     plt.imshow(labels_with_errors)
#     plt.axis('off')
#     plt.savefig('labels_with_semseg_errors.png')


# 0.1. 实现对前景标签的随机替换（foreground label swapping）
def tr_synthetic_swapFgd_labels(pred_labels_trainIds, instances=None, swap_fraction=0.2, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	if instances is None:
		instances_encode_class = False
		instances = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']
	else:
		instances_encode_class = True  # gt instances

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds,
		instances,
		only_objects=instances_encode_class,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,  # 在感兴趣区域里被故意篡改（引入假错误）的位置, 即交换的位置标记
	)


def tr_synthetic_swapFgd_labels_onGT(pred_labels_trainIds, instances, only_objects=True, swap_fraction=0.2, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds, 
		instances, 
		only_objects=only_objects,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)


def tr_synthetic_swapAll_labels(pred_labels_trainIds, instances, swap_fraction=0.2, allow_road=False, min_size=500, **_):
	"""
	Swap background connected-components in addition to object instances.
	"""
	
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255
	
	obj_mask = instances >= 24000
	
	labels_for_cc = pred_labels_trainIds.copy()
	labels_for_cc[CTC_ROI_neg] = 255 # exclude outside of ROI
	labels_for_cc[obj_mask] = 255 # exclude objects, they have their own instances
	
	if not allow_road:
		labels_for_cc[labels_for_cc == 0] = 255
	
	stuff_instances = tr_instances_from_semantics(
		labels_for_cc, 
		min_size=min_size,
		forbidden_classes=[255],
	)['instances']
	
# 	show([stuff_instances, stuff_instances == 0])
	
	new_instances = stuff_instances
	new_instances[obj_mask] = instances[obj_mask]

	# Pass this instance map to the standard swapper
	
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds, 
		new_instances, 
		only_objects = False,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)


def tr_synthetic_swapNdisap(pred_labels_trainIds, instances, **_):
	labels_orig = pred_labels_trainIds.copy()
	labels_orig[CTC_ROI_neg] = 255
	
	P_SWAP = 0.3
	P_DISAP = 0.3

	inst_uniq = np.unique(instances)
	inst_uniq_objects = inst_uniq[inst_uniq >= 24000]

	num_obj = inst_uniq_objects.__len__()
	
	if num_obj == 0:
		labels = labels_orig
		
	else:

		num_disap = int(np.floor(P_DISAP * num_obj))
		num_swap = int(np.floor(P_SWAP * num_obj))

		inst_uniq_objects = np.random.permutation(inst_uniq_objects)

		inst_to_disap = inst_uniq_objects[:num_disap]
		inst_to_swap = inst_uniq_objects[num_disap:num_disap+num_swap]


		labels_orig = pred_labels_trainIds.copy()
		labels_orig[CTC_ROI_neg] = 255

		labels = tr_disappear_inst(
			labels_orig,
			instances,
			inst_ids=inst_to_disap
		)['labels_fakeErr']


		labels = tr_swap_labels(
			labels,
			instances,
			inst_ids=inst_to_swap,
		)['labels_fakeErr']
	
	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels,
		semseg_errors=(labels != labels_orig) & CTC_ROI,
	)

	
	
	
	
	
	
	
tr_show_dis = TrsChain(
	tr_disappear_inst,
	SemSegLabelsToColorImg(
		{'labels_fakeErr': 'labels_fakeErr_colorimg', 'labels_source': 'labels_source_colorimg'},
		CityscapesLabelInfo.colors_by_id,
	),
	TrShow(['labels_source_colorimg', 'labels_fakeErr_colorimg']),
	TrSemSegLabelTranslation(
		fields={'labels_fakeErr': 'pred_labels'},
		table=CityscapesLabelInfo.table_label_to_trainId,
	),
	tr_label_to_validEval,
	tr_get_errors,
	TrShow('semseg_errors'),
)

BUS_UNK_TRAINID = [CityscapesLabelInfo.name2label[n].trainId for n in ['bus', 'truck', 'train']]
BUS_UNK_ID = [CityscapesLabelInfo.name2label[n].id for n in ['bus', 'truck', 'train']]

CAR_TRAINID = CityscapesLabelInfo.name2label['car'].trainId

OUT_OF_ROI_ID = [CityscapesLabelInfo.name2label[n].id for n in ['ego vehicle', 'rectification border', 'out of roi']]

def tr_bus_errors(pred_labels, labels, labels_validEval, **_):
	bus_mask = np.any([labels == c for c in BUS_UNK_TRAINID], axis=0)

	pred_car_mask = pred_labels == CAR_TRAINID

	bus_pred_as_nocar_mask = bus_mask & (~pred_car_mask)

	bus_as_car = labels.copy()
	bus_as_car[bus_mask] = CAR_TRAINID

	semseg_errors_busiscar = (bus_as_car != pred_labels) & labels_validEval

	return dict(
		bus_mask = bus_mask,
		bus_pred_as_nocar_mask = bus_pred_as_nocar_mask,
		semseg_errors_busiscar = semseg_errors_busiscar,
	)

def tr_bus_error_simple(labels, **_):
	return dict(
		bus_mask = np.any([labels == c for c in BUS_UNK_TRAINID], axis=0),
	)


def tr_unlabeled_error_simple(labels_source, labels_validEval, **_):

	mask_out_of_roi = np.any([labels_source == c for c in OUT_OF_ROI_ID], axis=0)

	unlabeled_mask = ~( mask_out_of_roi | labels_validEval )

	return dict(
		unlabeled_mask = unlabeled_mask,
		labels_validEval = mask_out_of_roi,
	)
