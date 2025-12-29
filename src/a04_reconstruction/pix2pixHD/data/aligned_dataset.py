### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import util.util as util

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)
        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, 'gtFine', phase)
        label_paths_all = make_dataset(label_dir)
        label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]


        image_dir = os.path.join(root, 'leftImg8bit', phase)
        image_paths = make_dataset(image_dir)

        if not opt.no_instance:
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.label_paths[index]
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.image_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.instance_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst).int() # sometimes inst_tensor is float and it crashes the dataloader

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'Cityscapes'