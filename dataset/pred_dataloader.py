from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

def get_loader(args):
    pred_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )


    ## pred dict part
    test_img = []
    test_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_test.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["img"].split('.')[0]
            test_img.append(os.path.join(args.data_root_path, each["img"]))
            test_name.append(name)
    data_dicts_test = [{'image': image, 'name': name}
                for image, name in zip(test_img, test_name)]
    print('pred len {}'.format(len(data_dicts_test)))
    

    if args.cache_dataset:
        pred_dataset = CacheDataset(data=data_dicts_test, transform=pred_transforms, cache_rate=args.cache_rate)
    else:
        pred_dataset = Dataset(data=data_dicts_test, transform=pred_transforms)
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return pred_loader, pred_transforms
