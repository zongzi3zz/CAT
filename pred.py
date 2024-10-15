import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from models.CAT_pred import CAT
from dataset.pred_dataloader import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):

    test_item = args.pretrain_weights.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_dir, test_item)
    pred_save_path = os.path.join(save_path,'predict') 
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].cuda(), batch["name"]
        
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            if template_key.startswith('10'):
                pred_path = os.path.join(pred_save_path,  name[b].split('/')[0], name[b].split('/')[1])
            else:
                pred_path = os.path.join(pred_save_path, name[b].split('/')[0])
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, pred_path, dataset_id = name[b].split('/')[0], case_id = name[b].split('/')[-1], args = args)
            pred_hard_post = torch.tensor(pred_hard_post)
            
            ### testing phase for this function
            one_channel_label_v1 = merge_label(pred_hard_post, name)
            batch['one_channel_label_v1'] = one_channel_label_v1.cpu()

            visualize_label(batch, pred_path, name[b].split('/')[-1], val_transforms)        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    ## logging
    parser.add_argument('--log_name', default='CAT_test', help='The path resume from checkpoint')
    parser.add_argument('--save_dir', default='CAT_pred', help='save dir')
    ## model load
    parser.add_argument('--pretrain_weights', default='./pretrained_weights/CAT_weights_part.pth', help='The path resume from checkpoint')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['Total'])
    parser.add_argument('--data_root_path', default='./CT_data/', help='data root path')
    parser.add_argument('--data_file_path', default='./datalist/', help='data txt path')
    parser.add_argument('--anatomical_prompts_paths', default='./prompts/vprompts_path.json', help='visual_prompt_path')
    parser.add_argument('--text_prompt_path', default='./prompts/tprompts_path.json', help='text_prompt_path')
    
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument("--only_last", action="store_false", help="only atten last feat")
    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = CAT(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    args=args
                    )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain_weights)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict, strict=True)
    print('Use pretrained weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
