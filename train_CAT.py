import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from models.CAT import CAT
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
torch.multiprocessing.set_sharing_strategy('file_system')

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, loss_scaler):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    loss_sv_ave = 0
    loss_st_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].as_tensor().to(args.device), batch["post_label"].float().as_tensor().to(args.device), batch['name']
        with torch.cuda.amp.autocast(enabled=args.amp):
            logit_map, loss_sv, loss_st = model(x)
            term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
            term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
            loss = term_seg_BCE + term_seg_Dice + loss_sv + loss_st
            if loss_scaler is not None:
                loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
                optimizer.zero_grad()
            else:        
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        if args.local_rank == 0:
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f, sv_loss=%2.5f, st_loss=%2.5f)" % (
                    args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item(), loss_sv.item(), loss_st.item())
            )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        loss_sv_ave += loss_sv.item()
        loss_st_ave += loss_st.item()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f, ave_sv_loss=%2.5f, ave_st_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), loss_sv_ave/len(epoch_iterator), loss_st_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), loss_sv_ave/len(epoch_iterator), loss_st_ave/len(epoch_iterator)

def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        print(name, image.shape)
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = F.sigmoid(pred)
        
        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    if args.local_rank == 0:
        with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                print(content)
                f.write(content)
                f.write('\n')
            content = 'Average | '
            for i in range(NUM_CLASS):
                content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][organ-1] / ave_organ_dice[1][organ-1])
            print(content)
            f.write(content)
            f.write('\n')
            

                        


def process(args):
    if args.dist:
        misc.init_distributed_mode(args)
    if args.distributed:
        args.num_tasks = misc.get_world_size()
        args.global_rank = misc.get_rank()
    
    print("dist:", dist.is_initialized())
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None
        
    device = torch.device(args.device)
    # prepare the 3D model
    model = CAT(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    args=args
                    )
    def model_size_in_gb(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_size_bytes = total_params * 4  # Assuming parameters are float32
        total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to GB
        return total_size_gb
    size_in_gb = model_size_in_gb(model)
    print(f"Model size: {size_in_gb} GB")
    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    args.local_rank = args.global_rank
    model.to(device=device)
    model.train()
    if args.dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # criterion and optimizer
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    while args.epoch < args.max_epoch:
        if args.dist:
            train_sampler.set_epoch(args.epoch)
        scheduler.step()
        
        loss_dice, loss_bce, loss_sv, loss_st = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, loss_scaler)
        if args.local_rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('train_sv_loss', loss_sv, args.epoch)
            writer.add_scalar('train_st_loss', loss_st, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)
        if (args.epoch % args.store_num == 0 and args.epoch != 0) and args.local_rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir('out/' + args.log_name):
                os.mkdir('out/' + args.log_name)
            torch.save(checkpoint, 'out/' + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='CAT', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt',  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')

    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
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
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=4, type=int, help='sample number in each ct')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                           '07', '08', '09', '12', '10_03', 
                                           '10_06', '10_07', '10_08', '10_09', '10_10'],
                                           help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--amp", action="store_true", help="use amp for training")
    parser.add_argument("--only_last", action="store_false", help="only atten last feat")
    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()