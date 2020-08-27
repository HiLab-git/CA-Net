import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.utils.data as Data
from utils.binary import assd
from distutils.version import LooseVersion

from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform

from Models.networks.network import Comprehensive_Atten_Unet

from utils.dice_loss import get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic

from time import *

Test_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet}

Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'ISIC2018': ISIC2018_transform}


def test_isic(test_loader, model):
    isic_dice = []
    isic_iou = []
    isic_assd = []
    infer_time = []

    model.eval()
    for step, (img, lab) in enumerate(test_loader):
        image = img.float().cuda()
        target = lab.float().cuda()

        # output, atten2_map, atten3_map = model(image)  # model output
        begin_time = time()
        output = model(image)
        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label

        # input_arr = np.squeeze(image.cpu().numpy()).astype(np.float32)
        label_arr = target_soft.cpu().numpy().astype(np.uint8)
        # label_shw = np.squeeze(target.cpu().numpy()).astype(np.uint8)
        output_arr = output_soft.cpu().byte().numpy().astype(np.uint8)

        isic_b_dice = val_dice_isic(output_soft, target_soft, args.num_classes)  # the dice accuracy
        isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)  # the iou accuracy
        isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])

        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()
        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
        isic_assd.append(isic_b_asd)

    # df = pd.DataFrame(data=dice_np)
    # df.to_csv(args.ckpt + '/refine_result.csv')
    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

    isic_assd_mean = np.average(isic_assd)
    isic_assd_std = np.std(isic_assd)

    all_time = np.sum(infer_time)
    print('The ISIC mean Accuracy: {isic_dice_mean: .4f}; The ISIC Accuracy std: {isic_dice_std: .4f}'.format(
        isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
    print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
        isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
    print('The ISIC mean assd: {isic_asd_mean: .4f}; The ISIC assd std: {isic_asd_std: .4f}'.format(
        isic_asd_mean=isic_assd_mean, isic_asd_std=isic_assd_std))
    print('The inference time: {time: .4f}'.format(time=all_time))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='U-net add Attention mechanism for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet_fetus')
    # Path related arguments
    parser.add_argument('--root_path', default='./data/ISIC2018_Task1_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save', default='./result',
                        help='folder to outoput result')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--epoch', type=int, default=300, metavar='N',
                        help='choose the specific epoch checkpoints')

    # other arguments
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--att_pos', default='dec', type=str,
                        help='where attention to plug in (enc, dec, enc\&dec)')
    parser.add_argument('--view', default='axial', type=str,
                        help='use what views data to test (for fetal MRI)')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)

    # loading the dataset
    print('loading the {0} dataset ...'.format('test'))
    testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test', transform=Test_Transform[args.data])
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    print('Loading is done\n')

    # Define model
    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        if args.data == 'Fetus':
            args.num_input = 1
            args.num_classes = 3
            args.out_size = (256, 256)
        elif args.data == 'ISIC2018':
            args.num_input = 3
            args.num_classes = 2
            args.out_size = (224, 300)
        model = Test_Model[args.id](args, args.num_input, args.num_classes).cuda()
        # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Load the trained best model
    modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        # start_epoch = checkpoint['epoch']

        # multi-GPU transfer to one GPU
        # model_dict = model.state_dict()
        # pretrained_dict = checkpoint['state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_dict.items():
        #     name = k[7:]
        #     new_state_dict[name] = v
        #
        # model_dict.update(new_state_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    test_isic(testloader, model)
