import os
import numpy as np
import torch
import argparse
import random
import shutil
import datetime
import time
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import MultiStepLR
from torch.amp import autocast, GradScaler
import glob

from dataset import MultiModal_Dataset, MultiModal_WithLVI_Dataset
from Networks import *


def test_one_epoch(net, test_dataloader, loss_fc, epoch, args, phase='test'):
    print(f'--------Start {phase} | epoch:{epoch + 1}/{epoch + 1}-------')

    with torch.no_grad():
        net.eval()
        test_epoch_ID = []
        test_epoch_class_label = []
        test_epoch_one_hot_label = []
        test_epoch_pred_scores = []
        test_epoch_pred_class = []
        test_epoch_loss = []
        # for i, (ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label) in enumerate(test_dataloader):
        start = time.perf_counter()  # 记录开始时刻（高精度）
        for i, (ID, features, LVI_Label, SLN_Label) in enumerate(test_dataloader):
            
            features_on_device = [feature.cuda(non_blocking=True).float() for feature in features] # 被隐式地解释为 'cuda:0'

            # img_MR_DCE = img_MR_DCE.cuda().float() # 被隐式地解释为 'cuda:0'
            # img_MR_DWI = img_MR_DWI.cuda().float()
            # img_MG_CC = img_MG_CC.cuda().float()
            # img_MG_MLO = img_MG_MLO.cuda().float()
            # img_US_H = img_US_H.cuda().float()
            # img_US_V = img_US_V.cuda().float()
            # clinical = clinical_features.cuda().float()

            if args.pred_label == 'LVI':
                labels = LVI_Label.cuda(non_blocking=True).long()
            elif args.pred_label == 'SLN':
                labels = SLN_Label.cuda(non_blocking=True).long()
            labels_one_hot = torch.zeros((labels.size(0), args.num_classes),dtype=torch.float32,device=labels.device)\
                                                                            .scatter_(1, labels.unsqueeze(1), 1) # 在GPU上
            # # img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,x
            # # _,_,_,_,_,_,_,outputs = net(img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical)
            # outputs = net(*features_on_device)[-1]  # 取最后一个输出
            # loss = loss_fc(outputs, labels)

            # with torch.no_grad():
            # with autocast('cuda'):  # 自动混合精度
            outputs = net(features_on_device)[-1] # 取最后一个输出
            loss = loss_fc(outputs, labels)
            outputs_softmax = torch.softmax(outputs, dim=1)  # 在GPU上
            predicted = torch.argmax(outputs_softmax, dim=1) # 在GPU上

            # 收集数据：一次性移动到CPU并转换为所需格式
            test_epoch_pred_scores.append(outputs_softmax.cpu())  # 保持列表中的张量格式，后面处理
            test_epoch_one_hot_label.append(labels_one_hot.cpu()) # 保持列表中的张量格式，后面处理
            test_epoch_loss.append(loss.item())
            test_epoch_class_label.append(labels.cpu().numpy())
            test_epoch_pred_class.append(predicted.cpu().numpy())
            test_epoch_ID.extend(ID) # 使用 extend 因为 ID 通常是批次的列表
            print(f'[{epoch + 1}/{epoch + 1}, {i + 1}/{len(test_dataloader)}] test loss: {loss.item():.10f}')

    end = time.perf_counter()    # MR 570秒
    elapsed = end - start
    print(f"数据加载时间: {elapsed:.6f} 秒")

    return test_epoch_ID ,test_epoch_class_label,test_epoch_one_hot_label,\
                test_epoch_pred_scores,test_epoch_pred_class,test_epoch_loss


def val(net, loss_fc, args, save_dir, val_dataloader):

    # 开启训练日志
    val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    print(f'save_dir: {save_dir}')
    val_writer.add_scalar(save_dir, 1, 0)  # step=0 即可 # identifier 标记

    epoch = 0
    # ----------测试 验证数据----------
    val_epoch_ID, val_epoch_class_label, val_epoch_one_hot_label, \
        val_epoch_pred_scores, val_epoch_pred_class, val_epoch_loss \
            = test_one_epoch(net, val_dataloader, loss_fc, epoch, args, phase='val')

    # ----------结果整合----------
    # val_epoch_ID = np.concatenate(val_epoch_ID) # ID 已经是列表

    val_epoch_class_label = np.concatenate(val_epoch_class_label)
    val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()
    val_epoch_pred_class = np.concatenate(val_epoch_pred_class)

    # 计算AUC,ACC,Loss
    val_AUC = roc_auc_score(val_epoch_class_label, val_epoch_pred_scores[:,1])
    val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)
    val_epoch_loss = np.mean(val_epoch_loss)

    print(f'epoch:{epoch + 1}/{epoch + 1}')
    print(f'val loss: {val_epoch_loss:.3f}, val AUC: {val_AUC:.3f}, val ACC: {val_ACC:.3f}')

    val_writer.add_scalar('Loss', val_epoch_loss, epoch)
    val_writer.add_scalar('AUC', val_AUC, epoch)
    val_writer.add_scalar('ACC', val_ACC, epoch)
    np.save(os.path.join(save_dir, 'log/val/Truelabel_{}.npy'.format(epoch)), val_epoch_class_label)
    np.save(os.path.join(save_dir, 'log/val/Predlabel_{}.npy'.format(epoch)), val_epoch_pred_class)
    np.save(os.path.join(save_dir, 'log/val/Predscores_{}.npy'.format(epoch)), val_epoch_pred_scores)
    np.save(os.path.join(save_dir, 'log/val/ID_{}.npy'.format(epoch)), val_epoch_ID)
    val_writer.close()



def parse_args(gpus, batch_size, random_seed, pred_label, model_base_path, model_auc_name,\
                    data_path, center, split_path, metadata_df_path, lvi_preded_df_path, \
                    dataset_type, US_pad, MG_pad, MR_pad, \
                    model_name, modality, lvi_type ,num_classes, channels, gtv_type, clinical_inchannels):
    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--gpus', type=str, default=gpus, help='which gpu is used')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument('--random_seed', type=int, default=random_seed, help='random seed')
    parser.add_argument('--pred_label', type=str, default=pred_label, help='prediction label')
    parser.add_argument('--model_base_path', type=str, default=model_base_path, help='model base path')
    parser.add_argument('--model_auc_name', type=str, default=model_auc_name, help='model auc name')
    # data args
    parser.add_argument('--data_path', type=str, default=data_path, help='data path')
    parser.add_argument('--center', type=str, default=center, help='train center')
    parser.add_argument('--split_path', type=str, default=split_path, help='split path')
    parser.add_argument('--metadata_df_path', type=str, default=metadata_df_path, help='metadata df path')
    parser.add_argument('--lvi_preded_df_path', type=str, default=lvi_preded_df_path, help='lvi preded df path')
    parser.add_argument('--dataset_type', type=str, default=dataset_type, help='dataset type:all_img or spatial_img')
    parser.add_argument('--US_pad', type=int, default=US_pad, help='US pad of the spatial img')
    parser.add_argument('--MG_pad', type=int, default=MG_pad, help='MG pad of the spatial img')
    parser.add_argument('--MR_pad', type=int, default=MR_pad, help='MR pad of the spatial img')
    # model args
    parser.add_argument('--model_name', type=str, default=model_name, help='model name')
    parser.add_argument('--modality', type=str, default=modality, help='modality')
    parser.add_argument('--lvi_type', type=str, default=lvi_type, help='lvi type')
    parser.add_argument('--num_classes', type=int, default=num_classes, help='number of classes')
    parser.add_argument('--channels', type=int, default=channels, help='number of channels')
    parser.add_argument('--gtv_type', type=str, default=gtv_type, help='gtv type')
    parser.add_argument('--clinical_inchannels', type=int, default=clinical_inchannels, help='number of clinical features')

    return parser.parse_args() #如果有 --modality MG，就覆盖为 'MG'


if __name__ == '__main__':

    # -----------------------------------------验证参数-----------------------------------------
    random_seed = 42
    # date = f"{datetime.datetime.now().month:02d}{datetime.datetime.now().day:02d}" # 开始训练日期

    center_list = []
    data_path_list = []  # 测试数据路径
    split_path_list = []
    metadata_df_path_list = []
    lvi_preded_df_path_list = []
    
    dataset_type = 'all_img' 

    gtv_types = ['all','origin','img']  # all(3 channels), origin(2 channels), img(2 channels)
    gtv_type = gtv_types[1] # 先只使用 origin

    batch_size = 10 #10 4 2 batch size小，信息范围小，不容易拟合/batch size 大，信息范围大，容易过拟合
    gpus = '0'

    # -----------------------------------------模型参数-----------------------------------------

    modality_list = ['MRMGUS_Clinical_LVI','MRMGUS_Clinical','MRMGUS','MRMG','MRUS','MGUS','MR','MG','US','Clinical']
    
    num_classes = 2 # 分类类别数
    channels = 2 
    clinical_inchannels = 31 


    model_base_path =  r'...' #先预测SLN再微调预测LVI
    model_auc_name = r'...'
    model_name = 'MF_ResNet_AttnFusion'
    modality = 'MRMGUS_Clinical'
    pred_label = 'LVI'
    lvi_type = 'null'



    # -----------------------------------------模型配置-----------------------------------------
    set_determinism(random_seed) # 设置随机种子
    print(f"随机种子已设置为: {random_seed}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus) # 规定程序可以看见那些GPU

    loss_fc = CrossEntropyLoss() # 损失函数

    # -----------------------------------------遍历各中心数据，验证-----------------------------------------
    for index in range(len(center_list)):
        data_path = data_path_list[index]
        center = center_list[index]
        split_path = split_path_list[index]
        metadata_df_path = metadata_df_path_list[index]
        lvi_preded_df_path = lvi_preded_df_path_list[index]

        # 前面的参数集中管理，parse_args执行，命令行传入参数覆盖
        args = parse_args(gpus, batch_size, random_seed, pred_label, model_base_path, model_auc_name,\
                            data_path, center, split_path, metadata_df_path, lvi_preded_df_path, \
                            dataset_type, US_pad, MG_pad, MR_pad, \
                            model_name, modality, lvi_type,num_classes, channels, gtv_type, clinical_inchannels)
        # ------------------------------------------------------------------------------------------
        trained_results = r'/root/epfs/trained_log_results/'+args.pred_label
        model_path = glob.glob(os.path.join(trained_results, args.model_base_path, args.model_auc_name))[0]
        print(model_path)

        if args.model_name == 'MF_ResNet_FC':
            net = MF_ResNet_FC(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                        input_size_US=(384,512),clinical_features=clinical_inchannels, num_classes=2, dropout=0.1)
        elif args.model_name == 'MF_ResNet_AttnFusion':
            net = MF_ResNet_AttnFusion(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                            input_size_US=(384,512),clinical_features=clinical_inchannels, num_classes=2, dropout=0.1)
        elif args.model_name == 'MF_ResNet_AttnFusion_LVIGuide':
            net = MF_ResNet_AttnFusion_LVIGuide(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                            input_size_US=(384,512),clinical_features=clinical_inchannels, num_classes=2, dropout=0.1)

        net_dict = net.state_dict() # 获得模型的参数字典
        pretrain_dict = torch.load(model_path, weights_only=True, map_location='cpu') # 加载训练好的模型字典到CPU
        net_dict.update(pretrain_dict) # 将适配后的训练权重更新到网络参数中
        net.load_state_dict(net_dict)  # 加载更新后的网络参数到模型中
        net = net.cuda() # 加载模型到GPU

        # -----------------------------------------结果目录-----------------------------------------
        center_result_path  = r'/root/epfs/val_log_results/{}/{}/{}'.format(args.pred_label, model_base_path, center)
        os.makedirs(center_result_path, exist_ok=True)
        print(center_result_path)

        # -----------------------------------------数据配置-----------------------------------------
        print('--------Data configuration-------')
        if args.dataset_type == 'all_img':
            if args.modality == 'MRMGUS_Clinical':
                val_dataset = MultiModal_Dataset(args, data_set='val', augment=False)
            elif args.modality == 'MRMGUS_Clinical_LVI':
                val_dataset = MultiModal_WithLVI_Dataset(args, data_set='val', augment=False)

       
        print(len(val_dataset))

        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True,  drop_last=False)
        print('--------Data configuration completed-------')

        # -----------------------------------------验证-----------------------------------------
        val(net, loss_fc, args, center_result_path, val_dataloader)

