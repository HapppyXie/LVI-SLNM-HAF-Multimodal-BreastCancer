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
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.amp import autocast, GradScaler
import random
from dataset import MultiModal_Dataset,MultiModal_WithLVI_Dataset
from Networks import *


def get_model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def train_one_epoch(net, augmentGPU, train_dataloader, optimizer, loss_fc, scaler, epoch, args):

    print(f'--------Start training | epoch:{epoch + 1}/{args.epochs}-------')
    net.train()
    train_epoch_ID = []
    train_epoch_class_label = []
    train_epoch_one_hot_label = []
    train_epoch_pred_scores = []
    train_epoch_pred_class = []
    train_epoch_loss = []

    clip_count = 0          # 统计裁剪次数
    nan_count = 0           # 统计 NaN 次数

    # ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label
    # for i, (ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label) in enumerate(train_dataloader):
    start = time.perf_counter()  # 记录开始时刻（高精度）
    for i, (ID, features, LVI_Label, SLN_Label) in enumerate(train_dataloader):
        # features_on_device = [feature.to('cuda:0', non_blocking=True).float() for feature in features]
        features_on_device = [feature.cuda().float() for feature in features] # 被隐式地解释为 'cuda:0'

        # img_MR_DCE = img_MR_DCE.cuda().float() # 被隐式地解释为 'cuda:0'
        # img_MR_DCE = img_MR_DCE.cuda().float()
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

        # -------------------------------------------------------------------------
        # FP32 
        # optimizer.zero_grad()
        # outputs = net(features_on_device)[-1] # 取最后一个输出 
        # loss = loss_fc(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # -------------------------------------------------------------------------
        optimizer.zero_grad()
        with autocast('cuda'):  # 自动混合精度 FP16,可能数值溢出产生nan
            outputs = net(features_on_device)[-1] # 取最后一个输出

            # 检测 outputs 是否异常，异常则跳过整个 batch
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                nan_count += 1
                print(f"[WARN] NaN/Inf detected | epoch={epoch} batch={i}")
                # optimizer.zero_grad()
                # continue

            loss = loss_fc(outputs, labels) # loss为0，因为分类头处缺少归一化，导致数值不稳定，产生极端值。
        scaler.scale(loss).backward()

        # scaler.unscale_(optimizer)  # 梯度裁剪，必须先 对梯度进行 unscale
        # grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        # if grad_norm > 1.0:
        #     print(str(grad_norm)+' 需要梯度裁剪')

        scaler.step(optimizer)
        scaler.update()
        # -------------------------------------------------------------------------

        # 使用 torch.no_grad() 块，因为后续操作不需要梯度。块内操作不会计算梯度，也不会更新模型参数。
        with torch.no_grad():
            outputs_softmax = torch.softmax(outputs, dim=1) # 在GPU上
            predicted = torch.argmax(outputs_softmax, dim=1) # 在GPU上
        
        # print(outputs_softmax)
        # print(predicted)

        # 收集数据：一次性移动到CPU并转换为所需格式
        train_epoch_pred_scores.append(outputs_softmax.cpu())  # 保持列表中的张量格式，后面处理
        train_epoch_one_hot_label.append(labels_one_hot.cpu()) # 保持列表中的张量格式，后面处理
        train_epoch_loss.append(loss.item())
        train_epoch_class_label.append(labels.cpu().numpy())
        train_epoch_pred_class.append(predicted.cpu().numpy())
        train_epoch_ID.extend(ID) # 使用 extend 因为 ID 通常是批次的列表
        print(f'[{epoch + 1}/{args.epochs}, {i + 1}/{len(train_dataloader)}] train loss: {loss.item():.10f}')


    end = time.perf_counter()    # MR 570秒
    elapsed = end - start
    print(f"数据加载时间: {elapsed:.6f} 秒")
    
    return net,train_epoch_ID, train_epoch_class_label, train_epoch_one_hot_label, \
            train_epoch_pred_scores, train_epoch_pred_class, train_epoch_loss


def test_one_epoch(net, test_dataloader, loss_fc, epoch, args,phase='test'):
    print(f'--------Start {phase} | epoch:{epoch + 1}/{args.epochs}-------')

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
            # features_on_device = [feature.to('cuda:0', non_blocking=True).float() for feature in features]
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
            with autocast('cuda'):  # 自动混合精度 FP16
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
            print(f'[{epoch + 1}/{args.epochs}, {i + 1}/{len(test_dataloader)}] test loss: {loss.item():.10f}')

    end = time.perf_counter()    # MR 570秒
    elapsed = end - start
    print(f"数据加载时间: {elapsed:.6f} 秒")

    return test_epoch_ID ,test_epoch_class_label,test_epoch_one_hot_label,\
                test_epoch_pred_scores,test_epoch_pred_class,test_epoch_loss


def train_And_Test(net, optimizer, lr_scheduler, loss_fc, scaler, args, save_dir, \
            augmentGPU, train_dataloader, train_true_dataloader, val_dataloader, test_dataloader):

    # 开启训练日志
    train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
    train_true_writer = SummaryWriter(os.path.join(save_dir, 'log/train_true'), flush_secs=2)
    val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    test_writer = SummaryWriter(os.path.join(save_dir, 'log/test'), flush_secs=2)
    print(f'save_dir: {save_dir}')

    # identifier 标记
    train_writer.add_scalar(save_dir, 1, 0)  # step=0 即可

    best_AUC_val = 0
    best_AUC_test = 0

    for epoch in range(args.epochs):
        # 记录每个epoch的学习率
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        # ----------训练----------
        net, train_epoch_ID, train_epoch_class_label, train_epoch_one_hot_label, \
            train_epoch_pred_scores, train_epoch_pred_class, train_epoch_loss \
                = train_one_epoch(net, augmentGPU, train_dataloader, optimizer, loss_fc, scaler, epoch, args)
        lr_scheduler.step() # 训练一个epoch训练完，更新学习率

        # ----------测试 真实训练数据----------
        train_true_epoch_ID ,train_true_epoch_class_label,train_true_epoch_one_hot_label,\
            train_true_epoch_pred_scores,train_true_epoch_pred_class,train_true_epoch_loss\
                = test_one_epoch(net, train_true_dataloader, loss_fc, epoch, args, phase='train_true')

        # ----------测试 验证数据----------
        val_epoch_ID, val_epoch_class_label, val_epoch_one_hot_label, \
            val_epoch_pred_scores, val_epoch_pred_class,val_epoch_loss \
                = test_one_epoch(net, val_dataloader, loss_fc, epoch, args, phase='val')

        # ----------测试 测试数据----------
        test_epoch_ID, test_epoch_class_label, test_epoch_one_hot_label,\
            test_epoch_pred_scores, test_epoch_pred_class,test_epoch_loss = \
                test_one_epoch(net, test_dataloader, loss_fc, epoch, args, phase='test')


        # ----------结果整合----------
        # train_epoch_ID = np.concatenate(train_epoch_ID) # ID 已经是列表
        # train_true_epoch_ID = np.concatenate(train_true_epoch_ID)
        # val_epoch_ID = np.concatenate(val_epoch_ID)
        # test_epoch_ID = np.concatenate(test_epoch_ID)

        train_epoch_class_label = np.concatenate(train_epoch_class_label) # 真实标签 将各个批次的结果合并
        train_true_epoch_class_label = np.concatenate(train_true_epoch_class_label)
        val_epoch_class_label = np.concatenate(val_epoch_class_label)
        test_epoch_class_label = np.concatenate(test_epoch_class_label)

        train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy() # 预测概率
        train_true_epoch_pred_scores = torch.cat(train_true_epoch_pred_scores, dim=0).numpy()
        val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()
        test_epoch_pred_scores = torch.cat(test_epoch_pred_scores, dim=0).numpy()

        train_epoch_pred_class = np.concatenate(train_epoch_pred_class) # 预测标签
        train_true_epoch_pred_class = np.concatenate(train_true_epoch_pred_class)
        val_epoch_pred_class = np.concatenate(val_epoch_pred_class)
        test_epoch_pred_class = np.concatenate(test_epoch_pred_class)

        # 计算AUC,ACC,Loss
        train_AUC = roc_auc_score(train_epoch_class_label, train_epoch_pred_scores[:,1]) # 512,1  /  512,2 -> 512,1 取出预测阳性概率
        train_true_AUC = roc_auc_score(train_true_epoch_class_label, train_true_epoch_pred_scores[:,1])
        val_AUC = roc_auc_score(val_epoch_class_label, val_epoch_pred_scores[:,1])
        test_AUC = roc_auc_score(test_epoch_class_label, test_epoch_pred_scores[:,1])

        train_ACC = accuracy_score(train_epoch_class_label, train_epoch_pred_class)
        train_true_ACC = accuracy_score(train_true_epoch_class_label, train_true_epoch_pred_class)
        val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)
        test_ACC = accuracy_score(test_epoch_class_label, test_epoch_pred_class)

        train_epoch_loss = np.mean(train_epoch_loss)
        train_true_epoch_loss = np.mean(train_true_epoch_loss)
        val_epoch_loss = np.mean(val_epoch_loss)
        test_epoch_loss = np.mean(test_epoch_loss)

        print(f'epoch:{epoch + 1}/{args.epochs}')
        print(f'train loss: {train_epoch_loss:.3f}, train true loss: {train_true_epoch_loss:.3f}, val loss: {val_epoch_loss:.3f}, test loss: {test_epoch_loss:.3f}')
        print(f'train AUC: {train_AUC:.3f}, train true AUC: {train_true_AUC:.3f}, val AUC: {val_AUC:.3f}, test AUC: {test_AUC:.3f}')

        # 结果保存
        if val_AUC > best_AUC_val:
            best_AUC_val = val_AUC
            torch.save(net.state_dict(), os.path.join(save_dir, f'best_AUC_{val_AUC:.3f}_val.pth'))
            # torch.save(get_model_state_dict(net), os.path.join(save_dir, f'best_AUC_{val_AUC:.3f}_val.pth')) # 自动拆解
        if test_AUC > best_AUC_test:
            best_AUC_test = test_AUC
            torch.save(net.state_dict(), os.path.join(save_dir, f'best_AUC_{test_AUC:.3f}_test.pth'))
            # torch.save(get_model_state_dict(net), os.path.join(save_dir, f'best_AUC_{test_AUC:.3f}_test.pth')) # 自动拆解


        train_writer.add_scalar('Learning Rate', lr, epoch)
        train_writer.add_scalar('Loss', train_epoch_loss, epoch)
        train_writer.add_scalar('AUC', train_AUC, epoch)
        train_writer.add_scalar('ACC', train_ACC, epoch)
        np.save(os.path.join(save_dir, 'log/train/Truelabel_{}.npy'.format(epoch)), train_epoch_class_label)
        np.save(os.path.join(save_dir, 'log/train/Predlabel_{}.npy'.format(epoch)), train_epoch_pred_class)
        np.save(os.path.join(save_dir, 'log/train/Predscores_{}.npy'.format(epoch)), train_epoch_pred_scores)
        np.save(os.path.join(save_dir, 'log/train/ID_{}.npy'.format(epoch)), train_epoch_ID)

        train_true_writer.add_scalar('Loss', train_true_epoch_loss, epoch)
        train_true_writer.add_scalar('AUC', train_true_AUC, epoch)
        train_true_writer.add_scalar('ACC', train_true_ACC, epoch)
        np.save(os.path.join(save_dir, 'log/train_true/Truelabel_{}.npy'.format(epoch)), train_true_epoch_class_label)
        np.save(os.path.join(save_dir, 'log/train_true/Predlabel_{}.npy'.format(epoch)), train_true_epoch_pred_class)
        np.save(os.path.join(save_dir, 'log/train_true/Predscores_{}.npy'.format(epoch)), train_true_epoch_pred_scores)
        np.save(os.path.join(save_dir, 'log/train_true/ID_{}.npy'.format(epoch)), train_true_epoch_ID)

        val_writer.add_scalar('Loss', val_epoch_loss, epoch)
        val_writer.add_scalar('AUC', val_AUC, epoch)
        val_writer.add_scalar('ACC', val_ACC, epoch)
        val_writer.add_scalar('best_AUC_val', best_AUC_val, epoch)
        np.save(os.path.join(save_dir, 'log/val/Truelabel_{}.npy'.format(epoch)), val_epoch_class_label)
        np.save(os.path.join(save_dir, 'log/val/Predlabel_{}.npy'.format(epoch)), val_epoch_pred_class)
        np.save(os.path.join(save_dir, 'log/val/Predscores_{}.npy'.format(epoch)), val_epoch_pred_scores)
        np.save(os.path.join(save_dir, 'log/val/ID_{}.npy'.format(epoch)), val_epoch_ID)

        test_writer.add_scalar('Loss', test_epoch_loss, epoch)
        test_writer.add_scalar('AUC', test_AUC, epoch)
        test_writer.add_scalar('ACC', test_ACC, epoch)
        test_writer.add_scalar('best_AUC_test', best_AUC_test, epoch)
        np.save(os.path.join(save_dir, 'log/test/Truelabel_{}.npy'.format(epoch)), test_epoch_class_label)
        np.save(os.path.join(save_dir, 'log/test/Predlabel_{}.npy'.format(epoch)), test_epoch_pred_class)
        np.save(os.path.join(save_dir, 'log/test/Predscores_{}.npy'.format(epoch)), test_epoch_pred_scores)
        np.save(os.path.join(save_dir, 'log/test/ID_{}.npy'.format(epoch)), test_epoch_ID)

        if epoch + 1 == args.epochs:
            torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))
            # torch.save(get_model_state_dict(net), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth')) # 自动拆解


    train_writer.close()
    train_true_writer.close()
    val_writer.close()
    test_writer.close()
    print('saved_model_path:', save_dir)


def parse_args(gpus, batch_size ,epochs, random_seed,\
                    learning_rate, weight_decay, date, loss_threshold, pred_label,\
                    data_path, center, split_path, metadata_df_path, lvi_preded_df_path, result_save_path,\
                    transfer_learning_flag, dataset_type,US_pad,MG_pad,MR_pad,\
                    model_name, modality, lvi_type, num_classes, channels, gtv_type, 
                    clinical_inchannels, dropout, decline_epoch):
    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--gpus', type=str, default=gpus, help='which gpu is used')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument('--epochs', type=int, default=epochs, help='all epochs')
    parser.add_argument('--random_seed', type=int, default=random_seed, help='random seed')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=weight_decay, help='L2 regularization')
    parser.add_argument('--date', type=str, default=date, help='experiment date')
    parser.add_argument('--loss_threshold', type=float, default=loss_threshold, help='loss threshold')
    parser.add_argument('--pred_label', type=str, default=pred_label, help='prediction label')
    parser.add_argument('--transfer_learning_flag', type=bool, default=transfer_learning_flag, help='transfer learning flag')
    # data args
    parser.add_argument('--data_path', type=str, default=data_path, help='data path')
    parser.add_argument('--center', type=str, default=center, help='train center')
    parser.add_argument('--split_path', type=str, default=split_path, help='split path')
    parser.add_argument('--metadata_df_path', type=str, default=metadata_df_path, help='metadata df path')
    parser.add_argument('--lvi_preded_df_path', type=str, default=lvi_preded_df_path, help='lvi preded df path')
    parser.add_argument('--result_save_path', type=str, default=result_save_path, help='result save path')
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
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout rate')
    parser.add_argument('--decline_epoch', type=float, default=decline_epoch, help='lr decline epoch')

    return parser.parse_args()


if __name__ == '__main__':
    
    # -----------------------------------------训练参数-----------------------------------------
    
    random_seed = 42
    date = f"{datetime.datetime.now().month:02d}{datetime.datetime.now().day:02d}" # 开始训练日期
    loss_threshold = 0.0   # 损失阈值
    learning_rate = 0.00001  # 最大学习率  0.0001
    weight_decay = 0.00005    # 0.001 权重衰减系数 0.00005 (AdamW默认使用L2正则化)

    center = r'...'
    data_path = r'...' # 训练测试数据路径
    split_path = r'...' # 训练测试数据划分文件路径
    metadata_df_path = r'...' # 临床病理文件路径
    lvi_preded_df_path = r'...' # LVI 预测文件路径
    result_save_path = r'...'  # 存放训练结果的目录

    transfer_learning_flag = False # 迁移学习
    # transfer_learning_flag = True

    # pred_label = 'LVI' 
    pred_label = 'SLN'

    dataset_type = 'all_img'

    gtv_types = ['all','origin','img']  # all(3 channels), origin(2 channels), img(2 channels)
    gtv_type = gtv_types[1] # 先只使用 origin

    # 4,6,8,10,12,14,16
    batch_size = 10 
    epochs = 100
    gpus = '0'

    # -----------------------------------------模型参数-----------------------------------------

    modality_list = ['MRMGUS_Clinical_LVI','MRMGUS_Clinical','MRMGUS','MRMG','MRUS','MGUS','MR','MG','US','Clinical']
    lvi_type = 'Origin' # Pred, Origin
    # modality = 'MRMGUS_Clinical'
    modality = 'MRMGUS_Clinical_LVI'
    
    model_name = 'MF_ResNet_FC'
    # model_name = 'MF_ResNet_AttnFusion'
    # model_name = 'MF_ResNet_AttnFusion_LVIGuide'

    num_classes = 2 # 分类类别数
    channels = 2 # 3,2
    clinical_inchannels = 31 # 31
    dropout = 0   # 0, 0.1
    decline_epoch = 9  # 学习率开始下降的epoch = decline_epoch*10

    # 前面的参数集中管理，parse_args执行，命令行传入参数覆盖
    args = parse_args(gpus, batch_size, epochs, random_seed, \
                            learning_rate, weight_decay, date, \
                            loss_threshold, pred_label,\
                            data_path, center, split_path, metadata_df_path, lvi_preded_df_path, \
                            result_save_path,\
                            transfer_learning_flag, dataset_type, US_pad, MG_pad, MR_pad,\
                            model_name, modality, lvi_type, num_classes, channels, gtv_type, \
                        clinical_inchannels, dropout, decline_epoch)

    
    # -----------------------------------------结果目录-----------------------------------------
    if args.modality == 'Clinical':
        result_save_path2 = r'trained_log_results/{}/{}_{}_bs{}_{}_Date{}'\
            .format(args.pred_label, 'MLP', args.modality, args.batch_size, args.center, args.date)
            
    elif args.modality == 'MRMGUS_Clinical_LVI':
        result_save_path2 = r'trained_log_results/{}/{}_{}_bs{}_{}_{}_Date{}'\
            .format(args.pred_label, args.model_name, args.modality+args.lvi_type, \
                args.batch_size, args.dataset_type, args.center, args.date)
    else:
        result_save_path2 = r'trained_log_results/{}/{}_{}_bs{}_{}_{}_Date{}'\
            .format(args.pred_label, args.model_name, args.modality, \
                args.batch_size, args.dataset_type, args.center, args.date)
    
    result_save_path = os.path.join(args.result_save_path, result_save_path2)
    os.makedirs(result_save_path, exist_ok=True)
    # print(result_save_path)

    # -----------------------------------------训练配置-----------------------------------------
    set_determinism(args.random_seed) # 设置随机种子
    print(f"随机种子已设置为: {args.random_seed}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus) # 规定程序可以看见那些GPU

    if args.model_name == 'MF_ResNet_FC':
        net = MF_ResNet_FC(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                    input_size_US=(384,512),clinical_features=args.clinical_inchannels, num_classes=2, dropout=0.1)
    elif args.model_name == 'MF_ResNet_AttnFusion':
        net = MF_ResNet_AttnFusion(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                        input_size_US=(384,512),clinical_features=args.clinical_inchannels, num_classes=2, dropout=0.1)
    elif args.model_name == 'MF_ResNet_AttnFusion_LVIGuide':
        net = MF_ResNet_AttnFusion_LVIGuide(args.modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                        input_size_US=(384,512),clinical_features=args.clinical_inchannels, num_classes=2, dropout=0.1)

    # -----------------------------------------配置-----------------------------------------
    if transfer_learning_flag == True:
        # MRMGUS_Clinical
        model_path = r'...'
        net_dict = net.state_dict() # 获得模型的参数字典
        pretrain_dict = torch.load(model_path, weights_only=True, map_location='cpu') # 加载训练好的模型字典到CPU
        net_dict.update(pretrain_dict) # 将适配后的训练权重更新到网络参数中
        net.load_state_dict(net_dict)  # 加载更新后的网络参数到模型中


    net = net.cuda() # 加载模型到GPU
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # 优化器
    lr_scheduler = MultiStepLR(optimizer, milestones=[int((args.decline_epoch / 10) * args.epochs)], gamma=0.1, last_epoch=-1) # 学习率调度器 90个epoch后开始调度
    loss_fc = CrossEntropyLoss() # 损失函数
    scaler = GradScaler('cuda') # 初始化梯度缩放器

    # -----------------------------------------数据配置-----------------------------------------
    print('--------Data configuration-------')
    if args.dataset_type == 'all_img':
        train_dataset = MultiModal_Dataset(args, data_set='train', augment=True) # 关闭CPU图像数据增强
        train_true_dataset = MultiModal_Dataset(args, data_set='train', augment=False)
        val_dataset = MultiModal_Dataset(args, data_set='val', augment=False)
        test_dataset = MultiModal_Dataset(args, data_set='test', augment=False)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    train_true_dataloader = DataLoader(dataset=train_true_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,  drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,  drop_last=False)
    # print(len(train_true_dataloader))
    # print(len(val_dataloader))
    # print(len(test_dataloader))
    print('--------Data configuration completed-------')


    # -----------------------------------------训练，测试-----------------------------------------

    train_And_Test(net, optimizer, lr_scheduler, loss_fc, scaler, args, result_save_path, \
                   augmentGPU, train_dataloader, train_true_dataloader, val_dataloader, test_dataloader)


