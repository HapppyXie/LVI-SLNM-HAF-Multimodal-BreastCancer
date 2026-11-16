from Nii_utils import NiiDataRead, NiiDataRead_2D
from Nii_utils import NiiDataWrite,NiiDataWrite2D
import torch
import pickle
import numpy as np
import pandas as pd
from monai.transforms import Compose, Rand2DElasticd, Rand3DElasticd, RandRotated, RandGaussianNoised
from data_augmentGPU import AugmentGPU
import os
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from utils import masked_Zscore_norm
import glob
from monai.utils import set_determinism
import argparse
import time
set_determinism(seed=42)


def Extract_clinical_features(metadata_df, ID):
    '''
    输入：metadata_df(临床数据Dataframe), ID(病人id)
    输出：该病人对应的Clinical_features列表
    '''
    # print(ID)
    Clinical_features = [] # 存储编码后的临床特征


    # 左/右乳
    affected_side = str(metadata_df.loc[metadata_df['ID号'] == ID, '左/右乳'].values[0])
    if '左' in affected_side and '右' not in affected_side:
        Clinical_features.append(torch.tensor([1, 0]))
    elif '右' in affected_side and '左' not in affected_side:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))


    # 术前肿物个数
    num_tumor = str(metadata_df.loc[metadata_df['ID号'] == ID, '术前肿物个数'].values[0])
    if '单发' in num_tumor and '单象限多发（多灶性）' not in num_tumor and '多象限多发（多中心）' not in num_tumor:
        Clinical_features.append(torch.tensor([1, 0, 0]))
    elif '单象限多发（多灶性）' in num_tumor and '多象限多发（多中心）' not in num_tumor and '单发' not in num_tumor:
        Clinical_features.append(torch.tensor([0, 1, 0]))
    elif '多象限多发（多中心）' in num_tumor and '单象限多发（多灶性）' not in num_tumor and '单发' not in num_tumor:
        Clinical_features.append(torch.tensor([0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0]))

    # 四个象限
    # Preoperative_four_location = str(metadata_df.loc[metadata_df['ID号'] == ID, '四个象限'].values[0])
    # four_location_list = [0, 0, 0, 0]
    # if '25' in Preoperative_four_location:
    #     four_location_list[0] = 1
    # if '26' in Preoperative_four_location:
    #     four_location_list[1] = 1
    # if '27' in Preoperative_four_location:
    #     four_location_list[2] = 1
    # if '28' in Preoperative_four_location:
    #     four_location_list[3] = 1
    # Clinical_features.append(torch.tensor(four_location_list))

    # 术前肿物_最可疑_位置
    Preoperative_mass_position = str(metadata_df.loc[metadata_df['ID号'] == ID, '肿物位置（九个象限）'].values[0])
    P_mass_p_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if '11' in Preoperative_mass_position:
        P_mass_p_list[0] = 1
    if '12' in Preoperative_mass_position:
        P_mass_p_list[1] = 1
    if '13' in Preoperative_mass_position:
        P_mass_p_list[2] = 1
    if '14' in Preoperative_mass_position:
        P_mass_p_list[3] = 1
    if '25' in Preoperative_mass_position:
        P_mass_p_list[4] = 1
    if '26' in Preoperative_mass_position:
        P_mass_p_list[5] = 1
    if '27' in Preoperative_mass_position:
        P_mass_p_list[6] = 1
    if '28' in Preoperative_mass_position:
        P_mass_p_list[7] = 1
    if '39' in Preoperative_mass_position:
        P_mass_p_list[8] = 1
    if '40' in Preoperative_mass_position:
        P_mass_p_list[9] = 1
    Clinical_features.append(torch.tensor(P_mass_p_list))


    # cT临床分期
    cT_stage = str(metadata_df.loc[metadata_df['ID号'] == ID, 'cT'].values[0])
    if 'T1' in cT_stage:
        Clinical_features.append(torch.tensor([1, 0, 0, 0]))
    elif 'T2' in cT_stage:
        Clinical_features.append(torch.tensor([0, 1, 0, 0]))
    elif 'T3' in cT_stage:
        Clinical_features.append(torch.tensor([0, 0, 1, 0]))
    elif 'T4' in cT_stage:
        Clinical_features.append(torch.tensor([0, 0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0]))

    # 临床N分期
    N_stage = str(metadata_df.loc[metadata_df['ID号'] == ID, 'Axillary US'].values[0])
    if 'Negative' in N_stage:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Suspicious' in N_stage:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))

    # 肿物处钙化_钼靶
    calcify_MG = str(metadata_df.loc[metadata_df['ID号'] == ID, '肿物处钙化_钼靶'].values[0])
    if '是' in calcify_MG:
        Clinical_features.append(torch.tensor([1]))
    elif '否' in calcify_MG:
        Clinical_features.append(torch.tensor([-1]))
    else:
        Clinical_features.append(torch.tensor([0]))

    # 乳腺密度_钼靶
    density_MG = str(metadata_df.loc[metadata_df['ID号'] == ID, '乳腺密度_钼靶'].values[0])
    if 'A' in density_MG:
        Clinical_features.append(torch.tensor([1, 0, 0, 0]))
    elif 'B' in density_MG:
        Clinical_features.append(torch.tensor([0, 1, 0, 0]))
    elif 'C' in density_MG:
        Clinical_features.append(torch.tensor([0, 0, 1, 0]))
    elif 'D' in density_MG:
        Clinical_features.append(torch.tensor([0, 0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0]))

    # 编码成张量
    for feature_name in ['首次确诊年龄（岁）', 'T_最大径cm_超声1_长', 'T_最大径cm_超声2_宽', 'T_最大径cm_钼靶','T_最大径cm_MRI']:
        feature_value = float(metadata_df.loc[metadata_df['ID号'] == ID, feature_name].values[0])
        Clinical_features.append(torch.tensor([feature_value]))
    
    return torch.cat(Clinical_features)


def Extract_lvi_features(lvi_type, lvi_preded_df, ID):
    '''
    输入：lvi_preded_df(预测lvi数据Dataframe), ID(病人id)
    输出：该病人对应的lvi_features列表
    '''
    # print(ID)
    # lvi_features = [] # 存储编码后的临床特征

    if lvi_type == 'Pred':
        # Pred Score 即淋巴脉管侵犯风险的预测概率
        LVI_Pred_Positive_Score = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'Pred Score'].values[0]
        # calculate the pred score of negative and positive for LVI
        lvi_features = torch.tensor([1-LVI_Pred_Positive_Score, LVI_Pred_Positive_Score])

    elif lvi_type == 'Origin':
        # True Label 淋巴脉管是否侵犯(真实标签)
        True_Label = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'True Label'].values[0]
        if True_Label == 0:
            lvi_features = torch.tensor([1, 0])
        elif True_Label == 1:
            lvi_features = torch.tensor([0, 1])

    # # Pred Score 即淋巴脉管侵犯风险
    # LVI_Pred_Positive_Score = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'Pred Score'].values[0]
    # # calculate the pred score of negative and positive for LVI
    # LVI_Pred_Score = torch.tensor([1-LVI_Pred_Positive_Score, LVI_Pred_Positive_Score])
    # # Pred Label 即淋巴脉管是否侵犯
    # Pred_Label = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'Pred Label'].values[0]
    # Pred_Label = torch.tensor(Pred_Label)
    # # Pred Label 即淋巴脉管是否侵犯(最佳阈值)
    # Pred_Label_Best_Threshold = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'Pred Label(Best Threshold)'].values[0]
    # Pred_Label_Best_Threshold = torch.tensor(Pred_Label_Best_Threshold)
    # # True Label 淋巴脉管是否侵犯(真实标签)
    # True_Label = lvi_preded_df.loc[lvi_preded_df['ID'] == ID, 'True Label'].values[0]
    # True_Label = torch.tensor(True_Label)

    return lvi_features


def Extract_pathology_features(metadata_df, ID):
    '''
    输入：metadata_df(临床数据Dataframe), ID(病人id)
    输出：该病人对应的pathology_features列表
    '''
    pathology_features = [] # 存储编码后的病理特征

    # 病理组织学类型
    pathological_types = str(metadata_df.loc[metadata_df['ID号'] == ID, '穿刺病理类型'].values[0])
    #  if '浸润性导管癌' in pathological_types:
    #     pathology_features.append(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]))
    # elif '浸润性小叶癌（含经典型小叶癌、实性经典型小叶癌）' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]))
    # elif '包裹性/实性乳头状（原位）癌' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]))
    # elif '浸润性癌（未分类，含单纯癌）' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]))
    # elif '黏液（腺）癌' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]))
    # elif '浸润性微乳头状癌' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]))
    # elif '导管原位癌/导管内癌（未分类）' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]))
    # elif '伴神经内分泌特征的癌（未分类，含小细胞癌）' in pathological_types:
    #     pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]))
    # else:
    #     pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]))

    if '浸润性导管癌' in pathological_types:
        pathology_features.append(torch.tensor([1, 0, 0, 0, 0, 0]))
    elif '浸润性小叶癌' in pathological_types:
        pathology_features.append(torch.tensor([0, 1, 0, 0, 0, 0]))
    elif '浸润性癌' in pathological_types:
        pathology_features.append(torch.tensor([0, 0, 1, 0, 0, 0]))
    elif '浸润性微乳头状癌' in pathological_types:
        pathology_features.append(torch.tensor([0, 0, 0, 1, 0, 0]))
    elif '导管原位癌' in pathological_types:
        pathology_features.append(torch.tensor([0, 0, 0, 0, 1, 0]))
    elif '其它' in pathological_types:
        pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0, 0, 0, 0, 0]))

    # 组织学分级
    histological_grading = str(metadata_df.loc[metadata_df['ID号'] == ID, '病理分级'].values[0])
    if 'I' in histological_grading and 'II' not in histological_grading and 'III' not in histological_grading:
        pathology_features.append(torch.tensor([1, 0, 0]))
    elif 'II' in histological_grading and 'III' not in histological_grading and 'I' not in histological_grading:
        pathology_features.append(torch.tensor([0, 1, 0]))
    elif 'III' in histological_grading and 'I' not in histological_grading and 'II' not in histological_grading:
        pathology_features.append(torch.tensor([0, 0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0, 0]))

    # ER
    ER = str(metadata_df.loc[metadata_df['ID号'] == ID, 'ER'].values[0])
    if '-' in ER:
        pathology_features.append(torch.tensor([1, 0]))
    elif '+' in ER:
        pathology_features.append(torch.tensor([0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0]))
    
    # 手术病理_PR
    PR = str(metadata_df.loc[metadata_df['ID号'] == ID, 'PR'].values[0])
    if '-' in PR:
        pathology_features.append(torch.tensor([1, 0]))
    elif '+' in PR:
        pathology_features.append(torch.tensor([0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0]))
    
    # 手术病理_HER2
    HER2 = str(metadata_df.loc[metadata_df['ID号'] == ID, 'HER2'].values[0])
    if '-' in HER2:
        pathology_features.append(torch.tensor([1, 0]))
    elif '+' in HER2:
        pathology_features.append(torch.tensor([0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0]))

    # for feature_name in ['Ki67']:
    #     feature_value = float(metadata_df.loc[metadata_df['ID号'] == ID, feature_name].values[0])
    #     pathology_features.append(torch.tensor([feature_value]))
    # 手术病理_Ki67
    Ki67 = str(metadata_df.loc[metadata_df['ID号'] == ID, 'Ki67'].values[0])
    if '<20%' in Ki67:
        pathology_features.append(torch.tensor([1, 0]))
    elif '>=20%' in Ki67:
        pathology_features.append(torch.tensor([0, 1]))
    else:
        pathology_features.append(torch.tensor([0, 0]))

    return torch.cat(pathology_features)


def extract_3D_GTVAndGTVnd1(img, GTV, GTVnd1):
    """
    输入: img(原始图像), GTV(GTV或GTVnd1, 原发灶或淋巴结区域), pad(扩展尺寸，以覆盖周围组织) 
    输出: final_GTV_And_GTVnd1(提取的GTV空间区域), img_GTV_And_GTVnd1(提取的GTV空间区域，从图像中抠出)

    功能: 返回 GTV并上GTVnd1的数据(原始GTV并上GTVnd1), 以及周围的组织区域
    """

    # 将GTV和GTVnd1区域合并
    GTV[GTVnd1 > 0] = 1
    # 添加立方体背景，方便展示
    final_GTV_And_GTVnd1 = np.zeros_like(GTV)
    final_GTV_And_GTVnd1[GTV > 0] = GTV[GTV > 0]

    # 用GTV和GTVnd1合并的区域 抠出图像区域（有强度差异），用有强度差异的区域进行引导
    img_GTV_And_GTVnd1 = np.zeros_like(img)
    img_GTV_And_GTVnd1[final_GTV_And_GTVnd1 > 0] = img[final_GTV_And_GTVnd1 > 0]

    return final_GTV_And_GTVnd1, img_GTV_And_GTVnd1


def extract_2D_MG_CC_GTV(img, GTV):
    '''
    输入: img(原始图像), GTV(GTV或GTVnd1, 原发灶或淋巴结区域), pad(扩展尺寸，以覆盖周围组织) 
    输出: final_img(提取的图像空间区域), final_GTV_And_GTVnd1(提取的GTV空间区域), img_GTV_And_GTVnd1(提取的GTV空间区域，从图像中抠出)

    功能: 提取GTV(或GTVnd1)的空间区域,以及周围的组织区域
    '''
    img_GTV = np.zeros_like(GTV)
    img_GTV[GTV > 0] = img[GTV > 0]

    return GTV, img_GTV


def extract_2D_MG_MLO_GTVAndGTVnd1(img, GTV, GTVnd1):

    '''
    输入: img(原始图像), GTV(GTV或GTVnd1, 原发灶或淋巴结区域), pad(扩展尺寸，以覆盖周围组织) 
    输出: final_img(提取的图像空间区域), final_GTV_And_GTVnd1(提取的GTV空间区域), img_GTV_And_GTVnd1(提取的GTV空间区域，从图像中抠出)

    功能: 提取GTV(或GTVnd1)的空间区域,以及周围的组织区域
    '''
    # 合并GTV和GTVnd1区域
    GTV[GTVnd1 > 0] = 1
    # 添加平面背景，方便展示
    final_GTV_And_GTVnd1 = np.zeros_like(GTV)
    final_GTV_And_GTVnd1[GTV > 0] = GTV[GTV > 0]

    # 用GTV和GTVnd1合并的区域 抠出图像区域（有强度差异），用有强度差异的区域进行引导
    img_GTV_And_GTVnd1 = np.zeros_like(img)
    img_GTV_And_GTVnd1[final_GTV_And_GTVnd1 > 0] = img[final_GTV_And_GTVnd1 > 0]

    return final_GTV_And_GTVnd1, img_GTV_And_GTVnd1


def extract_2D_US_GTV(img, GTV):
    '''
    输入: img(原始图像), GTV(GTV原发灶), pad(扩展尺寸，以覆盖周围组织) 
    输出: final_img(提取的图像空间区域), final_GTV_And_GTVnd1(提取的GTV空间区域), img_GTV_And_GTVnd1(提取的GTV空间区域，从图像中抠出)

    功能: 提取GTV(或GTVnd1)的空间区域,以及周围的组织区域
    '''

    img_GTV = np.zeros_like(GTV)
    img_GTV[GTV > 0] = img[GTV > 0]

    return GTV ,img_GTV

# -------------------------------------------------------------------------

class MultiModal_Dataset(Dataset):

    def __init__(self, args, data_set='train', augment=True):
        self.data_path = args.data_path
        split_data = pickle.load(open(args.split_path, 'rb'))
        # self.ID_list = split_data[data_set][0]
        # self.label_list = split_data[data_set][1]
        self.LVI_SLN_split_list = split_data[data_set]     # index key: ID, LVI Label, SLN Label
        self.augment = augment
        self.gtv_type = args.gtv_type
        self.modality = args.modality
        self.metadata_df = pd.read_csv(args.metadata_df_path, dtype={'ID号': str})

        # 是否进行数据增强
        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.LVI_SLN_split_list)

    def __getitem__(self, idx):
        ID = self.LVI_SLN_split_list[idx]['ID']
        LVI_Label = self.LVI_SLN_split_list[idx]['LVI Label']
        SLN_Label = self.LVI_SLN_split_list[idx]['SLN Label']
        # print(ID)

        # Clinical
        if 'Clinical' in self.modality:
            clinical_features = Extract_clinical_features(self.metadata_df, ID)


        # MR DCE
        if 'MR' in self.modality:
            # print('MR DCE')
            MR_DCE, MR_DCE_spacing, MR_DCE_origin, MR_DCE_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_N4.nii*'.format(ID)))[0])
            if MR_DCE.max() < 0:
                MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                            percentile_max=99.5)
            else:
                MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                            percentile_max=99.5)
            MR_DCE_GTV, MR_DCE_GTV_spacing, MR_DCE_GTV_origin, MR_DCE_GTV_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
            MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
            MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
            MR_DCE_GTVnd1, MR_DCE_GTVnd1_spacing, MR_DCE_GTVnd1_origin, MR_DCE_GTVnd1_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
            MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
            MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MR_DCE.shape != MR_DCE_GTV.shape:
                MR_DCE_GTV = transform.resize(MR_DCE_GTV, output_shape=MR_DCE.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DCE_GTV.dtype) # 直接转换数据类型
            if MR_DCE.shape != MR_DCE_GTVnd1.shape:
                MR_DCE_GTVnd1 = transform.resize(MR_DCE_GTVnd1, output_shape=MR_DCE.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DCE_GTVnd1.dtype) # 直接转换数据类型

            # 合并GTV和GTVnd1，并提取不同gtv_type的mask图像
            MR_DCE_GTV_And_GTVnd1, MR_DCE_img_GTV_And_GTVnd1 = \
                        extract_3D_GTVAndGTVnd1(MR_DCE, MR_DCE_GTV, MR_DCE_GTVnd1)

            img_MR_DCE = MR_DCE[np.newaxis, ...]
            img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MR_DCE = np.concatenate((MR_DCE_GTV_And_GTVnd1[np.newaxis, ...], MR_DCE_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MR_DCE = MR_DCE_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DCE = transform.resize(mask_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MR_DCE = MR_DCE_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DCE = transform.resize(mask_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MR_DCE.shape, mask_MR_DCE.shape)


            # MR DWI
            # print('MR DWI')
            MR_DWI, MR_DWI_spacing, MR_DWI_origin, MR_DWI_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI.nii*'.format(ID)))[0])
            if MR_DWI.max() < 0:
                MR_DWI = masked_Zscore_norm(MR_DWI, mask_min=np.min(MR_DWI), mask_max=np.max(MR_DWI), percentile_min=None,
                                            percentile_max=99.5)
            else:
                MR_DWI = masked_Zscore_norm(MR_DWI, mask_min=0, mask_max=np.max(MR_DWI), percentile_min=None,
                                            percentile_max=99.5)
            MR_DWI_GTV, MR_DWI_GTV_spacing, MR_DWI_GTV_origin, MR_DWI_GTV_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI_*GTV.nii*'.format(ID)))[0])
            MR_DWI_GTV[MR_DWI_GTV <= 0.5] = 0
            MR_DWI_GTV[MR_DWI_GTV > 0.5] = 1
            MR_DWI_GTVnd1, MR_DWI_GTVnd1_spacing, MR_DWI_GTVnd1_origin, MR_DWI_GTVnd1_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI_*GTV*nd1.nii*'.format(ID)))[0])
            MR_DWI_GTVnd1[MR_DWI_GTVnd1 <= 0.5] = 0
            MR_DWI_GTVnd1[MR_DWI_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MR_DWI.shape != MR_DWI_GTV.shape:
                MR_DWI_GTV = transform.resize(MR_DWI_GTV, output_shape=MR_DWI.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DWI_GTV.dtype) # 直接转换数据类型
            if MR_DWI.shape != MR_DWI_GTVnd1.shape:
                MR_DWI_GTVnd1 = transform.resize(MR_DWI_GTVnd1, output_shape=MR_DWI.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DWI_GTVnd1.dtype) # 直接转换数据类型
            # 合并GTV和GTVnd1，并提取不同gtv_type的mask图像
            MR_DWI_GTV_And_GTVnd1, MR_DWI_img_GTV_And_GTVnd1 = \
                        extract_3D_GTVAndGTVnd1(MR_DWI, MR_DWI_GTV, MR_DWI_GTVnd1)
            
            img_MR_DWI = MR_DWI[np.newaxis, ...]
            img_MR_DWI = transform.resize(img_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MR_DWI = np.concatenate((MR_DWI_GTV_And_GTVnd1[np.newaxis, ...], MR_DWI_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MR_DWI = transform.resize(mask_MR_DWI, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MR_DWI = MR_DWI_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DWI = transform.resize(mask_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MR_DWI = MR_DWI_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DWI = transform.resize(mask_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MR_DWI.shape, mask_MR_DWI.shape)


        # MG CC
        if 'MG' in self.modality:
            # print('MG CC')
            MG_CC, MG_CC_img_spacing, MG_CC_img_origin, MG_CC_img_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
            if MG_CC.max() <= 4095:
                MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                            percentile_max=99.5)
            else:
                MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                        percentile_max=99.5)
            MG_CC_GTV, MG_CC_GTV_spacing, MG_CC_GTV_origin, MG_CC_GTV_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
            MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
            MG_CC_GTV[MG_CC_GTV > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MG_CC.shape != MG_CC_GTV.shape:
                MG_CC_GTV = transform.resize(MG_CC_GTV, output_shape=MG_CC.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_CC_GTV.dtype) # 直接转换数据类型

            MG_CC_GTV, MG_CC_img_GTV = extract_2D_MG_CC_GTV(MG_CC, MG_CC_GTV)

            img_MG_CC = MG_CC[np.newaxis, ...]
            img_MG_CC = transform.resize(img_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MG_CC = np.concatenate((MG_CC_GTV[np.newaxis, ...], MG_CC_img_GTV[np.newaxis, ...]), axis=0)
                mask_MG_CC = transform.resize(mask_MG_CC, (2, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
                mask_MG_CC = transform.resize(mask_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MG_CC = MG_CC_img_GTV[np.newaxis, ...]
                mask_MG_CC = transform.resize(mask_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MG_CC.shape, mask_MG_CC.shape)


            # MG MLO
            # print('MG MLO')
            MG_MLO, MG_MLO_img_spacing, MG_MLO_img_origin, MG_MLO_img_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
            if MG_MLO.max() <= 4095:
                MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                        percentile_max=99.5)
            else:
                MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                        percentile_max=99.5)
            MG_MLO_GTV, MG_MLO_GTV_spacing, MG_MLO_GTV_origin, MG_MLO_GTV_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
            MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
            MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
            MG_MLO_GTVnd1,  MG_MLO_GTVnd1_spacing, MG_MLO_GTVnd1_origin, MG_MLO_GTVnd1_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
            MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
            MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MG_MLO.shape != MG_MLO_GTV.shape:
                MG_MLO_GTV = transform.resize(MG_MLO_GTV, output_shape=MG_MLO.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_MLO_GTV.dtype) # 直接转换数据类型
            if MG_MLO.shape != MG_MLO_GTVnd1.shape:
                MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, output_shape=MG_MLO.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_MLO_GTVnd1.dtype) # 直接转换数据类型

            MG_MLO_GTV_And_GTVnd1, MG_MLO_img_GTV_And_GTVnd1 = \
                        extract_2D_MG_MLO_GTVAndGTVnd1(MG_MLO, MG_MLO_GTV, MG_MLO_GTVnd1)

            img_MG_MLO = MG_MLO[np.newaxis, ...]
            img_MG_MLO = transform.resize(img_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MG_MLO = np.concatenate((MG_MLO_GTV_And_GTVnd1[np.newaxis, ...], MG_MLO_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MG_MLO = transform.resize(mask_MG_MLO, (2, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MG_MLO = MG_MLO_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MG_MLO = transform.resize(mask_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MG_MLO = MG_MLO_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MG_MLO = transform.resize(mask_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MG_MLO.shape, mask_MG_MLO.shape)


        # US H
        if 'US' in self.modality:
            # print('US H')
            US_H, US_H_spacing, US_H_origin, US_H_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_H.nii*'.format(ID)))[0])
            US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
            US_H_GTV, US_H_spacing, US_H_origin, US_H_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
            US_H_GTV[US_H_GTV <= 0.5] = 0
            US_H_GTV[US_H_GTV > 0.5] = 1
            
            # 检查数据尺寸，如果不匹配，则resize
            if US_H.shape != US_H_GTV.shape:
                US_H_GTV = transform.resize(US_H_GTV, output_shape=US_H.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(US_H_GTV.dtype) # 直接转换数据类型

            US_H_GTV, US_H_img_GTV = extract_2D_US_GTV(US_H, US_H_GTV)

            img_US_H = US_H[np.newaxis, ...]
            img_US_H = transform.resize(img_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_US_H = np.concatenate((US_H_GTV[np.newaxis, ...], US_H_img_GTV[np.newaxis, ...]), axis=0)
                mask_US_H = transform.resize(mask_US_H, (2, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_US_H = US_H_GTV[np.newaxis, ...]
                mask_US_H = transform.resize(mask_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_US_H = US_H_img_GTV[np.newaxis, ...]
                mask_US_H = transform.resize(mask_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_US_H.shape, mask_US_H.shape)

            # US V
            # print('US V')
            US_V, US_V_spacing, US_V_origin, US_V_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_V.nii*'.format(ID)))[0])
            US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
            US_V_GTV, US_V_spacing, US_V_origin, US_V_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
            US_V_GTV[US_V_GTV <= 0.5] = 0
            US_V_GTV[US_V_GTV > 0.5] = 1
            
            # 检查数据尺寸，如果不匹配，则resize
            if US_V.shape != US_V_GTV.shape:
                US_V_GTV = transform.resize(US_V_GTV, output_shape=US_V.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(US_V_GTV.dtype) # 直接转换数据类型

            US_V_GTV, US_V_img_GTV = extract_2D_US_GTV(US_V, US_V_GTV)

            img_US_V = US_V[np.newaxis, ...]
            img_US_V = transform.resize(img_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            
            # all, origin, img
            if self.gtv_type == 'all':
                mask_US_V = np.concatenate((US_V_GTV[np.newaxis, ...], US_V_img_GTV[np.newaxis, ...]), axis=0)
                mask_US_V = transform.resize(mask_US_V, (2, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_US_V = US_V_GTV[np.newaxis, ...]
                mask_US_V = transform.resize(mask_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_US_V = US_V_img_GTV[np.newaxis, ...]
                mask_US_V = transform.resize(mask_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_US_V.shape, mask_US_V.shape)


        if self.augment:
            if 'MR' in self.modality:
                augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
                img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)
                augmented_MR_DWI = self.transforms_3D({'img': img_MR_DWI, 'mask': mask_MR_DWI})
                img_MR_DWI = np.concatenate((augmented_MR_DWI['img'], augmented_MR_DWI['mask']), axis=0)

            if 'MG' in self.modality:
                augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
                img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)
                augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
                img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            if 'US' in self.modality:
                augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
                img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)
                augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
                img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)

        else:
            if 'MR' in self.modality:
                img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
                img_MR_DWI = np.concatenate((img_MR_DWI, mask_MR_DWI), axis=0)
            if 'MG' in self.modality:
                img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
                img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            if 'US' in self.modality:
                img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
                img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)

        if 'MR' in self.modality:
            img_MR_DCE = torch.from_numpy(img_MR_DCE)
            img_MR_DWI = torch.from_numpy(img_MR_DWI)
        if 'MG' in self.modality:
            img_MG_CC = torch.from_numpy(img_MG_CC)
            img_MG_MLO = torch.from_numpy(img_MG_MLO)
        if 'US' in self.modality:
            img_US_H = torch.from_numpy(img_US_H)
            img_US_V = torch.from_numpy(img_US_V)

        # LVI_Label 转换为0/1
        if LVI_Label == 'Visible':
            LVI_Label = torch.tensor(1)
        else:
            LVI_Label = torch.tensor(0)
        # SLN_Label 转换为0/1
        if SLN_Label == 'Positive':
            SLN_Label = torch.tensor(1)
        else:
            SLN_Label = torch.tensor(0)
                
        # return ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label,\
        #         MR_DCE_spacing, MR_DCE_origin, MR_DCE_direction,\
        #         MR_DCE_GTV_spacing, MR_DCE_GTV_origin, MR_DCE_GTV_direction,\
        #         MR_DCE_GTVnd1_spacing, MR_DCE_GTVnd1_origin, MR_DCE_GTVnd1_direction,\
        #         MR_DWI_spacing, MR_DWI_origin, MR_DWI_direction,\
        #         MR_DWI_GTV_spacing, MR_DWI_GTV_origin, MR_DWI_GTV_direction,\
        #         MR_DWI_GTVnd1_spacing, MR_DWI_GTVnd1_origin, MR_DWI_GTVnd1_direction,\
        #         MG_CC_img_spacing, MG_CC_img_origin, MG_CC_img_direction,\
        #         MG_CC_GTV_spacing, MG_CC_GTV_origin, MG_CC_GTV_direction,\
        #         MG_MLO_img_spacing, MG_MLO_img_origin, MG_MLO_img_direction,\
        #         MG_MLO_GTV_spacing, MG_MLO_GTV_origin, MG_MLO_GTV_direction,\
        #         MG_MLO_GTVnd1_spacing, MG_MLO_GTVnd1_origin, MG_MLO_GTVnd1_direction,\
        #         US_H_spacing, US_H_origin, US_H_direction,\
        #         US_V_spacing, US_V_origin, US_V_direction
    
        # return ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label

        # 收集数据
        data_list = []
        if 'MR' in self.modality:
            data_list.append(img_MR_DCE)
            data_list.append(img_MR_DWI)
        if 'MG' in self.modality:
            data_list.append(img_MG_CC)
            data_list.append(img_MG_MLO)
        if 'US' in self.modality:
            data_list.append(img_US_H)
            data_list.append(img_US_V)
        if 'Clinical' in self.modality:
            data_list.append(clinical_features)

        return ID, data_list, LVI_Label, SLN_Label

    def __len__(self):
        return self.len


class MultiModal_WithLVI_Dataset(Dataset):

    def __init__(self, args, data_set='train', augment=True):
        self.data_path = args.data_path
        split_data = pickle.load(open(args.split_path, 'rb'))
        # self.ID_list = split_data[data_set][0]
        # self.label_list = split_data[data_set][1]
        self.LVI_SLN_split_list = split_data[data_set]     # index key: ID, LVI Label, SLN Label
        self.augment = augment
        self.gtv_type = args.gtv_type
        self.modality = args.modality
        self.metadata_df = pd.read_csv(args.metadata_df_path, dtype={'ID号': str})
        self.lvi_preded_df = pd.read_csv(args.lvi_preded_df_path, dtype={'ID': str})
        self.lvi_type = args.lvi_type

        # 是否进行数据增强
        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.LVI_SLN_split_list)

    def __getitem__(self, idx):
        ID = self.LVI_SLN_split_list[idx]['ID']
        LVI_Label = self.LVI_SLN_split_list[idx]['LVI Label']
        SLN_Label = self.LVI_SLN_split_list[idx]['SLN Label']
        # print(ID)

        # Clinical
        if 'Clinical' in self.modality:
            clinical_features = Extract_clinical_features(self.metadata_df, ID)
        
        # LVI preded data
        if 'LVI' in self.modality:
            lvi_features = Extract_lvi_features(self.lvi_type, self.lvi_preded_df, ID)

        # MR DCE
        if 'MR' in self.modality:
            # print('MR DCE')
            MR_DCE, MR_DCE_spacing, MR_DCE_origin, MR_DCE_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_N4.nii*'.format(ID)))[0])
            if MR_DCE.max() < 0:
                MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                            percentile_max=99.5)
            else:
                MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                            percentile_max=99.5)
            MR_DCE_GTV, MR_DCE_GTV_spacing, MR_DCE_GTV_origin, MR_DCE_GTV_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
            MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
            MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
            MR_DCE_GTVnd1, MR_DCE_GTVnd1_spacing, MR_DCE_GTVnd1_origin, MR_DCE_GTVnd1_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
            MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
            MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MR_DCE.shape != MR_DCE_GTV.shape:
                MR_DCE_GTV = transform.resize(MR_DCE_GTV, output_shape=MR_DCE.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DCE_GTV.dtype) # 直接转换数据类型
            if MR_DCE.shape != MR_DCE_GTVnd1.shape:
                MR_DCE_GTVnd1 = transform.resize(MR_DCE_GTVnd1, output_shape=MR_DCE.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DCE_GTVnd1.dtype) # 直接转换数据类型

            # 合并GTV和GTVnd1，并提取不同gtv_type的mask图像
            MR_DCE_GTV_And_GTVnd1, MR_DCE_img_GTV_And_GTVnd1 = \
                        extract_3D_GTVAndGTVnd1(MR_DCE, MR_DCE_GTV, MR_DCE_GTVnd1)

            img_MR_DCE = MR_DCE[np.newaxis, ...]
            img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MR_DCE = np.concatenate((MR_DCE_GTV_And_GTVnd1[np.newaxis, ...], MR_DCE_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MR_DCE = MR_DCE_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DCE = transform.resize(mask_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MR_DCE = MR_DCE_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DCE = transform.resize(mask_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                                preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MR_DCE.shape, mask_MR_DCE.shape)


            # MR DWI
            # print('MR DWI')
            MR_DWI, MR_DWI_spacing, MR_DWI_origin, MR_DWI_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI.nii*'.format(ID)))[0])
            if MR_DWI.max() < 0:
                MR_DWI = masked_Zscore_norm(MR_DWI, mask_min=np.min(MR_DWI), mask_max=np.max(MR_DWI), percentile_min=None,
                                            percentile_max=99.5)
            else:
                MR_DWI = masked_Zscore_norm(MR_DWI, mask_min=0, mask_max=np.max(MR_DWI), percentile_min=None,
                                            percentile_max=99.5)
            MR_DWI_GTV, MR_DWI_GTV_spacing, MR_DWI_GTV_origin, MR_DWI_GTV_direction  = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI_*GTV.nii*'.format(ID)))[0])
            MR_DWI_GTV[MR_DWI_GTV <= 0.5] = 0
            MR_DWI_GTV[MR_DWI_GTV > 0.5] = 1
            MR_DWI_GTVnd1, MR_DWI_GTVnd1_spacing, MR_DWI_GTVnd1_origin, MR_DWI_GTVnd1_direction = NiiDataRead(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, '{}_DWI_*GTV*nd1.nii*'.format(ID)))[0])
            MR_DWI_GTVnd1[MR_DWI_GTVnd1 <= 0.5] = 0
            MR_DWI_GTVnd1[MR_DWI_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MR_DWI.shape != MR_DWI_GTV.shape:
                MR_DWI_GTV = transform.resize(MR_DWI_GTV, output_shape=MR_DWI.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DWI_GTV.dtype) # 直接转换数据类型
            if MR_DWI.shape != MR_DWI_GTVnd1.shape:
                MR_DWI_GTVnd1 = transform.resize(MR_DWI_GTVnd1, output_shape=MR_DWI.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MR_DWI_GTVnd1.dtype) # 直接转换数据类型
            # 合并GTV和GTVnd1，并提取不同gtv_type的mask图像
            MR_DWI_GTV_And_GTVnd1, MR_DWI_img_GTV_And_GTVnd1 = \
                        extract_3D_GTVAndGTVnd1(MR_DWI, MR_DWI_GTV, MR_DWI_GTVnd1)
            
            img_MR_DWI = MR_DWI[np.newaxis, ...]
            img_MR_DWI = transform.resize(img_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MR_DWI = np.concatenate((MR_DWI_GTV_And_GTVnd1[np.newaxis, ...], MR_DWI_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MR_DWI = transform.resize(mask_MR_DWI, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MR_DWI = MR_DWI_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DWI = transform.resize(mask_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MR_DWI = MR_DWI_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MR_DWI = transform.resize(mask_MR_DWI, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MR_DWI.shape, mask_MR_DWI.shape)


        # MG CC
        if 'MG' in self.modality:
            # print('MG CC')
            MG_CC, MG_CC_img_spacing, MG_CC_img_origin, MG_CC_img_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
            if MG_CC.max() <= 4095:
                MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                            percentile_max=99.5)
            else:
                MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                        percentile_max=99.5)
            MG_CC_GTV, MG_CC_GTV_spacing, MG_CC_GTV_origin, MG_CC_GTV_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
            MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
            MG_CC_GTV[MG_CC_GTV > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MG_CC.shape != MG_CC_GTV.shape:
                MG_CC_GTV = transform.resize(MG_CC_GTV, output_shape=MG_CC.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_CC_GTV.dtype) # 直接转换数据类型

            MG_CC_GTV, MG_CC_img_GTV = extract_2D_MG_CC_GTV(MG_CC, MG_CC_GTV)

            img_MG_CC = MG_CC[np.newaxis, ...]
            img_MG_CC = transform.resize(img_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MG_CC = np.concatenate((MG_CC_GTV[np.newaxis, ...], MG_CC_img_GTV[np.newaxis, ...]), axis=0)
                mask_MG_CC = transform.resize(mask_MG_CC, (2, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
                mask_MG_CC = transform.resize(mask_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MG_CC = MG_CC_img_GTV[np.newaxis, ...]
                mask_MG_CC = transform.resize(mask_MG_CC, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MG_CC.shape, mask_MG_CC.shape)


            # MG MLO
            # print('MG MLO')
            MG_MLO, MG_MLO_img_spacing, MG_MLO_img_origin, MG_MLO_img_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
            if MG_MLO.max() <= 4095:
                MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                        percentile_max=99.5)
            else:
                MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                        percentile_max=99.5)
            MG_MLO_GTV, MG_MLO_GTV_spacing, MG_MLO_GTV_origin, MG_MLO_GTV_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
            MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
            MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
            MG_MLO_GTVnd1,  MG_MLO_GTVnd1_spacing, MG_MLO_GTVnd1_origin, MG_MLO_GTVnd1_direction = NiiDataRead_2D(glob.glob(
                os.path.join(self.data_path, 'MR+MG', ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
            MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
            MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1

            # 检查数据尺寸，如果不匹配，则resize
            if MG_MLO.shape != MG_MLO_GTV.shape:
                MG_MLO_GTV = transform.resize(MG_MLO_GTV, output_shape=MG_MLO.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_MLO_GTV.dtype) # 直接转换数据类型
            if MG_MLO.shape != MG_MLO_GTVnd1.shape:
                MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, output_shape=MG_MLO.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(MG_MLO_GTVnd1.dtype) # 直接转换数据类型

            MG_MLO_GTV_And_GTVnd1, MG_MLO_img_GTV_And_GTVnd1 = \
                        extract_2D_MG_MLO_GTVAndGTVnd1(MG_MLO, MG_MLO_GTV, MG_MLO_GTVnd1)

            img_MG_MLO = MG_MLO[np.newaxis, ...]
            img_MG_MLO = transform.resize(img_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_MG_MLO = np.concatenate((MG_MLO_GTV_And_GTVnd1[np.newaxis, ...], MG_MLO_img_GTV_And_GTVnd1[np.newaxis, ...]), axis=0)
                mask_MG_MLO = transform.resize(mask_MG_MLO, (2, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_MG_MLO = MG_MLO_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MG_MLO = transform.resize(mask_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_MG_MLO = MG_MLO_img_GTV_And_GTVnd1[np.newaxis, ...]
                mask_MG_MLO = transform.resize(mask_MG_MLO, (1, 640, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_MG_MLO.shape, mask_MG_MLO.shape)


        # US H
        if 'US' in self.modality:
            # print('US H')
            US_H, US_H_spacing, US_H_origin, US_H_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_H.nii*'.format(ID)))[0])
            US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
            US_H_GTV, US_H_spacing, US_H_origin, US_H_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
            US_H_GTV[US_H_GTV <= 0.5] = 0
            US_H_GTV[US_H_GTV > 0.5] = 1
            
            # 检查数据尺寸，如果不匹配，则resize
            if US_H.shape != US_H_GTV.shape:
                US_H_GTV = transform.resize(US_H_GTV, output_shape=US_H.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(US_H_GTV.dtype) # 直接转换数据类型

            US_H_GTV, US_H_img_GTV = extract_2D_US_GTV(US_H, US_H_GTV)

            img_US_H = US_H[np.newaxis, ...]
            img_US_H = transform.resize(img_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            # all, origin, img
            if self.gtv_type == 'all':
                mask_US_H = np.concatenate((US_H_GTV[np.newaxis, ...], US_H_img_GTV[np.newaxis, ...]), axis=0)
                mask_US_H = transform.resize(mask_US_H, (2, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_US_H = US_H_GTV[np.newaxis, ...]
                mask_US_H = transform.resize(mask_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_US_H = US_H_img_GTV[np.newaxis, ...]
                mask_US_H = transform.resize(mask_US_H, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_US_H.shape, mask_US_H.shape)

            # US V
            # print('US V')
            US_V, US_V_spacing, US_V_origin, US_V_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_V.nii*'.format(ID)))[0])
            US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
            US_V_GTV, US_V_spacing, US_V_origin, US_V_direction = NiiDataRead_2D(
                glob.glob(os.path.join(self.data_path, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
            US_V_GTV[US_V_GTV <= 0.5] = 0
            US_V_GTV[US_V_GTV > 0.5] = 1
            
            # 检查数据尺寸，如果不匹配，则resize
            if US_V.shape != US_V_GTV.shape:
                US_V_GTV = transform.resize(US_V_GTV, output_shape=US_V.shape, order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False).astype(US_V_GTV.dtype) # 直接转换数据类型

            US_V_GTV, US_V_img_GTV = extract_2D_US_GTV(US_V, US_V_GTV)

            img_US_V = US_V[np.newaxis, ...]
            img_US_V = transform.resize(img_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                        preserve_range=True, anti_aliasing=False)
            
            # all, origin, img
            if self.gtv_type == 'all':
                mask_US_V = np.concatenate((US_V_GTV[np.newaxis, ...], US_V_img_GTV[np.newaxis, ...]), axis=0)
                mask_US_V = transform.resize(mask_US_V, (2, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'origin':
                mask_US_V = US_V_GTV[np.newaxis, ...]
                mask_US_V = transform.resize(mask_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            elif self.gtv_type == 'img':
                mask_US_V = US_V_img_GTV[np.newaxis, ...]
                mask_US_V = transform.resize(mask_US_V, (1, 384, 512), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
            else:
                raise ValueError('gtv_type must be all, origin, or img')
            # print(img_US_V.shape, mask_US_V.shape)


        if self.augment:
            if 'MR' in self.modality:
                augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
                img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)
                augmented_MR_DWI = self.transforms_3D({'img': img_MR_DWI, 'mask': mask_MR_DWI})
                img_MR_DWI = np.concatenate((augmented_MR_DWI['img'], augmented_MR_DWI['mask']), axis=0)

            if 'MG' in self.modality:
                augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
                img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)
                augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
                img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            if 'US' in self.modality:
                augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
                img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)
                augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
                img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)

        else:
            if 'MR' in self.modality:
                img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
                img_MR_DWI = np.concatenate((img_MR_DWI, mask_MR_DWI), axis=0)
            if 'MG' in self.modality:
                img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
                img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            if 'US' in self.modality:
                img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
                img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)

        if 'MR' in self.modality:
            img_MR_DCE = torch.from_numpy(img_MR_DCE)
            img_MR_DWI = torch.from_numpy(img_MR_DWI)
        if 'MG' in self.modality:
            img_MG_CC = torch.from_numpy(img_MG_CC)
            img_MG_MLO = torch.from_numpy(img_MG_MLO)
        if 'US' in self.modality:
            img_US_H = torch.from_numpy(img_US_H)
            img_US_V = torch.from_numpy(img_US_V)

        # LVI_Label 转换为0/1
        if LVI_Label == 'Visible':
            LVI_Label = torch.tensor(1)
        else:
            LVI_Label = torch.tensor(0)
        # SLN_Label 转换为0/1
        if SLN_Label == 'Positive':
            SLN_Label = torch.tensor(1)
        else:
            SLN_Label = torch.tensor(0)
                
        # return ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label,\
        #         MR_DCE_spacing, MR_DCE_origin, MR_DCE_direction,\
        #         MR_DCE_GTV_spacing, MR_DCE_GTV_origin, MR_DCE_GTV_direction,\
        #         MR_DCE_GTVnd1_spacing, MR_DCE_GTVnd1_origin, MR_DCE_GTVnd1_direction,\
        #         MR_DWI_spacing, MR_DWI_origin, MR_DWI_direction,\
        #         MR_DWI_GTV_spacing, MR_DWI_GTV_origin, MR_DWI_GTV_direction,\
        #         MR_DWI_GTVnd1_spacing, MR_DWI_GTVnd1_origin, MR_DWI_GTVnd1_direction,\
        #         MG_CC_img_spacing, MG_CC_img_origin, MG_CC_img_direction,\
        #         MG_CC_GTV_spacing, MG_CC_GTV_origin, MG_CC_GTV_direction,\
        #         MG_MLO_img_spacing, MG_MLO_img_origin, MG_MLO_img_direction,\
        #         MG_MLO_GTV_spacing, MG_MLO_GTV_origin, MG_MLO_GTV_direction,\
        #         MG_MLO_GTVnd1_spacing, MG_MLO_GTVnd1_origin, MG_MLO_GTVnd1_direction,\
        #         US_H_spacing, US_H_origin, US_H_direction,\
        #         US_V_spacing, US_V_origin, US_V_direction
    
        # return ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label

        # 收集数据
        data_list = []
        if 'MR' in self.modality:
            data_list.append(img_MR_DCE)
            data_list.append(img_MR_DWI)
        if 'MG' in self.modality:
            data_list.append(img_MG_CC)
            data_list.append(img_MG_MLO)
        if 'US' in self.modality:
            data_list.append(img_US_H)
            data_list.append(img_US_V)
        if 'Clinical' in self.modality:
            data_list.append(clinical_features)
        if 'LVI' in self.modality:
            data_list.append(lvi_features)

        return ID, data_list, LVI_Label, SLN_Label

    def __len__(self):
        return self.len



def parse_args(data_path, split_path, metadata_df_path, lvi_preded_df_path, gtv_type, modality, lvi_type):
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('--data_path', type=str, default=data_path, help='data path')
    parser.add_argument('--split_path', type=str, default=split_path, help='split path')
    parser.add_argument('--metadata_df_path', type=str, default=metadata_df_path, help='metadata df path')
    parser.add_argument('--lvi_preded_df_path', type=str, default=lvi_preded_df_path, help='lvi preded df path')
    parser.add_argument('--gtv_type', type=str, default=gtv_type, help='gtv type')
    parser.add_argument('--modality', type=str, default=modality, help='modality')
    parser.add_argument('--lvi_type', type=str, default=lvi_type, help='lvi type')

    return parser.parse_args()


if __name__ == '__main__':
    # 图像尺寸
    # MR (3, 144, 320, 320)
    # MG (3, 640, 512)
    # US (3, 384, 512)

    os.environ["CUDA_VISIBLE_DEVICES"] = str('0') # 规定程序可以看见那些GPU
    data_path = r'/root/epfs/share_center1'
    split_path = r'/root/epfs/all_centers_statistics_info5/LVI_SLN_split_center1(611)_8(189).pkl'
    metadata_df_path = r'/root/epfs/all_centers_statistics_info5/center1_8_Clinicopathological_info_611_189.csv'
    lvi_preded_df_path = r'/root/epfs/all_centers_statistics_info5/LVI_pred_center1(611)_8(189).csv'

    # tmp_df = pd.read_csv(lvi_preded_df_path, dtype={'ID': str})
    # tmp_df['ID'] = tmp_df['ID'].str.lstrip('0')
    # Extract_lvi_features(tmp_df, '377555')

    gtv_type = 'img' # all, origin, img
    modality_list = ['MRMGUS_Clinical_LVI','MRMGUS_Clinical', 'MRMGUS', 'MRMG', 'MRUS', 'MGUS', 'MR', 'MG', 'US','Clinical_LVI','Clinical','LVI']
    modality = modality_list[-3] # Clinical 31个特征
    lvi_type = 'Origin' # Pred, Origin
    args = parse_args(data_path, split_path, metadata_df_path, lvi_preded_df_path, gtv_type,  modality, lvi_type)

    # augmentGPU = AugmentGPU(modality)
    # dataset = MultiModal_Dataset(args,data_set='test', augment=False) # 关闭CPU数据增强
    dataset = MultiModal_WithLVI_Dataset(args,data_set='test', augment=False)
    # tmp = dataset.__getitem__(118)

    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=5, pin_memory=True)
    # 是数据本身的问题，操作前，resize使其一致即可，差别不大 3200947 MG
    
    # for i, (ID, img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features,LVI_Label, SLN_Label,\
    #             MR_DCE_spacing, MR_DCE_origin, MR_DCE_direction,\
    #             MR_DCE_GTV_spacing, MR_DCE_GTV_origin, MR_DCE_GTV_direction,\
    #             MR_DCE_GTVnd1_spacing, MR_DCE_GTVnd1_origin, MR_DCE_GTVnd1_direction,\
    #             MR_DWI_spacing, MR_DWI_origin, MR_DWI_direction,\
    #             MR_DWI_GTV_spacing, MR_DWI_GTV_origin, MR_DWI_GTV_direction,\
    #             MR_DWI_GTVnd1_spacing, MR_DWI_GTVnd1_origin, MR_DWI_GTVnd1_direction,\
    #             MG_CC_img_spacing, MG_CC_img_origin, MG_CC_img_direction,\
    #             MG_CC_GTV_spacing, MG_CC_GTV_origin, MG_CC_GTV_direction,\
    #             MG_MLO_img_spacing, MG_MLO_img_origin, MG_MLO_img_direction,\
    #             MG_MLO_GTV_spacing, MG_MLO_GTV_origin, MG_MLO_GTV_direction,\
    #             MG_MLO_GTVnd1_spacing, MG_MLO_GTVnd1_origin, MG_MLO_GTVnd1_direction,\
    #             US_H_spacing, US_H_origin, US_H_direction,\
    #             US_V_spacing, US_V_origin, US_V_direction) in enumerate(dataloader):

    start = time.perf_counter()  # 记录开始时刻（高精度）
    # for i, (ID, img_MR_DCE, img_MR_DWI, LVI_Label, SLN_Label) in enumerate(dataloader):
    for i, (ID, data_list, LVI_Label, SLN_Label) in enumerate(dataloader):

        # data_list = augmentGPU.augment(data_list)  # 在主进程使用GPU，进行数据增强

        # end = time.perf_counter()    # 记录结束时刻
        # elapsed = end - start
        # print(f"数据加载时间: {elapsed:.6f} 秒")
        # batch_size=5 num_workers=5 cpu增强65秒左右, 未增强20秒左右
        # batch_size=5 num_workers=10 cpu增强65秒左右, 未增强20秒左右
        # batch_size=20 num_workers=5 cpu增强200秒左右, 未增强75秒左右，去掉弹性100秒左右
        
        print(i,ID)
        print(data_list[0].shape)
        print(data_list[1].shape)
        # print(data_list[2].shape)
        # print(data_list[3].shape)
        # print(data_list[4].shape)
        # print(data_list[5].shape)
        # print(data_list[6].shape)
        # print(LVI_Label)
        # print(SLN_Label)

    end = time.perf_counter()    
    elapsed = end - start
    print(f"数据加载时间: {elapsed:.6f} 秒")
    # bs5 num10 MR 200秒 
    # bs8 num10 MR 230秒

        # img_MR_DCE = img_MR_DCE.numpy()
        # MR_DCE_spacing = MR_DCE_spacing.numpy()
        # MR_DCE_origin = [x.numpy()[0] for x in MR_DCE_origin]
        # MR_DCE_direction = [x.numpy()[0] for x in MR_DCE_direction]
        # MR_DCE_GTV_spacing = MR_DCE_GTV_spacing.numpy()
        # MR_DCE_GTV_origin = [x.numpy()[0] for x in MR_DCE_GTV_origin]
        # MR_DCE_GTV_direction = [x.numpy()[0] for x in MR_DCE_GTV_direction]

        # img_MR_DWI = img_MR_DWI.numpy()
        # MR_DWI_spacing = MR_DWI_spacing.numpy()
        # MR_DWI_origin = [x.numpy()[0] for x in MR_DWI_origin]
        # MR_DWI_direction = [x.numpy()[0] for x in MR_DWI_direction]
        # MR_DWI_GTV_spacing = MR_DWI_GTV_spacing.numpy()
        # MR_DWI_GTV_origin = [x.numpy()[0] for x in MR_DWI_GTV_origin]
        # MR_DWI_GTV_direction = [x.numpy()[0] for x in MR_DWI_GTV_direction]

        # img_MG_CC = img_MG_CC.numpy()
        # MG_CC_img_spacing = MG_CC_img_spacing.numpy()
        # MG_CC_img_origin = [x.numpy()[0] for x in MG_CC_img_origin]
        # MG_CC_img_direction = [x.numpy()[0] for x in MG_CC_img_direction]
        # MG_CC_GTV_spacing = MG_CC_GTV_spacing.numpy()
        # MG_CC_GTV_origin = [x.numpy()[0] for x in MG_CC_GTV_origin]
        # MG_CC_GTV_direction = [x.numpy()[0] for x in MG_CC_GTV_direction]

        # img_MG_MLO = img_MG_MLO.numpy()
        # MG_MLO_img_spacing = MG_MLO_img_spacing.numpy()
        # MG_MLO_img_origin = [x.numpy()[0] for x in MG_MLO_img_origin]
        # MG_MLO_img_direction = [x.numpy()[0] for x in MG_MLO_img_direction]
        # MG_MLO_GTV_spacing = MG_MLO_GTV_spacing.numpy()
        # MG_MLO_GTV_origin = [x.numpy()[0] for x in MG_MLO_GTV_origin]
        # MG_MLO_GTV_direction = [x.numpy()[0] for x in MG_MLO_GTV_direction]

        # img_US_H = img_US_H.numpy()
        # US_H_spacing = US_H_spacing.numpy()
        # US_H_origin = [x.numpy()[0] for x in US_H_origin]
        # US_H_direction = [x.numpy()[0] for x in US_H_direction]
        
        # img_US_V = img_US_V.numpy()
        # US_V_spacing = US_V_spacing.numpy()
        # US_V_origin = [x.numpy()[0] for x in US_V_origin]
        # US_V_direction = [x.numpy()[0] for x in US_V_direction]

        
