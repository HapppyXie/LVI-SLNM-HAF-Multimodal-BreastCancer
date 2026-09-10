from audioop import bias
from time import daylight
import torch
from torch import nn
import torch.nn.functional as F

from .ResNet_3D_base import * # 作为Networks整体时设置
from .ResNet_2D_base import *
from .AttentionFusionNetwork import *

import ssl
# 绕过SSL证书验证的Python配置
# 将默认的SSL上下文创建函数替换为不验证证书的版本
# ssl._create_default_https_context = ssl._create_unverified_context



class MF_ResNet_FC(nn.Module):
    def __init__(self, modality='MRMGUS_Clinical', input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                input_size_US=(384,512),clinical_features=34, num_classes=2, dropout=0.1):
        super(MF_ResNet_FC, self).__init__()
        self.modality = modality
        extractor_count = 0
        extract_feature_num = 512  # resNet18/34 512, resNet50 1024

        if 'MR' in self.modality: # 根据模态创建并特征提取器，并自动完成初始化
            # self.extractor_MR_DCE = resnet18_3D(input_channels=input_channels,input_size=input_size_MR)
            # self.extractor_MR_DWI = resnet18_3D(input_channels=input_channels,input_size=input_size_MR)

            self.extractor_MR_DCE = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)
            self.extractor_MR_DWI = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)
            extractor_count += 2
        if 'MG' in self.modality:
            # self.extractor_MG_CC = resnet18_2D(input_channels=input_channels,input_size=input_size_MG)
            # self.extractor_MG_MLO = resnet18_2D(input_channels=input_channels,input_size=input_size_MG)

            self.extractor_MG_CC = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)
            self.extractor_MG_MLO = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)
            extractor_count += 2
        if 'US' in self.modality:
            # self.extractor_US_H = resnet18_2D(input_channels=input_channels,input_size=input_size_US)
            # self.extractor_US_V = resnet18_2D(input_channels=input_channels,input_size=input_size_US)

            self.extractor_US_H = resnet34_2D(input_channels=input_channels,input_size=input_size_US)
            self.extractor_US_V = resnet34_2D(input_channels=input_channels,input_size=input_size_US)
            extractor_count += 2
        if 'Clinical' in self.modality:
            self.extractor_Clinical = nn.Sequential(
                                    nn.Linear(clinical_features, 256, bias=False), 
                                    nn.BatchNorm1d(256),  # 加上归一化防止数值不稳定，输出logits过大影响loss计算
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(dropout),
                                    nn.Linear(256, 512, bias=False)
                                ) # 标准MLP
            extractor_count += 1
        
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(extract_feature_num*extractor_count),  # 单模态单模式内 加上归一化防止数值不稳定，输出logits过大影响loss计算，防止极端值
            nn.LayerNorm(extract_feature_num*extractor_count),  # 多模态，多模式融合时更稳定。且收到bs影响小
            nn.Dropout(dropout),
            nn.Linear(extract_feature_num*extractor_count, num_classes)
        )


        self._initialize_weights() #初始化权重 适合resNet类型
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层常用 Kaiming 初始化 (配合 ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                # 批归一化层的标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
    def forward(self, data_list):
        if 'MRMGUS_Clinical' == self.modality:
            img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical = data_list # 赋值>=2个，才可解包
            img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
            img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
            img_MG_CC = self.extractor_MG_CC(img_MG_CC)
            img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
            img_US_H = self.extractor_US_H(img_US_H)
            img_US_V = self.extractor_US_V(img_US_V)
            clinical = self.extractor_Clinical(clinical)
            x = torch.cat((img_MR_DCE,img_MR_DWI,img_MG_CC, img_MG_MLO,img_US_H, img_US_V,clinical), dim=1)
            x = self.fc(x)
            return img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,x

        if 'MRMGUS' == self.modality:
            img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V = data_list
            img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
            img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
            img_MG_CC = self.extractor_MG_CC(img_MG_CC)
            img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
            img_US_H = self.extractor_US_H(img_US_H)
            img_US_V = self.extractor_US_V(img_US_V)
            x = torch.cat((img_MR_DCE,img_MR_DWI,img_MG_CC, img_MG_MLO,img_US_H, img_US_V), dim=1)
            x = self.fc(x)
            return img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,x

        if 'MRMG' == self.modality:
            img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO = data_list
            img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
            img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
            img_MG_CC = self.extractor_MG_CC(img_MG_CC)
            img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
            x = torch.cat((img_MR_DCE,img_MR_DWI,img_MG_CC, img_MG_MLO), dim=1)
            x = self.fc(x)
            return img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,x

        if 'MRUS' == self.modality:
            img_MR_DCE, img_MR_DWI, img_US_H, img_US_V = data_list
            img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
            img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
            img_US_H = self.extractor_US_H(img_US_H)
            img_US_V = self.extractor_US_V(img_US_V)
            x = torch.cat((img_MR_DCE,img_MR_DWI,img_US_H, img_US_V), dim=1)
            x = self.fc(x)
            return img_MR_DCE,img_MR_DWI,img_US_H,img_US_V,x

        if 'MGUS' == self.modality:
            img_MG_CC, img_MG_MLO, img_US_H, img_US_V = data_list
            img_MG_CC = self.extractor_MG_CC(img_MG_CC)
            img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
            img_US_H = self.extractor_US_H(img_US_H)
            img_US_V = self.extractor_US_V(img_US_V)
            x = torch.cat((img_MG_CC, img_MG_MLO,img_US_H, img_US_V), dim=1)
            x = self.fc(x)
            return img_MG_CC,img_MG_MLO,img_US_H,img_US_V,x

        if 'MR' == self.modality:
            img_MR_DCE, img_MR_DWI = data_list
            img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
            img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
            x = torch.cat((img_MR_DCE,img_MR_DWI), dim=1)
            x = self.fc(x)
            return img_MR_DCE,img_MR_DWI,x

        if 'MG' == self.modality:
            img_MG_CC, img_MG_MLO = data_list
            img_MG_CC = self.extractor_MG_CC(img_MG_CC)
            img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
            x = torch.cat((img_MG_CC, img_MG_MLO), dim=1)
            x = self.fc(x)
            return img_MG_CC,img_MG_MLO,x
        
        if 'US' == self.modality:
            img_US_H, img_US_V = data_list
            img_US_H = self.extractor_US_H(img_US_H)
            img_US_V = self.extractor_US_V(img_US_V)
            x = torch.cat((img_US_H, img_US_V), dim=1)
            x = self.fc(x)
            return img_US_H,img_US_V,x

        if 'Clinical' == self.modality:
            clinical = data_list[0]
            clinical = self.extractor_Clinical(clinical)
            x = self.fc(clinical)
            return clinical,x


class MF_ResNet_AttnFusion(nn.Module):
    """
    阶段一模型
    三模态和临床 预测 LVI 和 SLN
    """
    def __init__(self, modality='MRMGUS_Clinical', input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                input_size_US=(384,512),clinical_features=34, num_classes=2, dropout=0.1):
        super(MF_ResNet_AttnFusion, self).__init__()
        self.modality = modality
        extract_feature_num = 512  # resNet18/34 512, resNet50 1024

        self.extractor_MR_DCE = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)
        self.extractor_MR_DWI = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)

        self.extractor_MG_CC = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)
        self.extractor_MG_MLO = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)

        self.extractor_US_H = resnet34_2D(input_channels=input_channels,input_size=input_size_US)
        self.extractor_US_V = resnet34_2D(input_channels=input_channels,input_size=input_size_US)

        self.extractor_Clinical = nn.Sequential(
                                nn.Linear(clinical_features, 256, bias=False), 
                                nn.BatchNorm1d(256),  # 加上归一化防止数值不稳定，输出logits过大影响loss计算
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(256, 512, bias=False)) # 标准MLP

        # ---------------------------------------------------------------------------
        # 只有 2 个模式参与交互，depth=1, num_heads=2, dropout=0.1 即可
        self.self_attn_MR = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1) # 自注意力融合单模态
        self.self_attn_MG = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1)
        self.self_attn_US = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1)

        # ---------------------------------------------------------------------------
        # 4 个模态之间的交互关系更复杂, depth=1, num_heads=4, dropout=0.1
        self.cross_attn_MR = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1) # 交叉注意力，跨模态融合
        self.cross_attn_MG = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)
        self.cross_attn_US = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)
        self.cross_attn_Clinical = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)

        # ---------------------------------------------------------------------------
        # 对所有已交互的模态特征做最终整合
        self.self_attn_all = SelfAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1) # 自注意力融合多模态

        self.multiModality_Fusion = nn.Sequential(
                                        nn.LayerNorm(4*extract_feature_num),
                                        nn.Dropout(dropout),
                                        nn.Linear(4*extract_feature_num, num_classes)
                                    )

        # ---------------------------------------------------------------------------
        self._initialize_weights() #初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层常用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                # 批 归一化层 和 层归一化 的标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

       
    def forward(self, data_list):

        img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical = data_list # 赋值>=2个，才可解包

        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE) # (B, 512)
        img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        clinical_features = self.extractor_Clinical(clinical)
        
        # 强制 fp32 / 混合精度，会出现nan值崩溃
        with torch.amp.autocast('cuda', enabled=False):
            img_MR_DCE, img_MR_DWI = img_MR_DCE.float(), img_MR_DWI.float()
            img_MG_CC, img_MG_MLO = img_MG_CC.float(), img_MG_MLO.float()
            img_US_H, img_US_V = img_US_H.float(), img_US_V.float()
            clinical_features = clinical_features.float()

            # 单模态内融合
            MR_features = torch.stack((img_MR_DCE, img_MR_DWI),dim=1) # (B, 2, 512) 拼接
            MG_features = torch.stack((img_MG_CC, img_MG_MLO),dim=1)
            US_features = torch.stack((img_US_H, img_US_V),dim=1)

            MR_features, self_attn_weight_MR_list = self.self_attn_MR(MR_features) # (B, 1, 512) 融合
            MG_features, self_attn_weight_MG_list = self.self_attn_MG(MG_features)
            US_features, self_attn_weight_US_list = self.self_attn_US(US_features)
            clinical_features = clinical_features.unsqueeze(1) # (B, 1, 512) 增加模态通道维度

            # 跨模态融合
            # 对 MR_features, MG_features, US_features, clinical_features 进行组合
            context_MGUSClinical = torch.cat((MG_features, US_features, clinical_features), dim=1) # (B, 3, 512) 拼接
            context_MRUSClinical = torch.cat((MR_features, US_features, clinical_features), dim=1)
            context_MRMGClinical = torch.cat((MR_features, MG_features, clinical_features), dim=1)
            context_MRMGUS = torch.cat((MR_features, MG_features, US_features), dim=1)
                
            x_MR, cross_attn_weight_MR_ToMGUSClinical_list = self.cross_attn_MR(MR_features, context_MGUSClinical) # (B, 1, 512) 融合
            x_MG, cross_attn_weight_MG_ToMRUSClinical_list = self.cross_attn_MG(MG_features, context_MRUSClinical)
            x_US, cross_attn_weight_US_ToMRMGClinical_list = self.cross_attn_US(US_features, context_MRMGClinical)
            x_Clinical, cross_attn_weight_Clinical_ToMRMGUS_list = self.cross_attn_Clinical(clinical_features, context_MRMGUS)

            # 多模态融合
            x = torch.cat((x_MR, x_MG, x_US, x_Clinical), dim=1) # (B, 4, 512) # 拼接
            x, self_attn_weight_all_list = self.self_attn_all(x) # x: (B, 4, 512)self_attn_weight_all: (B, head, 4, 4)

            x = x.view(x.size(0), -1)          # (B, 4*512)

        # fp32 区域结束，回到 AMP 环境
        # x = self.norm(x)
        # x = self.multiModality_linear(x)   # (B, 2)
        x = self.multiModality_Fusion(x)   # (B, 2)
        
        # return img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical_features
        return self_attn_weight_MR_list, self_attn_weight_MG_list, self_attn_weight_US_list,\
                cross_attn_weight_MR_ToMGUSClinical_list, cross_attn_weight_MG_ToMRUSClinical_list, \
                cross_attn_weight_US_ToMRMGClinical_list, cross_attn_weight_Clinical_ToMRMGUS_list, \
                self_attn_weight_all_list, x


class MF_ResNet_AttnFusion_LVIGuide(nn.Module):
    """
    阶段二模型
    三模态和临床，以及预测的LVI数据，预测 SLN
    """
    def __init__(self, modality='MRMGUS_Clinical_LVI', input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                input_size_US=(384,512),clinical_features=34, num_classes=2, dropout=0.1):
        super(MF_ResNet_AttnFusion_LVIGuide, self).__init__()
        self.modality = modality
        extract_feature_num = 512  # resNet18/34 512, resNet50 1024

        self.extractor_MR_DCE = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)
        self.extractor_MR_DWI = resnet34_3D(input_channels=input_channels,input_size=input_size_MR)

        self.extractor_MG_CC = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)
        self.extractor_MG_MLO = resnet34_2D(input_channels=input_channels,input_size=input_size_MG)

        self.extractor_US_H = resnet34_2D(input_channels=input_channels,input_size=input_size_US)
        self.extractor_US_V = resnet34_2D(input_channels=input_channels,input_size=input_size_US)

        self.extractor_Clinical = nn.Sequential(
                                nn.Linear(clinical_features, 256, bias=False), 
                                nn.BatchNorm1d(256),  # 加上归一化防止数值不稳定，输出logits过大影响loss计算
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(256, 512, bias=False)) # 标准MLP

        # ---------------------------------------------------------------------------
        # 只有 2 个模式参与交互，depth=1, num_heads=2, dropout=0.1 即可
        self.self_attn_MR = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1) # 自注意力融合单模态
        self.self_attn_MG = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1)
        self.self_attn_US = SingleModalityFusion(2, extract_feature_num, depth=1, num_heads=2, dropout=0.1)

        # ---------------------------------------------------------------------------
        # 4 个模态之间的交互关系更复杂, depth=1, num_heads=4, dropout=0.1
        self.cross_attn_MR = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1) # 交叉注意力，跨模态融合
        self.cross_attn_MG = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)
        self.cross_attn_US = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)
        self.cross_attn_Clinical = CrossAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1)

        # ---------------------------------------------------------------------------
        # 对所有已交互的模态特征做最终整合
        self.self_attn_all = SelfAttentionBlock(extract_feature_num, depth=1, num_heads=4, dropout=0.1) # 自注意力融合多模态

        self.multiModality_Fusion = nn.Sequential(
                                        nn.LayerNorm(4*extract_feature_num),
                                        nn.Dropout(dropout),
                                        nn.Linear(4*extract_feature_num, num_classes)
                                    )
        
        # ---------------------------------------------------------------------------
        # 决策层融合：模型输出的SLN logtis(softmax成概率) + LVI 概率 → 最终 SLN 概率
        self.prob_fusion = nn.Linear(4,num_classes)

        # ---------------------------------------------------------------------------
        self._initialize_weights() #初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层常用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                # 批 归一化层 和 层归一化 的标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, data_list):

        img_MR_DCE, img_MR_DWI, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical, LVI_Pred_Score = data_list # 赋值>=2个，才可解包
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE) # (B, 512)
        img_MR_DWI = self.extractor_MR_DWI(img_MR_DWI)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        clinical_features = self.extractor_Clinical(clinical)
        
        # 强制 FP32 / 混合精度，会出现nan值崩溃；
        # 可以开启FP16，如果不要求跑完100个epoch，可以加速训练
        with torch.amp.autocast('cuda', enabled=False):
            img_MR_DCE, img_MR_DWI = img_MR_DCE.float(), img_MR_DWI.float()
            img_MG_CC, img_MG_MLO = img_MG_CC.float(), img_MG_MLO.float()
            img_US_H, img_US_V = img_US_H.float(), img_US_V.float()
            clinical_features = clinical_features.float()
            LVI_Pred_Score = LVI_Pred_Score.float()

            # 单模态内融合
            MR_features = torch.stack((img_MR_DCE, img_MR_DWI),dim=1) # (B, 2, 512) 拼接
            MG_features = torch.stack((img_MG_CC, img_MG_MLO),dim=1)
            US_features = torch.stack((img_US_H, img_US_V),dim=1)

            MR_features, self_attn_weight_MR_list = self.self_attn_MR(MR_features) # (B, 1, 512) 融合
            MG_features, self_attn_weight_MG_list = self.self_attn_MG(MG_features)
            US_features, self_attn_weight_US_list = self.self_attn_US(US_features)
            clinical_features = clinical_features.unsqueeze(1) # (B, 1, 512) 增加模态通道维度

            # 跨模态融合
            # 对 MR_features, MG_features, US_features, clinical_features 进行组合
            context_MGUSClinical = torch.cat((MG_features, US_features, clinical_features), dim=1) # (B, 3, 512) 拼接
            context_MRUSClinical = torch.cat((MR_features, US_features, clinical_features), dim=1)
            context_MRMGClinical = torch.cat((MR_features, MG_features, clinical_features), dim=1)
            context_MRMGUS = torch.cat((MR_features, MG_features, US_features), dim=1)
                    
            x_MR, cross_attn_weight_MR_ToMGUSClinical_list = self.cross_attn_MR(MR_features, context_MGUSClinical) # (B, 1, 512) 融合
            x_MG, cross_attn_weight_MG_ToMRUSClinical_list = self.cross_attn_MG(MG_features, context_MRUSClinical)
            x_US, cross_attn_weight_US_ToMRMGClinical_list = self.cross_attn_US(US_features, context_MRMGClinical)
            x_Clinical, cross_attn_weight_Clinical_ToMRMGUS_list = self.cross_attn_Clinical(clinical_features, context_MRMGUS)

            # 多模态融合
            x = torch.cat((x_MR, x_MG, x_US, x_Clinical), dim=1) # (B, 4, 512) # 拼接
            x, self_attn_weight_all_list = self.self_attn_all(x) # x: (B, 4, 512)self_attn_weight_all: (B, head, 4, 4)

            x = x.view(x.size(0), -1)          # (B, 4*512)
            x = self.multiModality_Fusion(x)   # (B, 2) sln logits

            # MF_ResNet_AttnFusion_LVIGuide_MRMGUS_Clinical_LVI_bs10_all_img_centers1_8_Date0717_只用概率融合
            # 统一到概率空间融合
            sln_prob = F.softmax(x, dim=-1)
            combined_prob = torch.cat([sln_prob, LVI_Pred_Score], dim=-1) # (B, 4)
            x_prob = self.prob_fusion(combined_prob)   # (4, 2)
                
            # 概率残差连接
            x = x + x_prob
        
        # return img_MR_DCE,img_MR_DWI,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical_features
        return self_attn_weight_MR_list, self_attn_weight_MG_list, self_attn_weight_US_list,\
                cross_attn_weight_MR_ToMGUSClinical_list, cross_attn_weight_MG_ToMRUSClinical_list, \
                cross_attn_weight_US_ToMRMGClinical_list, cross_attn_weight_Clinical_ToMRMGUS_list, \
                self_attn_weight_all_list, x


if __name__ == '__main__':
    # 原则：提取器里小dropout（0.1-0.3），融合层里大dropout（0.5）。

    img_MR_DCE = torch.rand((2, 2, 144, 320, 320))
    img_MR_DWI = torch.rand((2, 2, 144, 320, 320))
    img_MR_DCE_slice = torch.rand((2, 18, 320, 320))
    img_MR_DWI_slice = torch.rand((2, 18, 320, 320))
    input_size_MR = (144, 320, 320)
    img_MG_CC = torch.rand((2, 2, 640, 512))
    img_MG_MLO = torch.rand((2, 2, 640, 512))
    input_size_MG = (640, 512)
    img_US_H = torch.rand((2, 2, 384, 512))
    img_US_V = torch.rand((2, 2, 384, 512))
    input_size_US = (384, 512)
    clinical_features = torch.rand((2,31))
    LVI_Pred_Score = torch.rand((2,2))
    # pathology_features = torch.rand((2,20))

    channels= 2
    num_classes= 2
    clinical_inchannels= 31
    dropout = 0.1
    # modality_list = ['MRMGUS_Clinical_LVI','MRMGUS_Clinical','MRMGUS','MRMG','MRUS','MGUS','MR','MG','US','Clinical']
    modality_list = ['MRMGUS_Clinical_LVI']
    for i in range(len(modality_list)):
        modality = modality_list[i]

        # net = MF_ResNet_FC_Slice(input_channels=18, input_size_MR=(320,320),  num_classes=2, dropout=0.1).cuda()
        # net = MF_ResNet_FC(modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
        #             input_size_US=(384,512),clinical_features=31, num_classes=2, dropout=0.1).cuda()
        # net = MF_ResNet_AttnFusion(modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
        #             input_size_US=(384,512),clinical_features=31, num_classes=2, dropout=0.1).cuda()         
        net = MF_ResNet_AttnFusion_LVIGuide(modality, input_channels=2, input_size_MR=(144,320,320), input_size_MG=(640,512), \
                    input_size_US=(384,512),clinical_features=31, num_classes=2, dropout=0.1).cuda()   



        data_list = []
        if 'MR' in modality:
            data_list.append(img_MR_DCE)
            data_list.append(img_MR_DWI)
            # data_list.append(img_MR_DCE_slice)
            # data_list.append(img_MR_DWI_slice)
        if 'MG' in modality:
            data_list.append(img_MG_CC)
            data_list.append(img_MG_MLO)
        if 'US' in modality:
            data_list.append(img_US_H)
            data_list.append(img_US_V)
        if 'Clinical' in modality:
            data_list.append(clinical_features)
        if 'LVI' in modality:
            data_list.append(LVI_Pred_Score)
        data_list = [data.cuda() for data in data_list]
        net.eval()

        b = net(data_list)[-1]
        print(b)
        print(b.shape)

    