import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
        Standard Multi-Head Self-Attention returning weights.
        dim	        输入特征维度，也是输出特征维度
        num_heads	多头注意力的头数，默认8
        head_dim	每个头的维度 = dim / num_heads
        scale	    缩放因子 = 1/√(head_dim)，用于防止点积过大导致softmax梯度消失
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False) # 一个线性层同时生成 Query、Key、Value，输出维度是 dim*3
        self.proj = nn.Linear(dim, dim) # 输出投影层，将多头拼接后的结果映射回 dim 维
        self.attn_drop = nn.Dropout(dropout) # 对注意力权重矩阵做dropout，防止过拟合

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, D) batch_size, channel, dimension
        returns: (B, C, D), (B, heads, C, C)
        """
        B, C, D = x.shape
        
        # 工程上优化，qkv映射合并 Self-Attention 自注意力可以，自己查自己
        # x ──→ [Linear] ──→ Q
        # x ──→ [Linear] ──→ K    全是同一个 x
        # x ──→ [Linear] ──→ V
        qkv = self.qkv(x) # self.qkv(x)：(B, C, D) → (B, C, 3*D) 

        # 如果是 Cross-Attention 交叉注意力，则不能合并。Q 来自一个序列，K 和 V 来自另一个序列
        # x_dec ──→ [W_Q] ──→ Q     查询者：来自解码器
        # x_enc ──→ [W_K] ──→ K     被查者：来自编码器
        # x_enc ──→ [W_V] ──→ V     被查者：来自编码器

        qkv = qkv.reshape(B, C, 3, self.num_heads, self.head_dim)  # .reshape(...)：拆分为 (B, C, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, C, head_dim)
        q, k, v = qkv.unbind(0) # 得到 q, k, v 矩阵 , 形状 (B, heads, C, head_dim)

        k = k.transpose(-2, -1) # k.shape (B, heads, C, head_dim) 转置后 (B, heads, head_dim, C)
        # @ 为矩阵乘法，计算点积（相似度，相当于查询），也即注意力分数，再缩放
        # q @ k 形状 (B, heads, C, head_dim) × (B, heads, head_dim, C) → (B, heads, C, C)
        attn_score = (q @ k) * self.scale
        attn_weight = attn_score.softmax(dim=-1)  # softmax 转化为概率(注意力池化)，也即注意力权重
        attn_weights_detach = attn_weight.detach() # 断开计算图，返回注意力权重

        attn_weight = self.attn_drop(attn_weight) # (B, heads, C, C)

        out = (attn_weight @ v) # attn_weight @ v 形状 (B, heads, C, head_dim)
        out = out.transpose(1, 2)  # (B, heads, C, head_dim) → (B, C, heads, head_dim)
        out = out.reshape(B, C, D) #  heads * head_dim = D, 将多头拼接回 (B, C, D) 
        out = self.proj(out)  # (B, C, D) 

        return out, attn_weights_detach


class CrossAttention(nn.Module):
    """
        Multi-Head Cross-Attention returning weights.
        dim            输入特征维度，也是输出特征维度
        num_heads      多头注意力的头数，默认8
        head_dim       每个头的维度 = dim / num_heads
        scale          缩放因子 = 1/√(head_dim)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)   # Q 从 query 输入中投影
        self.kv = nn.Linear(dim, dim * 2, bias=False)  # K、V 从 context 输入中投影（可合并）
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        当输入的x和context相同时，自己查询自己，自注意力
        当输入的x和context不同时，x查询context，交叉注意力

        x:       (B, C_q,  D)  查询序列（如 decoder 输出）
        context: (B, C_kv, D)  被查询序列（如 encoder 输出）
        returns: (B, C_q, D), (B, heads, C_q, C_kv)
        """
        B, C_q, D = x.shape
        C_kv = context.shape[1]

        # Q 来自 x
        q = self.q(x)                                         # (B, C_q, D)
        q = q.reshape(B, C_q, self.num_heads, self.head_dim)  # (B, C_q, heads, head_dim)
        q = q.permute(0, 2, 1, 3)                             # (B, heads, C_q, head_dim)

        # K、V 来自 context，可以合并投影
        kv = self.kv(context)                                         # (B, C_kv, 2*D)
        kv = kv.reshape(B, C_kv, 2, self.num_heads, self.head_dim)    # (B, C_kv, 2, heads, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                                # (2, B, heads, C_kv, head_dim)
        k, v = kv.unbind(0)                                           # 各 (B, heads, C_kv, head_dim)

        k = k.transpose(-2, -1)  # (B, heads, head_dim, C_kv)
        attn_score = (q @ k) * self.scale  # (B, heads, C_q, C_kv)
        attn_weight = attn_score.softmax(dim=-1)
        attn_weights_detach = attn_weight.detach()

        attn_weight = self.attn_drop(attn_weight)  # (B, heads, C_q, C_kv)

        out = (attn_weight @ v)                    # (B, heads, C_q, head_dim)
        out = out.transpose(1, 2)                  # (B, C_q, heads, head_dim)
        out = out.reshape(B, C_q, D)               # (B, C_q, D)
        out = self.proj(out)

        return out, attn_weights_detach


class SelfAttentionBlock(nn.Module):
    """模态内 Self-Attention + FFN
    Attention 聚合 → 非线性变换 → Attention 聚合 → 非线性变换 → ...→ 表达能力理论无限
    可堆叠多层
    """
    def __init__(self, dim, depth=3, num_heads=8, dropout=0.1):
        super().__init__()

        # self.norm = nn.LayerNorm(dim)
        # self.attn = SelfAttention(dim, num_heads, dropout)
        # self.ffn_norm = nn.LayerNorm(dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim * 4), #  空间变大 4 倍，信息可以被更细致地分离、筛选、重组
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim * 4, dim),
        #     nn.Dropout(dropout),
        # )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm':      nn.LayerNorm(dim),
                'self_attn': SelfAttention(dim, num_heads, dropout),
                'ffn_norm':  nn.LayerNorm(dim),
                'ffn':       nn.Sequential(
                    nn.Linear(dim, dim * 4), #  空间变大 4 倍，信息可以被更细致地分离、筛选、重组
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, x):
        # residual = x
        # x, self_attn_weight = self.attn(self.norm(x))
        # x = x + residual
        # x = x + self.ffn(self.ffn_norm(x))
        
        # x = x + self.attn(self.norm(x))[0]
        # x = x + self.ffn(self.ffn_norm(x))

        self_attn_weight_list = []

        for layer in self.layers:
            # Self-Attention + 残差
            residual = x
            x, self_attn_weight = layer['self_attn'](layer['norm'](x))
            x = x + residual
            self_attn_weight_list.append(self_attn_weight)
            x = x + layer['ffn'](layer['ffn_norm'](x)) # FFN + 残差

        return x, self_attn_weight_list
    

class CrossAttentionBlock(nn.Module):
    """模态间 Cross-Attention + FFN，一个模态查询其他三个模态"""
    def __init__(self, dim, depth=3, num_heads=8, dropout=0.1):
        super().__init__()

        # self.norm_q = nn.LayerNorm(dim)
        # self.norm_ctx = nn.LayerNorm(dim)
        # self.cross_attn = CrossAttention(dim, num_heads, dropout)
        # self.ffn_norm = nn.LayerNorm(dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim * 4, dim),
        #     nn.Dropout(dropout),
        # )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm_q':    nn.LayerNorm(dim),
                'norm_ctx':  nn.LayerNorm(dim),
                'cross_attn': CrossAttention(dim, num_heads, dropout),
                'ffn_norm':  nn.LayerNorm(dim),
                'ffn':       nn.Sequential(
                    nn.Linear(dim, dim * 4), #  空间变大 4 倍，信息可以被更细致地分离、筛选、重组
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, x, context):
        """
        x: (B, L_q, D)  查询方
        context: (B, L_ctx, D)  被查询方（已拼接好的其他模态）
        """
        # residual = x
        # x, cross_attn_weight = self.cross_attn(self.norm_q(x), self.norm_ctx(context))
        # x = x + residual
        # x = x + self.ffn(self.ffn_norm(x))

        # x = x + self.cross_attn(self.norm_q(x), self.norm_ctx(context))[0]
        # x = x + self.ffn(self.ffn_norm(x))

        cross_attn_weight_list = []

        for layer in self.layers:
            # Cross-Attention + 残差
            residual = x
            x, cross_attn_weight = layer['cross_attn'](layer['norm_q'](x), layer['norm_ctx'](context))
            x = x + residual
            cross_attn_weight_list.append(cross_attn_weight)
            x = x + layer['ffn'](layer['ffn_norm'](x)) # FFN + 残差

        return x, cross_attn_weight_list


class SingleModalityFusion(nn.Module):
    """
    单模态，多模式融合
    使用自注意力，融合单个模态的不同模式的信息
    """
    def __init__(self, channel, dim, depth, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn_block = SelfAttentionBlock(dim, depth, num_heads, dropout)
        self.multiPattern_linear = nn.Linear(channel*dim, dim)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x, self_attn_weight = self.self_attn_block(x) # x: (B, C, 512)；self_attn_weight: (B, head, C, C)

        # 信息流如果没有残差连接且加上层归一化（抹平不同模式之间的信息差异），会丢失很多信息，前面性能下降的原因
        # 这里维度下降较大，容易出现极端值，前面不稳定原因，使用FP32可以容忍

        # 以下均为FP32
        # /root/epfs/trained_log_results/SLN/MF_ResNet_AttnFusion_MRMGUS_Clinical_bs10_all_img_centers1_8_Date0714_fp32_1e-3

        x = x.view(x.size(0), -1)         # (B, C*512) 
        x = self.multiPattern_linear(x)   # (B, 512)
        x = x.unsqueeze(1) # (B, 1, 512) 增加模态通道维度

        return x, self_attn_weight


class SingleModalityCLSFusion(nn.Module):
    def __init__(self, dim, depth, num_heads=8, dropout=0.1):
        super().__init__()
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim)) # 用可学习的token向量，池化融合两个模式
        self.attn_block = SelfAttentionBlock(dim, depth, num_heads, dropout)

    def forward(self, x):
        # x: (B, 2, 512)
        B = x.shape[0]
        pool = self.pool_token.expand(B, -1, -1)   # (B, 1, 512)
        x = torch.cat([pool, x], dim=1)            # (B, 3, 512)

        x, self_attn_weight = self.attn_block(x)    # (B, 3, 512)
        x = x[:, 0, :]                              # 只取 pool (B, 512)
        x = x.unsqueeze(1)                          # (B, 1, 512)

        return x, self_attn_weight


class MultiModalityCLSFusion(nn.Module):
    def __init__(self, dim, depth, num_heads=8, dropout=0.1):
        super().__init__()
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim)) # 用可学习的token向量，池化融合两个模式
        self.attn_block = SelfAttentionBlock(dim, depth, num_heads, dropout)

    def forward(self, x):
        # x: (B, 4, 512)
        B = x.shape[0]
        pool = self.pool_token.expand(B, -1, -1)   # (B, 1, 512)
        x = torch.cat([pool, x], dim=1)            # (B, 5, 512)

        x, self_attn_weight = self.attn_block(x)    # (B, 5, 512)
        x = x[:, 0, :]                              # 只取 pool (B, 512) 
        # x = x.unsqueeze(1)                          # (B, 1, 512)

        return x, self_attn_weight


class CrossModalityFusionTest(nn.Module):
    """
    多模态融合
    使用交叉注意力，让模态间先互相查询信息
    使用自注意力，融合多模态信息
    """
    def __init__(self, channel, dim, depth_cross=1, depth_self=2, num_classes=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn_MR = CrossAttentionBlock(dim, depth_cross, num_heads, dropout) # dim为输出维度, depth为交叉注意力模块层数
        self.cross_attn_MG = CrossAttentionBlock(dim, depth_cross, num_heads, dropout)
        self.cross_attn_US = CrossAttentionBlock(dim, depth_cross, num_heads, dropout)
        self.cross_attn_Clinical = CrossAttentionBlock(dim, depth_cross, num_heads, dropout)
        
        self.self_attn = SelfAttentionBlock(dim, depth_self, num_heads, dropout) # dim为输出维度, depth为自注意力模块层数
        self.norm = nn.LayerNorm(channel*dim)
        self.multiModality_linear = nn.Linear(channel*dim, num_classes)

    def forward(self, x, context):
        """
        x：询问模态 (B, 1, 512)
        context：被询问模态 (B, C, 512)
        """
        # 每个模态从其他模态中提取信息
        x_MR, cross_attn_MR_ToMGUSClinical = self.cross_attn_MR(x, context) # x_MR: (B, 1, 512)；cross_attn_weight_MR: (B, head, x_C, context_C)
        x_MG, cross_attn_MG_ToMRUSClinical = self.cross_attn_MG(x, context) # 使用交叉注意力，进行多模态交互
        x_US, cross_attn_US_ToMRMGClinical = self.cross_attn_US(x, context) 
        x_Clinical, cross_attn_Clinical_ToMRMGUS = self.cross_attn_Clinical(x, context)

        # 各模态互相理解
        x_all = torch.cat((x_MR, x_MG, x_US, x_Clinical), dim=1) # 拼接成四通道，自注意力，融合多模态
        x, self_attn_all = self.self_attn(x_all) # x: (B, C, 512)；self_attn_weight：(B, head, C, C)

        x = x.view(x.size(0), -1)          # (B, C*512)
        x = self.norm(x)
        x = self.multiModality_linear(x)   # (B, 2)

        return x, cross_attn_MR_ToMGUSClinical, cross_attn_MG_ToMRUSClinical, \
                cross_attn_US_ToMRMGClinical, cross_attn_Clinical_ToMRMGUS, \
                self_attn_all




if __name__ == '__main__':
    MR_DCE_feature = torch.rand((2, 512))
    MR_DWI_feature = torch.rand((2, 512))
    MR_features = torch.stack((MR_DCE_feature, MR_DWI_feature),dim=1)
    
    MG_CC_feature = torch.rand((2, 512))
    MG_MLO_feature = torch.rand((2, 512))
    MG_features = torch.stack((MG_CC_feature, MG_MLO_feature),dim=1)

    US_H_feature = torch.rand((2, 512))
    US_V_feature = torch.rand((2, 512))
    US_features = torch.stack((US_H_feature, US_V_feature),dim=1)

    clinical_feature = torch.rand((2, 512))

    all_features = torch.stack((MR_DCE_feature, MR_DWI_feature,\
                                MG_CC_feature, MG_MLO_feature,\
                                US_H_feature, US_V_feature, clinical_feature),dim=1)

    # MR_DCE_feature = torch.rand((2, 1, 512))
    # MR_DWI_feature = torch.rand((2, 1, 512))

    # self_attn_model = SelfAttention(dim=512)
    # a, b = self_attn_model(all_features)


    # cross_attn_model = CrossAttention(dim=512)
    # a, b = cross_attn_model(all_features, all_features)
    # print(a)


    # self_attn_modelBlock = SelfAttentionBlock(dim=512, depth=2)
    # a, b = self_attn_modelBlock(MR_features) # 自注意力，单模态多模式融合
    # print(a)

    # x = torch.rand((2, 1, 512))
    # context = MR_DWI_feature = torch.rand((2, 3, 512))
    # cross_attn_modelBlock = CrossAttentionBlock(dim=512, depth=2)
    # a, b = cross_attn_modelBlock(x, context)
    # print(a)


    single_modality_fusion = SingleModalityFusion(2, 512, 1, num_heads=2, dropout=0.1)
    a, b = single_modality_fusion(MR_features) # MR_features 
    print(a)

    # x = torch.rand((2, 1, 512))
    # context = MR_DWI_feature = torch.rand((2, 3, 512))
    # cross_modality_fusion = CrossModalityFusionTest(channel=4, dim=512, num_classes=2)
    # a, b = cross_modality_fusion(x, context) # MR_features
    # print(a)