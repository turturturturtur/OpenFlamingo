# loss/sin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import LOSS

@LOSS.register("triplet_loss")
class TripletSimilarityLoss(nn.Module):
    """
    一个基于余弦相似度的 Triplet Loss 变体。
    
    目标是最大化 (sim(q, d⁺) - sim(q, d⁻))，
    这等价于最小化 Loss = sim(q, d⁻) - sim(q, d⁺)。
    这个损失函数会推动 d⁺ 更接近 q，同时推动 d⁻ 更远离 q。
    """
    def __init__(self, margin=0.0):
        """
        Args:
            margin (float, optional): 一个可选的边际值。如果 (sim(q, d⁺) - sim(q, d⁻)) 已经大于 margin，
                                      则损失为0。这可以防止模型在已经很好的样本上过度优化。
                                      默认为0，即严格执行 sim(q, d⁺) > sim(q, d⁻)。
        """
        super().__init__()
        self.margin = margin
        # 使用 PyTorch 内置的余弦相似度模块，dim=1 表示在特征维度上计算
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, q_features, d_pos_features, d_neg_features):
        """
        计算损失。

        Args:
            q_features (torch.Tensor): 查询样本的特征向量，形状 [B, feature_dim]。
            d_pos_features (torch.Tensor): 正样本的特征向量，形状 [B, feature_dim]。
            d_neg_features (torch.Tensor): 负样本的特征向量，形状 [B, feature_dim]。

        Returns:
            torch.Tensor: 一个标量，表示该批次的平均损失。
        """
        # 计算相似度
        sim_pos = self.cosine_similarity(q_features, d_pos_features)
        sim_neg = self.cosine_similarity(q_features, d_neg_features)

        # 计算原始损失
        loss = sim_neg - sim_pos

        # 如果需要，可以加入 margin
        # loss_with_margin = F.relu(loss + self.margin)
        # return loss_with_margin.mean()

        # 根据你的要求，我们直接返回平均损失
        return loss.mean()