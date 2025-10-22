# mapper_model.py

import torch
import torch.nn as nn

class FeatureMapper(nn.Module):
    """
    一个特征映射器，将拼接后的CLIP特征映射到一个新的、任务相关的空间。
    架构基于 Attention + MLP。
    """
    def __init__(self, input_dim=1536, output_dim=128, num_heads=12, mlp_hidden_dim=512):
        """
        初始化模型层。
        - input_dim: ViT-L/14的图文特征拼接后维度 (768 + 768 = 1536)
        - output_dim: 优化后特征的维度
        - num_heads: 多头注意力机制的头数 (1536是12的倍数，所以12是个好选择)
        """
        super().__init__()
        
        # 使用一个Transformer编码器层作为Attention+MLP的简洁实现
        # 它内部包含了多头自注意力、残差连接、层归一化和前馈网络(MLP)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            activation='gelu',
            batch_first=True # 让输入batch维度在最前面
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 最后的线性映射层，将特征维度降到最终的目标维度
        self.output_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播逻辑。
        - x: 输入的原始特征，shape: (batch_size, input_dim)
        """
        # Transformer编码器期望的输入是 (batch, seq_len, embed_dim)
        # 我们将每个特征视为一个长度为1的序列
        x = x.unsqueeze(1) # -> shape: (batch_size, 1, input_dim)
        
        # 通过Transformer层
        transformer_output = self.transformer_encoder(x)
        
        # 将序列维度再移除
        transformer_output = transformer_output.squeeze(1) # -> shape: (batch_size, input_dim)
        
        # 通过最后的映射层得到优化特征
        mapped_feature = self.output_projection(transformer_output)
        
        return mapped_feature