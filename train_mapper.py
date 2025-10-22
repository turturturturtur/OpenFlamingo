# train_mapper.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import argparse
from tqdm import tqdm
import os
# 从本地文件导入模型
from mapper_model import FeatureMapper

# 1. 创建 Triplet Dataset 类
class TripletDataset(Dataset):
    def __init__(self, triplets_path, train_features_path, test_features_path):
        print(f"Loading triplet data from: {triplets_path}")
        with open(triplets_path, 'r') as f:
            # 只加载一部分用于快速测试
            self.triplets = json.load(f)
        
        print(f"Loading feature lookups...")
        # 使用 mmap_mode='r' 来“懒加载”大文件，避免消耗过多RAM
        self.train_features = np.load(train_features_path, mmap_mode='r')
        self.test_features = np.load(test_features_path, mmap_mode='r')
        print("Features loaded.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q_idx, dp_idx, dn_idx = self.triplets[idx]
        
        # 从 .npy 文件中按索引读取特征
        # 注意：这里的索引必须是整数
        q_feat = torch.from_numpy(self.test_features[int(q_idx)])
        dp_feat = torch.from_numpy(self.train_features[int(dp_idx)])
        dn_feat = torch.from_numpy(self.train_features[int(dn_idx)])
        
        return q_feat, dp_feat, dn_feat

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Feature Mapper model.")
    parser.add_argument('--triplets_path', type=str, required=True, help="Path to the final_triplets.json file")
    parser.add_argument('--features_dir', type=str, default="./feature_lookups", help="Directory containing train/test_features.npy")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--margin', type=float, default=0.2, help="Margin for the TripletMarginLoss")
    parser.add_argument('--output_path', type=str, default="feature_mapper.pth", help="Path to save the trained model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # 实例化所有组件
    dataset = TripletDataset(
        triplets_path=args.triplets_path,
        train_features_path=os.path.join(args.features_dir, 'train_features.npy'),
        test_features_path=os.path.join(args.features_dir, 'test_features.npy')
    )
    # 使用 num_workers 来加速数据加载
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = FeatureMapper().to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环
    print("--- Starting Training ---")
    model.train() # 设置为训练模式
    
    for epoch in range(args.epochs):
        # 使用tqdm包装dataloader
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        
        for q_feat, dp_feat, dn_feat in loop:
            q_feat, dp_feat, dn_feat = q_feat.to(DEVICE), dp_feat.to(DEVICE), dn_feat.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 通过映射器得到优化特征
            q_prime = model(q_feat)
            dp_prime = model(dp_feat)
            dn_prime = model(dn_feat)
            
            loss = criterion(q_prime, dp_prime, dn_prime)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 在进度条上显示实时loss
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.6f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), args.output_path)
    print(f"\nTraining finished. Model saved to {args.output_path}")

if __name__ == "__main__":
    main()