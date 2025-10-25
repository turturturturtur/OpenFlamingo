# utils/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

class TripletDataset(Dataset):
    """
    Dynamically creates (query, positive_doc, negative_doc) triplets for training.
    """
    def __init__(self, original_dataset, sample_pools):
        self.original_dataset = original_dataset
        self.sample_pools = sample_pools
        self.q_id_to_idx = {self.original_dataset[i]['question_id']: i for i in range(len(self.original_dataset))}
        
        self.valid_queries = [
            q_id for q_id, pools in self.sample_pools.items()
            if pools['positive'] and pools['negative']
        ]
        
        if not self.valid_queries:
            raise ValueError("No valid queries with both positive and negative samples found.")
        print(f"--- TripletDataset: Found {len(self.valid_queries)} queries with both positive and negative samples.")

    def __len__(self):
        return len(self.valid_queries)

    def __getitem__(self, index):
        q_id = self.valid_queries[index]
        pos_pool = self.sample_pools[q_id]['positive']
        neg_pool = self.sample_pools[q_id]['negative']
        
        d_pos_id = random.choice(pos_pool)
        d_neg_id = random.choice(neg_pool)
        
        q_idx = self.q_id_to_idx[q_id]
        d_pos_idx = self.q_id_to_idx[d_pos_id]
        d_neg_idx = self.q_id_to_idx[d_neg_id]
        
        q_data = self.original_dataset[q_idx]
        d_pos_data = self.original_dataset[d_pos_idx]
        d_neg_data = self.original_dataset[d_neg_idx]
        
        # [修改] 返回三元组的所有图像和问题
        return {
            'q': (q_data['image'], q_data['question']),
            'd_pos': (d_pos_data['image'], d_pos_data['question']),
            'd_neg': (d_neg_data['image'], d_neg_data['question'])
        }

class FeatureExtractorTrainer:
    def __init__(self, extractor, base_model, tokenizer, loss_fn, optimizer, device="cuda"):
        self.extractor = extractor
        self.base_model = base_model
        self.tokenizer = tokenizer # [新增] 需要分词器来处理文本
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.text_feature_encoder = AutoModelForCausalLM.from_pretrained(
            "anas-awadalla/mpt-1b-redpajama-200b",
            trust_remote_code=True,
        ).to(device)
        self.text_feature_encoder.eval()
        
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _get_multimodal_features(self, images, texts):
        """
        [新版本] 从冻结的 base_model 中提取视觉和文本特征。
        """
        # --- 1. 提取视觉特征 ---
        images = images.to(self.device)
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        vision_encoder.to(self.device)
        vision_embed = vision_encoder.encode_image(images, normalize=True)

        # --- 2. 提取文本特征 ---
        self.tokenizer.padding_side = "left"
        text_tokens = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)

        # 【遵从你的示例】调用独立的 text_encoder
        out = self.text_feature_encoder(
            **text_tokens,
            output_hidden_states=True,
            return_dict=True, # return_dict is the default and good practice
            use_cache=False
        )
        
        # 【遵从你的示例】取最后一层的隐藏状态
        text_embeds = out.hidden_states[-1]

        return vision_embed, text_embeds

    def train_one_epoch(self, dataloader):
        self.extractor.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training Epoch"):
            # [修改] 解包图像和文本
            q_images, q_texts = batch['q']
            d_pos_images, d_pos_texts = batch['d_pos']
            d_neg_images, d_neg_texts = batch['d_neg']
            
            with torch.no_grad():
                # 分别提取三组特征
                base_q_vis, base_q_txt = self._get_multimodal_features(q_images, q_texts)
                base_d_pos_vis, base_d_pos_txt = self._get_multimodal_features(d_pos_images, d_pos_texts)
                base_d_neg_vis, base_d_neg_txt = self._get_multimodal_features(d_neg_images, d_neg_texts)
            
            self.optimizer.zero_grad()
            
            # [修改] 将两种模态的特征都输入到 extractor 中
            q_features = self.extractor(base_q_vis, base_q_txt)
            d_pos_features = self.extractor(base_d_pos_vis, base_d_pos_txt)
            d_neg_features = self.extractor(base_d_neg_vis, base_d_neg_txt)
            
            loss = self.loss_fn(q_features, d_pos_features, d_neg_features)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def train(self, dataset, epochs, batch_size, num_workers=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        print("--- Starting training of the Feature Extractor ---")
        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
        
        print("--- Training finished ---")