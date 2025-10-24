# utils/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

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
        
        return (
            self.original_dataset[q_idx]['image'],
            self.original_dataset[d_pos_idx]['image'],
            self.original_dataset[d_neg_idx]['image']
        )

class FeatureExtractorTrainer:
    """
    Handles the training loop for the feature extractor model.
    """
    def __init__(self, extractor, base_model, loss_fn, optimizer, device="cuda"):
        self.extractor = extractor
        self.base_model = base_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _get_base_features(self, images):
        """
        Extracts post-perceiver features from the frozen base model.
        This is the "more optimal" approach.
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        # 1. Get raw patch features from the vision encoder
        vision_x = self.base_model.vision_encoder(images)[0]
        
        # 2. Resample features through the perceiver
        vision_x = self.base_model.perceiver(vision_x)
        
        return vision_x

    def train_one_epoch(self, dataloader):
        self.extractor.train()
        total_loss = 0.0
        
        for q_images, d_pos_images, d_neg_images in tqdm(dataloader, desc="Training Epoch"):
            with torch.no_grad():
                base_q_features = self._get_base_features(q_images)
                base_d_pos_features = self._get_base_features(d_pos_images)
                base_d_neg_features = self._get_base_features(d_neg_images)
            
            self.optimizer.zero_grad()
            
            q_features = self.extractor(base_q_features)
            d_pos_features = self.extractor(base_d_pos_features)
            d_neg_features = self.extractor(base_d_neg_features)
            
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