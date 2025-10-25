# pipeline.py

import os, json, random, torch
import model
import loss
import dataset
import argparse
import warnings
import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils import create_model
from PIL import Image
from pathlib import Path
from torch.optim import AdamW
from utils import read_cfg, set_seeds, create_dataset, create_loss, FeatureExtractorTrainer, TripletDataset
from open_flamingo import create_model_and_transforms
from sampler import ZeroshotSampler, OneShotSampler, TopKSampler
from transformers import logging


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def main(args):
    # 读取实验配置
    exp_cfg = read_cfg(args.exp_config)
    model_cfg = read_cfg(args.model_config)
    set_seeds(exp_cfg.get("seed", args.seed))
    margin = exp_cfg.get("margin")

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",   # 或 "laion2b_s32b_b82k"
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
    )
    model.to(device)
    model.eval()

    # 加载提取器
    if exp_cfg.get("extractor"):
        extractor = create_model("extractor", **model_cfg)
        extractor.to(device)
        extractor.train()
        print(f"--- 成功加载特征提取器 ---")

    # 加载loss
    loss_fn = create_loss(exp_cfg.get("loss_name"), **exp_cfg.get("loss_param"))

    # 初始化数据集
    print(f"--- 正在构建数据集 ---")
    dataset = create_dataset(exp_cfg.get("dataset"))
    dst_train = dataset.build(mode='train', image_processor=image_processor)
    dst_val = dataset.build(mode='val', image_processor=image_processor)

    # 计算0-shot
    print(f"--- 计算训练集的 0-shot 分数 ---")
    zeroshot_sampler = ZeroshotSampler(model, image_processor, tokenizer, device)
    zeroshot_scores = zeroshot_sampler.calculate_scores(dst_train)

    # 计算1-shot
    print(f"--- 挖掘训练集的 One-shot 样本 ---")
    oneshot_sampler = OneShotSampler(model, image_processor, tokenizer, device)
    sample_pools = oneshot_sampler.find_samples(dst_train, zeroshot_scores, margin=margin)

    # 训练提取器
    print(f"--- 训练特征提取器 ---")
    triplet_dataset = TripletDataset(original_dataset=dst_train, sample_pools=sample_pools)
    optimizer = AdamW(extractor.parameters(), lr=exp_cfg.get("lr", 1e-4))
    trainer = FeatureExtractorTrainer(
        extractor=extractor,
        base_model=model,
        tokenizer=tokenizer, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    trainer.train(
        dataset=triplet_dataset, 
        epochs=exp_cfg.get("epochs", 2), 
        batch_size=exp_cfg.get("batch_size", 8)
    )

    # 使用训练好的 Extractor 进行 Top-K 检索
    print("--- 正在使用训练好的 Extractor 检索 Top-1 ---")
    retriever = TopKSampler(
        extractor=extractor,
        tokenizer=tokenizer,
        device=device
    )
    # 在整个验证集上运行检索
    top_1_samples = retriever.find_top_k_for_each_query(dst_val, k=1, batch_size=32)
    oneshot_final_score = oneshot_sampler.eval(dst_val, top_1_samples)

    # 输出统计结果
    zeroshot_avg_score = np.mean(list(zeroshot_scores.values()))
    print("\n" + "="*50)
    print(f"0-shot 平均分数: {zeroshot_avg_score:.4f}")
    print(f"使用检索示例后的 One-shot 平均分数: {oneshot_final_score:.4f}")
    print(f"提升的性能为: {oneshot_final_score - zeroshot_avg_score:.4f}")
    print("="*50 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="config/experiment/main.yaml")
    parser.add_argument("--model_config", type=str, default="config/model/extractor.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)