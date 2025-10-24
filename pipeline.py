# pipeline.py

import os, json, random, torch
import model
import loss
import dataset
import argparse
import warnings
import tqdm
from torch.utils.data import DataLoader, Subset
from utils import create_model
from PIL import Image
from pathlib import Path
from torch.optim import AdamW
from utils import read_cfg, set_seeds, create_dataset, create_loss, FeatureExtractorTrainer, TripletDataset
from open_flamingo import create_model_and_transforms
from sampler import ZeroshotSampler, OneShotSampler
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
    dataset = create_dataset(exp_cfg.get("dataset"))
    dst_train = dataset.build(mode='train', image_processor=image_processor)
    dst_val = dataset.build(mode='val', image_processor=image_processor)

    # # 计算0-shot
    # zeroshot_sampler = ZeroshotSampler(model, image_processor, tokenizer, device)
    # zeroshot_scores = zeroshot_sampler.calculate_scores(dst_train)

    # # 计算1-shot
    # oneshot_sampler = OneShotSampler(model, image_processor, tokenizer, device)
    # sample_pools = oneshot_sampler.find_samples(dst_train, zeroshot_scores, margin=margin)




    import open_clip
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    vision_encoder.to(device)
    inputs = dst_train[0]['image'].unsqueeze(0).to(device)
    vision_embed = vision_encoder.encode_image(inputs, normalize=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # 加载 MPT 语言模型和 tokenizer
    tok = AutoTokenizer.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<PAD>"})

    lm = AutoModelForCausalLM.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True,
    ).to(device)
    lm.resize_token_embeddings(len(tok))

    # 输入文本
    text = dst_train[0]["gt_answer"]
    batch = tok(text, return_tensors="pt").to(device)

    # 前向传播，输出最后一层 hidden states
    with torch.no_grad():
        out = lm(**batch, output_hidden_states=True, return_dict=True)

    # 最后一层每个 token 的嵌入
    embeds = out.last_hidden_state      # shape [B, T, hidden_dim]
    print(embeds.shape)                 # 例如 [1, 12, 2048] 对 MPT-1B

    # 如果你只想要句向量（整句表示）
    lengths = batch["attention_mask"].sum(dim=1)
    sent_emb = embeds[torch.arange(embeds.size(0)), lengths - 1]   # [B, hidden_dim]


    # 为了演示，我们只对验证集的前10个样本进行完整的 one-shot 挖掘
    # 这将进行 10 (查询) * 9 (示例) = 90 次 one-shot 推理
    sample_indices = range(10) 
    val_subset = Subset(dst_val, sample_indices)
    print(f"--- 准备进行 One-shot 样本挖掘，使用验证集的 {len(val_subset)} 个样本 ---")

    # 步骤 1: 计算这些样本的 0-shot 分数作为基准
    zeroshot_sampler = ZeroshotSampler(model, image_processor, tokenizer, device)
    # 注意：我们只在子集上计算0-shot分数
    zeroshot_scores = zeroshot_sampler.calculate_scores(val_subset, batch_size=4) 

    # 步骤 2: 初始化 OneShotSampler 并开始挖掘
    oneshot_sampler = OneShotSampler(model, image_processor, tokenizer, device)
    # 将子集和对应的0-shot分数传入
    sample_pools = oneshot_sampler.find_samples(val_subset, zeroshot_scores, margin=margin)

    # 步骤 3: 打印结果
    print("\n--- 样本挖掘结果 ---")
    for q_id, pools in sample_pools.items():
        print(f"查询 ID: {q_id}")
        print(f"  - 找到的正样本: {pools['positive']}")
        print(f"  - 找到的负样本: {pools['negative']}")
        print("-" * 20)

    # 训练提取器
    # triplet_dataset = TripletDataset(original_dataset=val_subset, sample_pools=sample_pools)
    # optimizer = AdamW(extractor.parameters(), lr=exp_cfg.get("lr", 1e-4))
    # trainer = FeatureExtractorTrainer(
    #     extractor=extractor,
    #     base_model=model,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     device=device
    # )
    # trainer.train(
    #     dataset=triplet_dataset, 
    #     epochs=exp_cfg.get("epochs", 2), 
    #     batch_size=exp_cfg.get("batch_size", 8)
    # )






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="config/experiment/main.yaml")
    parser.add_argument("--model_config", type=str, default="config/model/extractor.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)