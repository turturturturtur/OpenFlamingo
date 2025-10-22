# create_feature_lookup.py

import torch
import clip
import numpy as np
from tqdm import tqdm
import argparse
import os

# 导入您自己的数据集类，请确保文件名正确
from contrastive_data_constructor import VQAv2Dataset, ImageNet100Dataset 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract and save CLIP features for all datasets.")
    # 添加所有与数据集加载相关的参数
    parser.add_argument('--task', type=str, required=True, choices=['vqa', 'imagenet'])
    # ... (此处省略了所有路径参数，请从您的主脚本中完整复制过来)
    parser.add_argument('--train_questions_path', type=str, help='Path to VQA training questions file (JSON)')
    parser.add_argument('--train_annotations_path', type=str, help='Path to VQA training annotations file (JSON)')
    parser.add_argument('--test_questions_path', type=str, help='Path to VQA test/val questions file (JSON)')
    parser.add_argument('--test_annotations_path', type=str, help='Path to VQA test/val annotations file (JSON)')
    parser.add_argument('--img_root', type=str, required=True, help='Root directory of images')
    parser.add_argument('--train_split', type=str, default='train2014', help='Split name for training images')
    parser.add_argument('--test_split', type=str, default='val2014', help='Split name for test images')
    # ...
    parser.add_argument('--output_dir', type=str, default="./feature_lookups", help="Directory to save the feature .npy files")
    return parser.parse_args()

def extract_and_save_features(dataset, clip_model, clip_preprocess, output_path):
    all_features = []
    # 使用tqdm包装dataset以显示进度条
    for sample in tqdm(dataset, desc=f"Extracting features for {os.path.basename(output_path)}"):
        if sample is None:
            # 添加一个零向量作为占位符，以保持索引的严格对齐
            # 假设ViT-L/14的图像和文本特征都是768维, 拼接后是1536维
            all_features.append(np.zeros((1536,), dtype=np.float32))
            continue
        
        image = clip_preprocess(sample['image']).unsqueeze(0).to(DEVICE)
        # 使用truncate=True来自动截断过长的文本
        text = clip.tokenize([sample['question']], truncate=True).to(DEVICE)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            
            # 特征归一化，这是一个很好的实践
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 将特征移动到CPU并转换为numpy
            image_features_np = image_features.cpu().numpy()
            text_features_np = text_features.cpu().numpy()

            # 拼接特征
            combined = np.concatenate([image_features_np, text_features_np], axis=-1)
            all_features.append(combined)

    # 将列表中的所有 (1, 1536) 数组堆叠成一个大的 (N, 1536) 数组
    features_array = np.vstack(all_features).astype('float32')
    print(f"Saving {features_array.shape[0]} feature vectors of shape {features_array.shape} to {output_path}")
    np.save(output_path, features_array)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CLIP model (ViT-L/14)...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval() # 设置为评估模式

    # 根据任务加载数据集
    if args.task == "vqa":
        print("Loading VQA train dataset...")
        train_dataset = VQAv2Dataset(
            questions_file=args.train_questions_path,
            annotations_file=args.train_annotations_path,
            image_root=args.img_root,
            split=args.train_split
        )
        print("Loading VQA test dataset...")
        test_dataset = VQAv2Dataset(
            questions_file=args.test_questions_path,
            annotations_file=args.test_annotations_path,
            image_root=args.img_root,
            split=args.test_split
        )
    # ... (此处可以添加elif args.task == "imagenet": 的逻辑)
    else:
        raise ValueError("Task not supported yet.")

    # 提取并保存训练集特征
    train_output_path = os.path.join(args.output_dir, "train_features.npy")
    extract_and_save_features(train_dataset, clip_model, clip_preprocess, train_output_path)

    # 提取并保存测试集特征
    test_output_path = os.path.join(args.output_dir, "test_features.npy")
    extract_and_save_features(test_dataset, clip_model, clip_preprocess, test_output_path)
    
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()