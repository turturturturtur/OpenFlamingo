import os
import argparse
import numpy as np
import torch
import clip
import faiss
from PIL import Image
from tqdm import tqdm

# 从您的主脚本中复制或导入VQAv2Dataset类
# (确保这里的 VQAv2Dataset 代码是最新的、能正确工作的版本)
from contrastive_data_constructor import VQAv2Dataset # 假设您的主脚本名为 your_main_script_name.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from a dataset and build a FAISS index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 这个格式化器会自动显示默认值
    )
    
    # --- 输入路径 ---
    parser.add_argument('--questions_path', type=str, 
                        default='/data/public/datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='Path to the VQA questions file (JSON)')
                        
    parser.add_argument('--annotations_path', type=str, 
                        default='/data/public/datasets/vqav2/v2_mscoco_train2014_annotations.json',
                        help='Path to the VQA annotations file (JSON)')

    parser.add_argument('--img_root', type=str, 
                        default='/data/public/datasets/coco',
                        help='Root directory of COCO images')

    parser.add_argument('--split', type=str, 
                        default='train2014',
                        help='Image split name (e.g., "train2014")')

    # --- 输出路径 ---
    parser.add_argument('--output_prefix', type=str, 
                        default="./faiss_index/vqa_train", 
                        help="Prefix for the saved FAISS index, features, and map files")
                        
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # 1. 加载CLIP模型
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    print("CLIP model loaded.")

    # 2. 加载完整的数据集 (候选池)
    candidate_dataset = VQAv2Dataset(
        questions_file=args.questions_path,
        annotations_file=args.annotations_path,
        image_root=args.img_root,
        split=args.split
    )
    
    all_features = []
    original_indices = []

    # 3. 遍历数据集，提取特征
    print("Extracting features from the candidate pool...")
    for i in tqdm(range(len(candidate_dataset))):
        sample = candidate_dataset[i]
        if sample is None:
            continue

        image = preprocess(sample['image']).unsqueeze(0).to(DEVICE)
        # 我们只对问题进行编码，因为答案未知。可以考虑只用图像特征或图文特征。
        # 这里我们将图像和问题特征拼接，以获得更丰富的表示。
        text = clip.tokenize([sample['question']], truncate=True).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # 归一化特征，这对于使用L2距离进行余弦相似度搜索很重要
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 将图文特征拼接起来
            combined_features = torch.cat([image_features, text_features], dim=-1)
            all_features.append(combined_features.cpu().numpy())
            original_indices.append(sample['index'])

    # 4. 构建并保存FAISS索引
    features_array = np.vstack(all_features).astype('float32')
    feature_dim = features_array.shape[1]
    
    print(f"Building FAISS index with {features_array.shape[0]} vectors of dimension {feature_dim}...")
    
    # 使用 IndexFlatL2，因为归一化向量的L2距离与余弦相似度成反比
    index = faiss.IndexFlatL2(feature_dim)
    index.add(features_array)

    # 保存索引和特征
    faiss_index_path = f"{args.output_prefix}.faiss"
    features_path = f"{args.output_prefix}_features.npy"
    indices_map_path = f"{args.output_prefix}_indices_map.npy"
    
    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)
    
    print(f"Saving features to {features_path}")
    np.save(features_path, features_array)

    print(f"Saving original indices map to {indices_map_path}")
    np.save(indices_map_path, np.array(original_indices))
    
    print("Done.")

if __name__ == "__main__":
    main()