import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip


class TopKSampler:
    """
    一个独立的采样器，使用训练好的多模态提取器为数据集中的每个查询找到 top-K 相似样本。
    它会在内部加载自己的特征编码器。
    """
    def __init__(self, extractor, tokenizer, device="cuda"):
        """
        初始化 TopKSampler.

        Args:
            extractor: 训练好的多模态特征提取器模型。
            tokenizer: 用于处理文本的分词器。
            device (str): 运行计算的设备。
        """
        self.device = device # 确保 device 先被定义

        print("--- TopKSampler: 正在加载独立的特征编码器 ---")
        self.extractor = extractor
        self.text_feature_encoder = AutoModelForCausalLM.from_pretrained(
            "anas-awadalla/mpt-1b-redpajama-200b",
            trust_remote_code=True,
        ).to(self.device)
        self.text_feature_encoder.eval()
        
        self.vision_encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.vision_encoder.to(self.device)
        
        self.tokenizer = tokenizer

        self.extractor.to(self.device)
        self.extractor.eval()
        print("--- TopKSampler: 编码器加载完成 ---")

    @torch.no_grad()
    def _get_multimodal_features(self, images, texts):
        # 1. Vision Feature Extraction
        images = images.to(self.device)
        vision_features = self.vision_encoder.encode_image(images, normalize=True)

        # 2. Text Feature Extraction
        self.tokenizer.padding_side = "left"
        text_tokens = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)

        # [关键修复] 使用正确的变量名 self.text_feature_encoder
        out = self.text_feature_encoder(
            **text_tokens,
            output_hidden_states=True,
        )
        text_features = out.hidden_states[-1]

        # 3. Fuse features
        fused_features = self.extractor(vision_features.detach(), text_features.detach())
        return fused_features

    @torch.no_grad()
    def find_top_k_for_each_query(self, dataset, k=1, batch_size=32):
        """
        For each sample in the dataset, finds the top-k most similar samples.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to search within.
            k (int): The number of top samples to retrieve for each query.
            batch_size (int): The batch size for feature computation.

        Returns:
            dict: A dictionary mapping each query_id to a list of the top-k
                  most similar example_ids.
                  e.g., {query_id_1: [top_1_ex_id, top_2_ex_id], ...}
        """
        print(f"--- Starting Top-{k} sample retrieval ---")
        
        # --- Step 1: Compute and cache all feature vectors for the dataset ---
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_features = []
        all_ids = []

        print("--> Step 1/3: Computing feature vectors for all samples...")
        for batch in tqdm(dataloader, desc="Computing Features"):
            images, texts = batch['image'], batch['question']
            
            features = self._get_multimodal_features(images, texts)
            all_features.append(features)
            all_ids.extend(batch['question_id'].tolist())

        all_features = torch.cat(all_features, dim=0)
        print(f"    Done. Computed {all_features.shape[0]} feature vectors of dimension {all_features.shape[1]}.")

        # --- Step 2: Compute the similarity matrix ---
        print("--> Step 2/3: Calculating similarity matrix...")
        # Normalize features for efficient cosine similarity calculation
        all_features_norm = all_features / all_features.norm(dim=1, keepdim=True)
        # Similarity matrix is the dot product of normalized features
        similarity_matrix = torch.matmul(all_features_norm, all_features_norm.T)
        print("    Done.")

        # --- Step 3: Find the top-k for each query ---
        print(f"--> Step 3/3: Retrieving top-{k} samples for each query...")
        retrieved_samples = {}
        for i in tqdm(range(len(all_ids)), desc="Retrieving Samples"):
            query_id = all_ids[i]
            
            # Get similarity scores for this query against all other samples
            scores = similarity_matrix[i]
            
            # A sample cannot be its own nearest neighbor, so set its self-similarity to a very low value
            scores[i] = -1.0
            
            # Get the top k scores and their indices
            _, top_k_indices = torch.topk(scores, k=k)
            
            # Map the indices back to their original question_ids
            top_k_ids = [all_ids[idx] for idx in top_k_indices]
            
            retrieved_samples[query_id] = top_k_ids

        print("--- Sample retrieval complete! ---")
        return retrieved_samples