import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

# 假设这些是你项目中的原有导入
from base.base_dataset import BaseDataset
from registry.register import DATASET
from utils.utils import read_cfg

# ==============================================================================
# 1. PyTorch Dataset Class (已按 OpenFlamingo 范式修改)
# ==============================================================================

class VQAv2Dataset(Dataset):
    """
    一个标准的 VQAv2 PyTorch 数据集。
    它现在接收一个 image_processor 来处理图片，以适配 OpenFlamingo 等模型。
    """
    def __init__(self, samples: List[Tuple[Dict, Dict, str]], image_processor: Any):
        """
        初始化函数现在需要传入一个 image_processor。

        Args:
            samples (List[Tuple[Dict, Dict, str]]): 预处理好的样本元数据。
            image_processor (Any): 来自模型的图像预处理器
                                   (例如, 从 transformers 加载的 CLIPImageProcessor)。
        """
        super().__init__()
        if not samples:
            raise ValueError("VQAv2Dataset initialized with an empty list of samples.")
        if image_processor is None:
            raise ValueError("VQAv2Dataset requires an image_processor to be provided.")
            
        self.samples = samples
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        q, a, img_path = self.samples[index]
        
        try:
            # 依然是使用 PIL 加载原始图片
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"警告：无法加载图片 {img_path}。错误: {e}")
            # 如果图片损坏，返回一个占位符或引发错误，这里我们返回 None
            # 注意：在你的 collate_fn 中可能需要处理 None 的情况
            image = None

        # 使用传入的 image_processor 来处理图片
        # 它会完成所有操作：缩放、转 Tensor、归一化
        # processor 通常返回一个包含 'pixel_values' 的字典或直接是张量
        # 我们假设它返回张量，如果不是，你需要相应调整
        if image is not None:
            # image_processor 会返回一个 [C, H, W] 的张量
            image_tensor = self.image_processor(image)
        else:
            # 如果图片加载失败，提供一个尺寸正确的零张量作为占位符
            # 这里的尺寸需要根据你的 image_processor 的配置来确定
            image_tensor = None

        return {
            'image_path': img_path,
            'image_id': q["image_id"],
            'question': q["question"],
            'question_id': q["question_id"],
            'gt_answer': a["multiple_choice_answer"],
            'all_answers': [ans["answer"] for ans in a["answers"]],
            'image': image_tensor  # <--- 返回的是经过 processor 处理好的张量
        }

# ==============================================================================
# 2. Dataset Builder Class (已修改以传递 image_processor)
# ==============================================================================

@DATASET.register('VQAv2')
class VQAv2Builder(BaseDataset):
    """
    VQAv2 数据集的构建器。
    """
    def __init__(self, config_path: str = "config/dataset/vqav2.yaml"):
        super().__init__()
        self.cfg = read_cfg(path=config_path)
        self.vqa_dir = self.cfg.get("vqa_dir")
        self.coco_dir = self.cfg.get("coco_dir")
        self.question_files = self.cfg.get("question_files")
        self.annotation_files = self.cfg.get("annotation_files")
        if not all([self.vqa_dir, self.coco_dir, self.question_files, self.annotation_files]):
            raise ValueError(
                f"Missing required keys in '{config_path}'."
            )

    def _get_coco_path(self, image_id: int, split: str) -> str:
        split_year = f"{split}2014"
        return os.path.join(self.coco_dir, split_year, f"COCO_{split_year}_{image_id:012d}.jpg")

    def build(self, mode: str = 'val', image_processor: Any = None) -> VQAv2Dataset:
        """
        构建并返回一个 VQAv2Dataset 实例。

        Args:
            mode (str): 'train' 或 'val'，决定加载哪个数据集分割。
            image_processor (Any): 必须提供一个图像预处理器。

        Returns:
            VQAv2Dataset: 一个配置好的数据集实例。
        """
        if image_processor is None:
            raise ValueError("The 'build' method requires an 'image_processor'.")

        assert mode in self.question_files, f"Invalid mode '{mode}'."

        cache_dir = "data/checkpoint"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"vqav2_{mode}_samples.pt")

        if os.path.exists(cache_path):
            valid_samples = torch.load(cache_path)
            # <--- 注意：这里需要传入 image_processor
            return VQAv2Dataset(samples=valid_samples, image_processor=image_processor)
        
        # ... (后续的 JSON 加载和路径验证逻辑保持不变) ...
        ques_path = os.path.join(self.vqa_dir, self.question_files[mode])
        ann_path = os.path.join(self.vqa_dir, self.annotation_files[mode])
        
        with open(ques_path, "r") as f:
            ques_data = json.load(f)["questions"]
        with open(ann_path, "r") as f:
            ann_data = json.load(f)["annotations"]

        ann_dict = {a["question_id"]: a for a in ann_data}
        valid_samples = []
        for q in tqdm(ques_data, desc=f"Processing '{mode}' samples"):
            qid = q["question_id"]
            aid = ann_dict.get(qid)
            
            if not aid:
                continue
            
            img_p = self._get_coco_path(q["image_id"], split=mode)
            if Path(img_p).exists():
                valid_samples.append((q, aid, img_p))
        
        if not valid_samples:
            raise RuntimeError(f"No valid samples found for mode '{mode}'.")

        # <--- 注意：在创建实例时传入 image_processor
        dataset_instance = VQAv2Dataset(samples=valid_samples, image_processor=image_processor)
        
        torch.save(valid_samples, cache_path)
        
        return dataset_instance