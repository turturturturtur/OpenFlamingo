# # contrastive_data_constructor.py

# # =========================
# #         Imports
# # =========================
# GEN_MAX_NEW_TOKENS = 77 # 增加到CLIP text encoder的最大长度
# GEN_NUM_BEAMS = 5      # 略微增加
# NO_REPEAT_NGRAM_SIZE = 3 # 禁止重复出现任何3-gram
# REPETITION_PENALTY = 1.5 # 施加重复惩罚

# import os
# import json
# import random
# import argparse
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Optional

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# from tqdm import tqdm
# import faiss
# import numpy as np
# import clip

# # 本地模块导入
# from open_flamingo import create_model_and_transforms
# from cider_score import compute_cider_score

# # =========================
# #          配置
# # =========================
# SEED = 42
# random.seed(SEED)
# torch.manual_seed(SEED)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 生成参数
# GEN_MAX_NEW_TOKENS = 50  # 增加了长度以允许更完整的答案
# GEN_NUM_BEAMS = 3

# # =========================
# #        工具函数
# # =========================
# def set_default_tensor_type():
#     if DEVICE == "cuda":
#         torch.set_default_tensor_type(torch.cuda.FloatTensor)
#     else:
#         torch.set_default_tensor_type(torch.FloatTensor)

# def format_mm_prompt(demo_qa: Optional[Tuple[str, str]] = None, query_q: str = "") -> str:
#     if demo_qa is None:
#         # 0-shot: 结尾加上 <|endofchunk|> 可能有助于模型理解这是任务的开始
#         return f"<image>Question: {query_q.strip()} Answer:"
#     else:
#         dq, da = demo_qa
#         # 1-shot: 在演示的答案结束后，明确地加上 <|endofchunk|>
#         return f"<image>Question: {dq.strip()} Answer: {da.strip()}<|endofchunk|><image>Question: {query_q.strip()} Answer:"

# def decode_generation(tokenizer, generated_ids, query_q: str) -> str:
#     """
#     [增强版] 解码模型输出，并进行后处理来提取更干净的答案。
#     """
#     # 1. 正常解码
#     full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
#     # 2. 尝试移除提示词的回声
#     # 找到最后一个 "Answer:" 的位置
#     answer_prompt = "Answer:"
#     last_answer_pos = full_text.rfind(answer_prompt)
    
#     if last_answer_pos != -1:
#         # 如果找到了，就取它后面的部分作为答案
#         clean_answer = full_text[last_answer_pos + len(answer_prompt):].strip()
#         # 如果清理后结果为空，说明模型可能只重复了提示，没有生成新内容
#         # 这种情况下，返回原始输出可能更好，让CIDEr去判断
#         if clean_answer:
#             return clean_answer

#     # 3. 如果上面的方法失败了（比如模型没输出"Answer:"），
#     #    一个备用策略是移除整个query的文本，如果它出现在输出里的话
#     query_text_cleaned = query_q.strip()
#     if full_text.startswith(query_text_cleaned):
#         return full_text[len(query_text_cleaned):].strip()

#     # 如果所有清理都失败了，返回原始解码文本
#     return full_text

# def build_vision_batch_from_images(image_processor, images: List[Image.Image]) -> torch.Tensor:
#     """[已修正] 正确处理torchvision transforms并构建batch"""
#     processed_images = [image_processor(img) for img in images]
#     batch_tensor = torch.stack(processed_images)
#     px = batch_tensor.unsqueeze(1)
#     px = px.unsqueeze(0)
#     return px.to(DEVICE)

# # =========================
# #     数据集定义
# # =========================
# class VQAv2Dataset(Dataset):
#     """[已修正] VQAv2数据集类，处理分离的问答文件"""
#     def __init__(self, questions_file: str, annotations_file: str, image_root: str, split: str):
#         # ... (此处省略代码，请确保您使用的是能正确合并问答的最新版本) ...
#         # ... (为了简洁，假设这个类的代码和我们之前修复的一样) ...
#         super().__init__()
#         with open(questions_file, "r") as f: q_data = json.load(f)['questions']
#         with open(annotations_file, "r") as f: a_data = json.load(f)['annotations']
#         self.image_root = image_root
#         self.split = split
#         qid_to_q = {q['question_id']: q for q in q_data}
#         self.items = [
#             {**ann, 'question': qid_to_q[ann['question_id']]['question']}
#             for ann in a_data if ann['question_id'] in qid_to_q
#         ]

#     def __len__(self): return len(self.items)

#     def __getitem__(self, idx):
#         item = self.items[idx]
#         image_id = item['image_id']
#         image_filename = f"COCO_{self.split}_{str(image_id).zfill(12)}.jpg"
#         full_image_path = os.path.join(self.image_root, self.split, image_filename)
#         try:
#             img = Image.open(full_image_path).convert("RGB")
#         except FileNotFoundError:
#             tqdm.write(f"Warning: Image not found at {full_image_path}")
#             return None
#         q = item["question"]
#         answers = [ans['answer'] for ans in item["answers"]]
#         item_index = item.get("question_id", idx)
#         return {"image": img, "question": q, "answers": answers, "index": item_index}

# # ... (ImageNet100Dataset 定义，如果需要的话) ...

# # =========================
# #       模型封装
# # =========================
# @dataclass
# class FlamingoBundle:
#     model: torch.nn.Module
#     image_processor: any
#     tokenizer: any

# def load_openflamingo_9b():
#     """
#     加载预训练好的 OpenFlamingo-9B 模型。
#     """
#     print("正在初始化模型结构和预处理器...")
    
#     # create_model_and_transforms 会创建模型的“骨架”以及图片、文本的“预处理器”
#     model, image_processor, tokenizer = create_model_and_transforms(
#         # 1. 视觉编码器 (模型的眼睛)，使用OpenCLIP的大模型
#         clip_vision_encoder_path="ViT-L-14",
#         clip_vision_encoder_pretrained="openai",
        
#         # 2. 语言模型 (模型的大脑)，使用MPT-7B
#         lang_encoder_path="anas-awadalla/mpt-7b",
#         tokenizer_path="anas-awadalla/mpt-7b",
        
#         # 3. 交叉注意力层 (连接眼睛和大脑的神经)
#         # 对于9B模型，官方建议每隔4层语言模型层插入一个交叉注意力层
#         cross_attn_every_n_layers=4
#     )

#     # 打印模型结构，确认模型已创建
#     # print(model)

#     print("模型结构创建完毕。正在从Hugging Face Hub下载预训练权重...")

#     # hf_hub_download 会自动从网上下载模型的“灵魂”（已训练好的参数）
#     # "openflamingo/OpenFlamingo-9B-vitl-mpt7b" 是9B模型在Hugging Face上的官方名称
#     checkpoint_path = hf_hub_download(
#         "openflamingo/OpenFlamingo-9B-vitl-mpt7b", 
#         "checkpoint.pt"
#     )
    
#     # 加载权重文件
#     model_weights = torch.load(checkpoint_path, map_location="cpu")
    
#     # 将“灵魂”注入“骨架”
#     model.load_state_dict(model_weights, strict=False)
    
#     print("OpenFlamingo-9B 模型加载成功！")
    
#     return model, image_processor, tokenizer

# # =========================
# #   推理与核心标注逻辑
# # =========================
# @torch.no_grad()
# def infer_one_shot(bundle: FlamingoBundle, demo_img: Image.Image, demo_q: str, demo_a: str, q_image: Image.Image, q_text: str) -> str:
#     vision_x = build_vision_batch_from_images(bundle.image_processor, [demo_img, q_image])
#     prompt = format_mm_prompt((demo_q, demo_a), q_text)
#     lang = bundle.tokenizer([prompt], return_tensors="pt", padding=True).to(DEVICE)
#     out = bundle.model.generate(
#         vision_x=vision_x, lang_x=lang["input_ids"], attention_mask=lang["attention_mask"],
#         max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS,no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,repetition_penalty=REPETITION_PENALTY,
#     )
#     return decode_generation(bundle.tokenizer, out[0], q_text)

# @torch.no_grad()
# def infer_zero_shot(bundle: FlamingoBundle, q_image: Image.Image, q_text: str) -> str:
#     vision_x = build_vision_batch_from_images(bundle.image_processor, [q_image])
#     prompt = format_mm_prompt(None, q_text)
#     lang = bundle.tokenizer([prompt], return_tensors="pt", padding=True).to(DEVICE)
#     out = bundle.model.generate(
#         vision_x=vision_x, lang_x=lang["input_ids"], attention_mask=lang["attention_mask"],
#         max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS,no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,repetition_penalty=REPETITION_PENALTY,
#     )
#     return decode_generation(bundle.tokenizer, out[0], q_text)

# @dataclass
# class CandidateResult:
#     idx: int; Pi: float; delta: float; label: str

# def brute_force_labeling(bundle: FlamingoBundle, query_sample: Dict, candidate_pool: List[Dict], delta_threshold: float) -> Tuple[float, List[CandidateResult]]:
#     pred0 = infer_zero_shot(bundle, query_sample["image"], query_sample["question"])
#     P0 = compute_cider_score(pred0, query_sample["answers"])
#     tqdm.write("-" * 50 + f"\n[DEBUG 0-shot] Q: {query_sample['question']}\n[DEBUG 0-shot] GTs: {query_sample['answers']}\n[DEBUG 0-shot] Pred: '{pred0}'\n[DEBUG 0-shot] Score P0: {P0:.4f}\n" + "-" * 50)
    
#     results = []
#     print_limit = 5
#     for i, cand in enumerate(candidate_pool):
#         if cand is None: continue
#         demo_a = cand["answers"][0] # 取第一个答案作为演示
#         pred_i = infer_one_shot(bundle, cand["image"], cand["question"], demo_a, query_sample["image"], query_sample["question"])
#         Pi = compute_cider_score(pred_i, query_sample["answers"])
#         delta = Pi - P0

#         if i < print_limit:
#             tqdm.write(f"\n--- [DEBUG 1-shot with cand #{cand['index']}] ---\n    Demo A: {demo_a}\n    Pred Pi: '{pred_i}'\n    Score Pi: {Pi:.4f}\n    Delta (Pi - P0): {delta:.4f}")

#         label = "neutral"
#         if delta >= delta_threshold: label = "pos"
#         elif delta <= -delta_threshold: label = "neg"
#         results.append(CandidateResult(idx=cand["index"], Pi=Pi, delta=delta, label=label))
#     return P0, results

# def build_pools(results: List[CandidateResult]) -> Tuple[List[int], List[int]]:
#     return [r.idx for r in results if r.label == "pos"], [r.idx for r in results if r.label == "neg"]

# def sample_triplets(query_idx: int, Dq_pos: List[int], Dq_neg: List[int], k: int) -> List[Tuple[int, int, int]]:
#     if not Dq_pos or not Dq_neg: return []
#     return [(query_idx, random.choice(Dq_pos), random.choice(Dq_neg)) for _ in range(k)]

# # =========================
# #     FAISS 检索函数
# # =========================
# def get_top_k_candidates_indices(q_sample, clip_model, clip_preprocess, faiss_index, k):
#     image = clip_preprocess(q_sample['image']).unsqueeze(0).to(DEVICE)
#     text = clip.tokenize([q_sample['question']], truncate=True).to(DEVICE)
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image)
#         text_features = clip_model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         query_features = torch.cat([image_features, text_features], dim=-1).cpu().numpy().astype('float32')
#     _, I = faiss_index.search(query_features, k)
#     return I[0]

# # =========================
# #         主程序
# # =========================
# def parse_args():
#     parser = argparse.ArgumentParser(description="Efficient Triplet Generation with VLM Feedback")
#     # ... (此处省略所有参数定义，请确保您使用的是包含所有路径的最新版本) ...
#     parser.add_argument('--task', type=str, required=True, choices=['vqa', 'imagenet'])
#     parser.add_argument('--train_questions_path', type=str)
#     parser.add_argument('--train_annotations_path', type=str)
#     parser.add_argument('--test_questions_path', type=str)
#     parser.add_argument('--test_annotations_path', type=str)
#     parser.add_argument('--img_root', type=str, required=True)
#     parser.add_argument('--train_split', type=str, default='train2014')
#     parser.add_argument('--test_split', type=str, default='val2014')
#     parser.add_argument('--output_dir', type=str, default="./triplet_output")
#     parser.add_argument('--delta_threshold', type=float, default=0.03)
#     parser.add_argument('--triplets_per_query', type=int, default=5)
#     parser.add_argument('--top_k_retrieval', type=int, default=100)
#     parser.add_argument('--faiss_index_path', type=str, default="./faiss_index/vqa_train.faiss")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     set_default_tensor_type()
    
#     # --- Phase 0: 加载所有模型和数据 ---
#     print("--- Phase 0: Loading Models and Data ---")
#     print("Loading OpenFlamingo-9B model (for expensive re-ranking)...")
#     # !!! 核心修正点: 确保调用的是正确的9B加载函数 !!!
#     model, image_processor, tokenizer = load_openflamingo_9b()
#     # 将加载好的组件打包
#     bundle = FlamingoBundle(model, image_processor, tokenizer)
#     print("OpenFlamingo-9B model successfully wrapped in bundle.")
#     clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
    
#     if args.task == "vqa":
#         # ... (此处省略VQA数据集加载的代码，确保它是最新版本)
#         train_dataset = VQAv2Dataset(args.train_questions_path, args.train_annotations_path, args.img_root, args.train_split)
#         test_dataset = VQAv2Dataset(args.test_questions_path, args.test_annotations_path, args.img_root, args.test_split)
#     else: 
#         raise NotImplementedError("ImageNet not fully implemented in this version.")

#     faiss_index = faiss.read_index(args.faiss_index_path)
#     print(f"Total candidates available in train_dataset: {len(train_dataset)}")
#     print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
    
#     # --- Phase 1: 主循环和两阶段筛选 ---
#     print("\n--- Phase 1: Starting TWO-STAGE Triplet Generation ---")
#     os.makedirs(args.output_dir, exist_ok=True)
#     details_dir = os.path.join(args.output_dir, "q_details")
#     os.makedirs(details_dir, exist_ok=True)
#     all_triplets = []
    
#     for q_sample in tqdm(test_dataset, desc="Processing Queries"):
#         if q_sample is None: continue
        
#         # --- STAGE 1: 快速粗筛 (Fast Pre-filtering) ---
        
#         # !!! 核心修正点: 确保这里的函数调用是完整的 !!!
#         retrieved_indices = get_top_k_candidates_indices(
#             q_sample=q_sample, 
#             clip_model=clip_model, 
#             clip_preprocess=clip_preprocess, 
#             faiss_index=faiss_index,
#             k=args.top_k_retrieval
#         )
        
#         # --- STAGE 2: 昂贵精筛 (按需加载) ---
#         candidate_pool = [train_dataset[idx] for idx in retrieved_indices]
#         tqdm.write(f"Query #{q_sample['index']}: Retrieved and loaded {len(candidate_pool)} candidates for re-ranking.")
        
#         P0, cand_results = brute_force_labeling(bundle, q_sample, candidate_pool, args.delta_threshold)
        
#         Dq_pos, Dq_neg = build_pools(cand_results)
        
#         # --- Phase 2: 结果处理 ---
#         triplets = []
#         generation_successful = False
#         if Dq_pos and Dq_neg:
#             generation_successful = True
#             triplets = sample_triplets(q_sample['index'], Dq_pos, Dq_neg, args.triplets_per_query)
#             all_triplets.extend(triplets)
#         else:
#             tqdm.write(f"Warning: Query #{q_sample['index']} could not generate triplets (pos: {len(Dq_pos)}, neg: {len(Dq_neg)}).")

#         # 无论成功与否，都保存详细的分析文件
#         with open(os.path.join(details_dir, f"q_{q_sample['index']}_details.json"), "w") as f:
#             json.dump({
#                 "query_index": q_sample['index'], "generation_successful": generation_successful,
#                 "P0": P0, "num_pos": len(Dq_pos), "num_neg": len(Dq_neg),
#                 "generated_triplets": triplets, "results": [r.__dict__ for r in cand_results], 
#             }, f, indent=2)

#     # (最后保存所有三元组的代码)
#     final_output_path = os.path.join(args.output_dir, "final_triplets.json")
#     print(f"\nTriplet generation finished. Total triplets generated: {len(all_triplets)}")
#     print(f"Saving the complete triplet dataset to: {final_output_path}")
#     with open(final_output_path, "w") as f: json.dump(all_triplets, f, indent=2)
#     print("Done.")
# if __name__ == "__main__":
#     main()


# contrastive_data_constructor.py (Final Version)

# =========================
#         Imports
# =========================
import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import faiss
import numpy as np
import clip
from huggingface_hub import hf_hub_download

# 本地模块导入
from open_flamingo import create_model_and_transforms
from cider_scorer import compute_cider_score # 确保文件名是 cider_scorer.py

# =========================
#          配置
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- [唯一且正确的] 生成参数 ---
GEN_MAX_NEW_TOKENS = 77 
GEN_NUM_BEAMS = 5      
NO_REPEAT_NGRAM_SIZE = 3 
REPETITION_PENALTY = 1.5 
# --------------------------------

# =========================
#        工具函数
# =========================
def set_default_tensor_type():
    if DEVICE == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

def format_mm_prompt(demo_qa: Optional[Tuple[str, str]] = None, query_q: str = "") -> str:
    if demo_qa is None:
        return f"<image>Question: {query_q.strip()} Answer:"
    else:
        dq, da = demo_qa
        return f"<image>Question: {dq.strip()} Answer: {da.strip()}<|endofchunk|><image>Question: {query_q.strip()} Answer:"

def decode_generation(tokenizer, generated_ids, query_q: str) -> str:
    """[增强版] 解码并清理模型输出"""
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer_prompt = "Answer:"
    last_answer_pos = full_text.rfind(answer_prompt)
    if last_answer_pos != -1:
        clean_answer = full_text[last_answer_pos + len(answer_prompt):].strip()
        if clean_answer:
            return clean_answer
    query_text_cleaned = query_q.strip()
    if full_text.startswith(query_text_cleaned):
        return full_text[len(query_text_cleaned):].strip()
    return full_text

def build_vision_batch_from_images(image_processor, images: List[Image.Image]) -> torch.Tensor:
    processed_images = [image_processor(img) for img in images]
    batch_tensor = torch.stack(processed_images)
    px = batch_tensor.unsqueeze(1).unsqueeze(0)
    return px.to(DEVICE)

# =========================
#     数据集定义
# =========================
class VQAv2Dataset(Dataset):
    def __init__(self, questions_file: str, annotations_file: str, image_root: str, split: str):
        super().__init__()
        with open(questions_file, "r") as f: q_data = json.load(f)['questions']
        with open(annotations_file, "r") as f: a_data = json.load(f)['annotations']
        self.image_root = image_root
        self.split = split
        qid_to_q = {q['question_id']: q for q in q_data}
        self.items = [
            {**ann, 'question': qid_to_q[ann['question_id']]['question']}
            for ann in a_data if ann['question_id'] in qid_to_q
        ]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_id = item['image_id']
        image_filename = f"COCO_{self.split}_{str(image_id).zfill(12)}.jpg"
        full_image_path = os.path.join(self.image_root, self.split, image_filename)
        try:
            img = Image.open(full_image_path).convert("RGB")
        except FileNotFoundError:
            tqdm.write(f"Warning: Image not found at {full_image_path}")
            return None
        q = item["question"]
        answers = [ans['answer'] for ans in item["answers"]]
        item_index = item.get("question_id", idx)
        return {"image": img, "question": q, "answers": answers, "index": item_index}

# =========================
#       模型封装
# =========================
@dataclass
class FlamingoBundle:
    model: torch.nn.Module
    image_processor: any
    tokenizer: any

def load_openflamingo_9b():
    """[已修正] 加载预训练好的 OpenFlamingo-9B 模型"""
    print("正在初始化OpenFlamingo-9B模型结构...")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4 # 9B模型的正确配置
    )
    
    # --- 网络后备方案 ---
    # 如果下面的 hf_hub_download 因为网络问题失败，请取消注释下面两行中的一行
    # 方案一: 使用国内镜像
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 方案二: 设置HTTP/HTTPS代理 (请替换为您的代理地址)
    # os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
    # os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
    # --------------------
    
    print("正在从Hugging Face Hub下载预训练权重...")
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    print("OpenFlamingo-9B 模型加载成功！")

    model = model.to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return FlamingoBundle(model, image_processor, tokenizer)

# =========================
#   推理与核心标注逻辑
# =========================
@torch.no_grad()
def infer_one_shot(bundle: FlamingoBundle, demo_img: Image.Image, demo_q: str, demo_a: str, q_image: Image.Image, q_text: str) -> str:
    vision_x = build_vision_batch_from_images(bundle.image_processor, [demo_img, q_image])
    prompt = format_mm_prompt((demo_q, demo_a), q_text)
    lang = bundle.tokenizer([prompt], return_tensors="pt", padding=True).to(DEVICE)
    out = bundle.model.generate(
        vision_x=vision_x, lang_x=lang["input_ids"], attention_mask=lang["attention_mask"],
        max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, repetition_penalty=REPETITION_PENALTY,
    )
    return decode_generation(bundle.tokenizer, out[0], q_text)

@torch.no_grad()
def infer_zero_shot(bundle: FlamingoBundle, q_image: Image.Image, q_text: str) -> str:
    vision_x = build_vision_batch_from_images(bundle.image_processor, [q_image])
    prompt = format_mm_prompt(None, q_text)
    lang = bundle.tokenizer([prompt], return_tensors="pt", padding=True).to(DEVICE)
    out = bundle.model.generate(
        vision_x=vision_x, lang_x=lang["input_ids"], attention_mask=lang["attention_mask"],
        max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, repetition_penalty=REPETITION_PENALTY,
    )
    return decode_generation(bundle.tokenizer, out[0], q_text)

@dataclass
class CandidateResult:
    idx: int; Pi: float; delta: float; label: str

def brute_force_labeling(bundle: FlamingoBundle, query_sample: Dict, candidate_pool: List[Dict], delta_threshold: float) -> Tuple[float, List[CandidateResult]]:
    pred0 = infer_zero_shot(bundle, query_sample["image"], query_sample["question"])
    P0 = compute_cider_score(pred0, query_sample["answers"])
    tqdm.write("-" * 50 + f"\n[DEBUG 0-shot] Q: {query_sample['question']}\n[DEBUG 0-shot] GTs: {query_sample['answers']}\n[DEBUG 0-shot] Pred: '{pred0}'\n[DEBUG 0-shot] Score P0: {P0:.4f}\n" + "-" * 50)
    
    results = []
    print_limit = 5
    for i, cand in enumerate(candidate_pool):
        if cand is None: continue
        demo_a = cand["answers"][0]
        pred_i = infer_one_shot(bundle, cand["image"], cand["question"], demo_a, query_sample["image"], query_sample["question"])
        Pi = compute_cider_score(pred_i, query_sample["answers"])
        delta = Pi - P0

        if i < print_limit:
            tqdm.write(f"\n--- [DEBUG 1-shot with cand #{cand['index']}] ---\n    Demo A: {demo_a}\n    Pred Pi: '{pred_i}'\n    Score Pi: {Pi:.4f}\n    Delta (Pi - P0): {delta:.4f}")

        label = "neutral"
        if delta >= delta_threshold: label = "pos"
        elif delta <= -delta_threshold: label = "neg"
        results.append(CandidateResult(idx=cand["index"], Pi=Pi, delta=delta, label=label))
    return P0, results

def build_pools(results: List[CandidateResult]) -> Tuple[List[int], List[int]]:
    return [r.idx for r in results if r.label == "pos"], [r.idx for r in results if r.label == "neg"]

def sample_triplets(query_idx: int, Dq_pos: List[int], Dq_neg: List[int], k: int) -> List[Tuple[int, int, int]]:
    if not Dq_pos or not Dq_neg: return []
    return [(query_idx, random.choice(Dq_pos), random.choice(Dq_neg)) for _ in range(k)]

# =========================
#     FAISS 检索函数
# =========================
def get_top_k_candidates_indices(q_sample, clip_model, clip_preprocess, faiss_index, k):
    image = clip_preprocess(q_sample['image']).unsqueeze(0).to(DEVICE)
    text = clip.tokenize([q_sample['question']], truncate=True).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_features = torch.cat([image_features, text_features], dim=-1).cpu().numpy().astype('float32')
    _, I = faiss_index.search(query_features, k)
    return I[0]

# =========================
#         主程序
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Efficient Triplet Generation with VLM Feedback")
    parser.add_argument('--task', type=str, required=True, choices=['vqa', 'imagenet'])
    parser.add_argument('--train_questions_path', type=str, required=True)
    parser.add_argument('--train_annotations_path', type=str, required=True)
    parser.add_argument('--test_questions_path', type=str, required=True)
    parser.add_argument('--test_annotations_path', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--train_split', type=str, default='train2014')
    parser.add_argument('--test_split', type=str, default='val2014')
    parser.add_argument('--output_dir', type=str, default="./triplet_output")
    parser.add_argument('--delta_threshold', type=float, default=0.1) # 稍微提高一点阈值，因为CIDEr分数波动可能更大
    parser.add_argument('--triplets_per_query', type=int, default=5)
    parser.add_argument('--top_k_retrieval', type=int, default=100)
    parser.add_argument('--faiss_index_path', type=str, default="./faiss_index/vqa_train.faiss")
    return parser.parse_args()

def main():
    args = parse_args()
    set_default_tensor_type()
    
    # --- Phase 0: 加载所有模型和数据 ---
    print("--- Phase 0: Loading Models and Data ---")
    bundle = load_openflamingo_9b()
    clip_model, clip_preprocess = clip.load("ViT-L-14", device=DEVICE)
    
    if args.task == "vqa":
        train_dataset = VQAv2Dataset(args.train_questions_path, args.train_annotations_path, args.img_root, args.train_split)
        test_dataset = VQAv2Dataset(args.test_questions_path, args.test_annotations_path, args.img_root, args.test_split)
    else: raise NotImplementedError("ImageNet not yet implemented.")

    faiss_index = faiss.read_index(args.faiss_index_path)
    print(f"Total candidates in train_dataset: {len(train_dataset)}")
    print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
    
    # --- Phase 1: 主循环 ---
    print("\n--- Phase 1: Starting TWO-STAGE Triplet Generation ---")
    os.makedirs(args.output_dir, exist_ok=True)
    details_dir = os.path.join(args.output_dir, "q_details")
    os.makedirs(details_dir, exist_ok=True)
    all_triplets = []
    
    for q_sample in tqdm(test_dataset, desc="Processing Queries"):
        if q_sample is None: continue
        
        retrieved_indices = get_top_k_candidates_indices(q_sample, clip_model, clip_preprocess, faiss_index, k=args.top_k_retrieval)
        candidate_pool = [train_dataset[idx] for idx in retrieved_indices]
        tqdm.write(f"Query #{q_sample['index']}: Retrieved {len(candidate_pool)} candidates for re-ranking.")
        
        P0, cand_results = brute_force_labeling(bundle, q_sample, candidate_pool, args.delta_threshold)
        Dq_pos, Dq_neg = build_pools(cand_results)
        
        triplets = []
        generation_successful = False
        if Dq_pos and Dq_neg:
            generation_successful = True
            triplets = sample_triplets(q_sample['index'], Dq_pos, Dq_neg, args.triplets_per_query)
            all_triplets.extend(triplets)
        else:
            tqdm.write(f"Warning: Query #{q_sample['index']} could not generate triplets (pos: {len(Dq_pos)}, neg: {len(Dq_neg)}).")

        with open(os.path.join(details_dir, f"q_{q_sample['index']}_details.json"), "w") as f:
            json.dump({
                "query_index": q_sample['index'], "generation_successful": generation_successful,
                "P0": P0, "num_pos": len(Dq_pos), "num_neg": len(Dq_neg),
                "generated_triplets": triplets, "results": [r.__dict__ for r in cand_results], 
            }, f, indent=2)

    # --- Phase 2: 保存结果 ---
    final_output_path = os.path.join(args.output_dir, "final_triplets.json")
    print(f"\nTriplet generation finished. Total triplets generated: {len(all_triplets)}")
    with open(final_output_path, "w") as f: json.dump(all_triplets, f, indent=2)
    print(f"Saving the complete triplet dataset to: {final_output_path}")
    print("Done.")

if __name__ == "__main__":
    main()