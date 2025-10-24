import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import numpy as np

class OneShotSampler:
    """
    一个用于在数据集中为每个查询样本挖掘正/负one-shot示例的采样器。

    它的核心任务是：
    1. 对于数据集中的每一个“查询”样本 (q)。
    2. 遍历数据集中的所有其他样本作为“示例” (e)。
    3. 使用 (e, q) 对进行 one-shot 推理，得到一个 one-shot 分数。
    4. 将此分数与 q 的原始 0-shot 分数进行比较。
    5. 根据预设的 margin，将 e 归类为 q 的正样本或负样本。
    """

    def __init__(self, model, image_processor, tokenizer, device="cuda"):
        """
        初始化 OneShotSampler.

        Args:
            model: 已加载的OpenFlamingo模型。
            image_processor: 用于处理图像的预处理器。
            tokenizer: 用于处理文本的分词器。
            device (str): 运行模型的设备 ('cuda' or 'cpu')。
        """
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _calculate_vqa_accuracy(self, prediction, gt_answers):
        """
        使用正则表达式计算单个预测的VQA Accuracy。
        """
        num_matches = 0
        for ans in gt_answers:
            escaped_ans = re.escape(ans)
            pattern = r'\b' + escaped_ans + r'\b'
            if re.search(pattern, prediction, re.IGNORECASE):
                num_matches += 1
        return min(num_matches / 3.0, 1.0)

    def _run_one_shot_inference(self, example_image, example_answer, query_image, query_question):
        """
        对一个 example-query 对执行 one-shot 推理。

        Args:
            example_image (torch.Tensor): 示例图像张量。
            example_answer (str): 示例的标准答案。
            query_image (torch.Tensor): 查询图像张量。
            query_question (str): 查询问题文本。

        Returns:
            str: 模型生成的对查询问题的答案。
        """
        # 1. 构造 One-shot Prompt
        # 格式为: <image> 示例问答 <|endofchunk|> <image> 查询问题
        # 注意: VQA数据集中没有示例问题，我们可以简化或复用查询问题，
        # 但更常见的做法是直接给出答案。为了简单且有效，我们假设一个隐式问题。
        # 这里我们直接用 "Answer: <ans>" 作为示例。
        prompt = f"<image>Answer: {example_answer}<|endofchunk|><image>Question: {query_question} Answer:"

        # 2. 图像预处理
        # 将两个图像张量堆叠在一起，形成一个批次
        # 形状: [2, C, H, W]
        vision_x = torch.stack([example_image, query_image], dim=0)
        # 调整为模型期望的形状: [B, T_img, F, C, H, W]
        # B=1 (batch), T_img=2 (num_media), F=1 (num_frames)
        vision_x = vision_x.unsqueeze(0).unsqueeze(2).to(self.device)

        # 3. 文本预处理
        self.tokenizer.padding_side = "left"
        lang_x = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # 4. 模型生成
        generated_ids = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        
        # 5. 解码
        generated_ids = generated_ids[:, lang_x["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.strip()

    def find_samples(self, dataset: Dataset, zeroshot_scores: dict, margin: float = 0.1):
        """
        遍历数据集，为每个查询找到正样本和负样本池。

        Args:
            dataset (torch.utils.data.Dataset): 用于查询和示例的数据集。
                为了节省时间，强烈建议传入一个小的 Subset。
            zeroshot_scores (dict): 从 ZeroshotSampler 计算出的0-shot分数。
            margin (float): 用于判断样本好坏的分数阈值。

        Returns:
            dict: 一个字典，映射 query_id 到其对应的正负样本池。
                  例如: {
                      query_id_1: {'positive': [ex_id_A, ex_id_B], 'negative': [ex_id_C]},
                      query_id_2: {'positive': [], 'negative': [ex_id_D]}
                  }
        """
        print("--- 开始挖掘 One-shot 正负样本 ---")
        if len(dataset)**2 > 10000:
             print(f"警告: 这将进行 {len(dataset)**2} 次模型推理，可能会非常耗时。")
             print("建议使用 torch.utils.data.Subset 来减小数据集规模进行测试。")

        sample_pools = {}

        with torch.no_grad():
            # 外层循环：遍历作为“查询”的每个样本
            for q_idx in tqdm(range(len(dataset)), desc="Processing Queries"):
                query_data = dataset[q_idx]
                q_id = query_data['question_id']
                q_image = query_data['image']
                q_question = query_data['question']
                q_gt_answers = query_data['all_answers']
                
                # 获取基准分数
                base_score = zeroshot_scores.get(q_id)
                if base_score is None:
                    print(f"警告: 在zeroshot_scores中找不到 question_id {q_id}，跳过此查询。")
                    continue

                positive_samples = []
                negative_samples = []

                # 内层循环：遍历作为“示例”的每个样本
                for e_idx in tqdm(range(len(dataset)), desc=f"Finding samples for Q:{q_id}", leave=False):
                    if q_idx == e_idx:
                        continue # 一个样本不能作为自己的示例

                    example_data = dataset[e_idx]
                    e_id = example_data['question_id']
                    e_image = example_data['image']
                    # 我们需要一个高质量的答案作为示例，通常用 'gt_answer'
                    e_gt_answer = example_data['gt_answer']

                    # 执行 one-shot 推理
                    prediction = self._run_one_shot_inference(
                        example_image=e_image,
                        example_answer=e_gt_answer,
                        query_image=q_image,
                        query_question=q_question
                    )

                    # 计算 one-shot 分数
                    one_shot_score = self._calculate_vqa_accuracy(prediction, q_gt_answers)
                    
                    # 对比分数并分类
                    if one_shot_score > base_score + margin:
                        positive_samples.append(e_id)
                    elif one_shot_score < base_score - margin:
                        negative_samples.append(e_id)

                sample_pools[q_id] = {
                    'positive': positive_samples,
                    'negative': negative_samples
                }

        print(f"--- 样本挖掘完成！共处理了 {len(sample_pools)} 个查询。 ---")
        return sample_pools