import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np # 用于计算平均分

class ZeroshotSampler:
    """
    一个用于计算数据集中每个样本的0-shot VQA Accuracy分数的采样器/评估器。
    [修改] 这个类现在支持批处理以提高评估速度。
    """

    def __init__(self, model, image_processor, tokenizer, device="cuda"):
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _run_inference(self, images, question_texts):
        """
        [修改] 对一批图像和问题执行0-shot推理。

        Args:
            images (torch.Tensor): 一个批次的图像张量，形状为 [B, C, H, W]。
            question_texts (List[str]): 一个包含 B 个问题文本的列表。

        Returns:
            List[str]: 一个包含 B 个模型生成答案的列表。
        """
        # 1. 构造一个批次的 Prompts
        prompts = [f"<image>Question: {q} Answer:" for q in question_texts]

        # 2. 图像预处理 (现在 images 已经是批处理好的)
        #    原始形状: [B, C, H, W]
        #    OpenFlamingo期望: [B, num_media, num_frames, C, H, W]
        #    所以我们需要在第1和第2维增加维度
        vision_x = images.to(self.device).unsqueeze(1).unsqueeze(2)

        # 3. 文本预处理 (现在处理的是一个 prompts 列表)
        self.tokenizer.padding_side = "left"
        #    [修改] 使用 padding=True 来处理不等长的问题
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # 4. 模型生成 (generate 方法原生支持批处理)
        generated_ids = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        
        # 5. 解码 (现在需要解码一个批次的结果)
        generated_ids = generated_ids[:, lang_x["input_ids"].shape[1]:]
        
        #    [修改] 使用列表推导式一次性解码所有生成的文本
        responses = [
            self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
        ]
        return [res.strip() for res in responses]

    def calculate_scores(self, dataset, batch_size=8): # [修改] 默认 bs=8
        """
        [修改] 遍历整个数据集，使用指定的 batch_size 计算 VQA Accuracy。

        Args:
            dataset (torch.utils.data.Dataset): 待评估的数据集。
            batch_size (int): 用于评估的批处理大小。

        Returns:
            dict: 一个字典，映射 question_id 到其对应的分数。
        """
        print(f"--- 开始计算 0-shot VQA Accuracy (batch_size={batch_size}) ---")
        # [修改] 使用传入的 batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_scores = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating VQA Scores"):
                # 1. 从数据加载器中解包一整个批次的数据
                images = batch['image']
                questions = batch['question']
                question_ids = batch['question_id']
                all_answers_batch = batch['all_answers']

                # 2. 获取整个批次的模型预测
                predictions = self._run_inference(images, questions)

                # 3. [修改] 遍历批次中的每一个结果来计算分数
                for i in range(len(predictions)):
                    # 提取当前样本的信息
                    prediction = predictions[i]
                    question_id = question_ids[i].item()
                    # 注意: all_answers_batch 的结构可能需要调整
                    # 我们假设它是一个列表的列表，所以用 [i] 来获取当前样本的答案列表
                    gt_answers = all_answers_batch[i] 

                    num_matches = 0
                    for ans in gt_answers:
                        # 正则匹配逻辑保持不变
                        escaped_ans = re.escape(ans)
                        pattern = r'\b' + escaped_ans + r'\b'
                        if re.search(pattern, prediction, re.IGNORECASE):
                            num_matches += 1
                    
                    score = min(num_matches / 3.0, 1.0)
                    all_scores[question_id] = score

        avg_score = np.mean(list(all_scores.values()))
        print(f"--- 分数计算完成！共处理了 {len(all_scores)} 个样本。 ---")
        print(f"--- 平均 VQA Accuracy: {avg_score:.4f} ---")
        return all_scores