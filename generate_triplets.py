import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from transformers import CLIPProcessor, CLIPModel
import random
import requests


# 假设你已经将OpenFlamingo加载到`model`、`image_processor` 和 `tokenizer` 中
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

# 加载图像处理器和tokenizer
# 你也可以直接用open_flamingo的预处理方法
vision_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 1. 执行零-shot推理
def perform_zero_shot_inference(model, image_processor, tokenizer, query_image, query_question):
    """
    执行零-shot推理。
    :param model: 已加载的OpenFlamingo模型
    :param image_processor: 图像处理器
    :param tokenizer: 文本处理器（tokenizer）
    :param query_image: 查询样本的图像
    :param query_question: 查询样本的问题
    :return: 模型预测的答案
    """
    # 预处理图像
    inputs = image_processor(query_image, return_tensors="pt")  # 获取字典，包含'pixel_values'
    
    vision_x = inputs['pixel_values']  # 从字典中提取图像张量
    # 确保 vision_x 的形状是 (b, T_img, F, C, H, W)
    vision_x = vision_x.unsqueeze(1).unsqueeze(2)  # 添加 T_img 和 F 的维度
    vision_x = vision_x.permute(0, 1, 2, 3, 4, 5)  # 确保顺序正确 (b, T_img, F, C, H, W)

    # 预处理问题
    lang_x = tokenizer(query_question, return_tensors="pt")  # 处理问题为tokenizer输入格式

    # 执行推理，生成模型预测
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,  # 控制输出长度
        num_beams=3,  # 使用beam search增加预测的多样性
    )

    # 解码生成的文本并返回
    generated_answer = tokenizer.decode(generated_text[0])
    return generated_answer

# 2. 计算VQA准确度（P0）
def calculate_vqa_accuracy(predicted_answer, ground_truth_answers):
    """
    计算VQA任务的准确度（准确匹配预测答案和真实答案）。
    :param predicted_answer: 模型预测的答案
    :param ground_truth_answers: 查询样本的真实答案
    :return: 准确度（0.0或1.0）
    """
    # 判断预测答案是否在真实答案列表中
    return 1.0 if predicted_answer in ground_truth_answers else 0.0

# 3. 主函数：加载查询样本并计算零-shot性能
if __name__ == "__main__":
    # 示例查询样本
    query_image_url = "http://images.cocodataset.org/test-stuff2017/000000028137.jpg"
    query_question = "What is in the image?"
    ground_truth_answers = ["a cat", "cat", "a black and white cat"]

    # 加载查询样本图像
    query_image = Image.open(requests.get(query_image_url, stream=True).raw)

    # 4. 执行零-shot推理
    predicted_answer = perform_zero_shot_inference(model, image_processor, tokenizer, query_image, query_question)
    print(f"模型预测答案: {predicted_answer}")

    # 5. 计算零-shot性能（P0）
    P0 = calculate_vqa_accuracy(predicted_answer, ground_truth_answers)
    print(f"零-shot性能（P0）: {P0}")
