# 导入必要的库
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

def load_openflamingo_9b():
    """
    加载预训练好的 OpenFlamingo-9B 模型。
    """
    print("正在初始化模型结构和预处理器...")
    
    # create_model_and_transforms 会创建模型的“骨架”以及图片、文本的“预处理器”
    model, image_processor, tokenizer = create_model_and_transforms(
        # 1. 视觉编码器 (模型的眼睛)，使用OpenCLIP的大模型
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        
        # 2. 语言模型 (模型的大脑)，使用MPT-7B
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        
        # 3. 交叉注意力层 (连接眼睛和大脑的神经)
        # 对于9B模型，官方建议每隔4层语言模型层插入一个交叉注意力层
        cross_attn_every_n_layers=4
    )

    # 打印模型结构，确认模型已创建
    # print(model)

    print("模型结构创建完毕。正在从Hugging Face Hub下载预训练权重...")

    # hf_hub_download 会自动从网上下载模型的“灵魂”（已训练好的参数）
    # "openflamingo/OpenFlamingo-9B-vitl-mpt7b" 是9B模型在Hugging Face上的官方名称
    checkpoint_path = hf_hub_download(
        "openflamingo/OpenFlamingo-9B-vitl-mpt7b", 
        "checkpoint.pt"
    )
    
    # 加载权重文件
    model_weights = torch.load(checkpoint_path, map_location="cpu")
    
    # 将“灵魂”注入“骨架”
    model.load_state_dict(model_weights, strict=False)
    
    print("OpenFlamingo-9B 模型加载成功！")
    
    return model, image_processor, tokenizer

# --- 如果直接运行这个文件，就执行加载函数 ---
if __name__ == "__main__":
    # 这一步会花一些时间，因为它需要下载几个GB的模型文件
    model, image_processor, tokenizer = load_openflamingo_9b()
    
    # 你可以把模型移动到GPU上（如果可用）
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)


# def load_openflamingo_9b():
#     """
#     加载预训练好的 OpenFlamingo-9B 模型。
#     """
#     print("正在初始化模型结构和预处理器...")

#     # create_model_and_transforms 会创建模型的“骨架”以及图片、文本的“预处理器”
#     model, image_processor, tokenizer = create_model_and_transforms(
#         clip_vision_encoder_path="ViT-L-14",
#         clip_vision_encoder_pretrained="openai",
#         lang_encoder_path="anas-awadalla/mpt-7b",
#         tokenizer_path="anas-awadalla/mpt-7b",
#         cross_attn_every_n_layers=4
#     )

#     print("模型结构创建完毕，正在下载模型权重...")

#     checkpoint_path = hf_hub_download(
#         "openflamingo/OpenFlamingo-9B-vitl-mpt7b", 
#         "checkpoint.pt"
#     )

#     # 打印下载的权重路径，确认是否正确
#     print(f"下载的模型权重路径：{checkpoint_path}")

#     # 加载权重文件
#     model_weights = torch.load(checkpoint_path, map_location="cpu")

#     # 将“灵魂”注入“骨架”
#     model.load_state_dict(model_weights, strict=False)

#     print("OpenFlamingo-9B 模型加载成功！")

#     # 打印模型结构
#     print(f"模型结构：{model}")

#     return model, image_processor, tokenizer
