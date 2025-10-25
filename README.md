<div align="center">

# In-Context Learning Retriever for VQAv2

</div>

## 项目简介

本项目实现了一条完整的「0-shot → 1-shot」性能提升流水线：

1. **OpenFlamingo 评估**：加载官方开放权重，在 VQAv2 上计算基线 0-shot 准确率。
2. **One-shot 样本挖掘**：遍历训练集，借助真实答案构造 one-shot prompt，找到能显著拉升准确率的正例/负例池。
3. **特征提取器训练**：使用挖掘出的 (query, positive, negative) 三元组，通过三元组损失训练专用的多模态检索模型。
4. **Top-K 检索与复评**：在验证集上用训练好的 extractor 检索最有价值的示例，再次调用 OpenFlamingo 验证 one-shot 增益。

借助注册表与配置文件，本项目可以方便地替换数据集、模型或损失函数，快速迭代 In-Context Learning (ICL) 方案。

## 目录结构

- `pipeline.py`：主入口，串联配置读取、模型加载、采样器和训练流程。
- `config/`：实验超参、模型和数据集的 YAML 配置。
- `dataset/`：VQAv2 构建器与 `torch.utils.data.Dataset` 实现，负责缓存样本元数据并注入图像处理器。
- `sampler/`：`ZeroshotSampler`、`OneShotSampler` 与 `TopKSampler`，覆盖评分、挖掘和检索。
- `model/`：可插拔的特征提取器（`Extractor`），通过注册表动态创建。
- `loss/`：当前使用的三元组相似度损失，可按需扩展。
- `utils/`：配置读取、随机种子、工厂注册、训练循环和 Triplet 数据集包装。
- `data/`：期望存放 VQAv2 与 MSCOCO 原始文件的目录（示例数据未随仓库分发）。

## 环境准备

1. **创建 Conda 环境**

   ```bash
   conda env create -f environment.yaml
   conda activate openflamingo
   ```

2. **准备额外依赖**

   - 若首次使用 Hugging Face 权重，请执行 `huggingface-cli login` 以访问 `anas-awadalla/mpt-1b-redpajama-200b`。
   - 确保具备至少 24GB 显存的 NVIDIA GPU；若仅有 CPU，请在运行脚本时期待大幅降低的速度。
   - 可选：通过 `pip install -e .` 的方式将本项目注册为包，方便多处导入。

## 数据准备

默认配置期望以下目录结构（均位于仓库根目录下的 `data/`）：

```
data/
├── vqav2
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_train2014_annotations.json
│   └── v2_mscoco_val2014_annotations.json
└── coco
    ├── train2014/COCO_train2014_000000xxxxxx.jpg
    └── val2014/COCO_val2014_000000xxxxxx.jpg
```

建议步骤：

1. 从 [VQAv2 官方页面](https://visualqa.org/download.html) 下载题目与标注文件，放入 `data/vqav2/`。
2. 下载 COCO 2014 的 `train2014` 与 `val2014` 图像集，放入 `data/coco/`。
3. 若磁盘上已有数据，可通过软链接的方式接入，例如：

   ```bash
   ln -s /path/to/vqa data/vqav2
   ln -s /path/to/coco2014 data/coco
   ```

首次构建数据集会在 `data/checkpoint/` 下缓存过滤后的样本列表，后续运行会自动复用。

## 配置说明

- `config/experiment/main.yaml`：实验级超参（数据集别名、margin、loss 名称、学习率、batch size、epochs 等）。
- `config/model/extractor.yaml`：Extractor 的隐藏维度、头数及输出维度。
- `config/dataset/vqav2.yaml`：数据路径与文件名，可在此切换不同数据分割或目录。

你可以通过新增配置文件并在运行脚本时替换 `--exp_config` 与 `--model_config` 参数来管理多组实验。

## 启动实验

1. 激活环境并确认 GPU 可见（`nvidia-smi`）。
2. 运行主脚本：

   ```bash
   python pipeline.py \
     --exp_config config/experiment/main.yaml \
     --model_config config/model/extractor.yaml \
     --seed 42
   ```

脚本将依次执行：

1. 载入 OpenFlamingo 与 tokenizer，设置随机种子。
2. 根据 `config/dataset/vqav2.yaml` 构建训练、验证集。
3. 计算训练集的 0-shot 分数并打印平均值。
4. 针对每个查询挖掘 positive/negative 示例池，并基于它们动态构造 TripletDataset。
5. 使用三元组损失训练特征提取器（默认 AdamW）。
6. 调用 `TopKSampler` 为验证集检索 top-1 示例，并重新评估 one-shot 准确率。
7. 输出 0-shot、one-shot 与增益指标。

### 可选参数与技巧

- **子集调试**：VQAv2 全量挖掘较慢，可在 `pipeline.py` 中将 `dataset.build` 替换为 `Subset` 以快速验证流程。
- **调节 margin**：`main.yaml` 中的 `margin` 控制正负样本阈值，值越大越严格，但 TripletDataset 样本数会减少。
- **缓存管理**：若更换数据或问题文件，记得删除 `data/checkpoint/vqav2_*.pt` 以重新生成缓存。
- **日志 / 监控**：可将训练循环接入自定义日志器或 Weights & Biases，只需在 `FeatureExtractorTrainer` 中加入记录逻辑。

## 故障排查

- **加载 Hugging Face 权重失败**：确认已登录并具备网络访问权，同时检查代理/防火墙。
- **CUDA 内存不足**：降低 `batch_size`、缩小数据子集或启用混合精度（需自行在代码中添加 `torch.cuda.amp`）。
- **没有有效的 Triplet 样本**：说明挖掘阶段的 margin 过大或 0-shot 分数缺失，可调小 margin 或检查缓存是否齐全。
- **OpenCLIP 权重下载缓慢**：提前使用 `python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14')"` 进行预热下载。


