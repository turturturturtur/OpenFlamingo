import json
import os

# --- 配置区 ---
# 将所有路径都定义在这里，方便修改
DATA_DIR = '/data/public/datasets/vqav2'
OUTPUT_DIR = '/data/public/datasets/vqav2' # 我们将迷你文件也生成在数据集目录里
NUM_ITEMS = 10 # 您想抽取的样本数量

def shrink_json(original_path, output_path, key_name, num_items):
    print(f"Reading from {original_path}...")
    try:
        with open(original_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"!!! ERROR: File not found at {original_path}")
        print("!!! Please check if the DATA_DIR path is correct.")
        return # 提前退出函数

    print(f"Original number of items in '{key_name}': {len(data[key_name])}")
    
    # 复制原始数据结构，但用切片替换核心列表
    shrunk_data = data.copy()
    shrunk_data[key_name] = data[key_name][:num_items]
    
    print(f"Shrinking to {len(shrunk_data[key_name])} items.")
    
    with open(output_path, 'w') as f:
        json.dump(shrunk_data, f, indent=2)
    print(f"Successfully created mini file at: {output_path}\n")

# --- VQA Validation Set ---
# 问题文件
original_questions_path = os.path.join(DATA_DIR, 'v2_OpenEnded_mscoco_val2014_questions.json')
mini_questions_path = os.path.join(OUTPUT_DIR, 'mini_v2_OpenEnded_mscoco_val2014_questions.json')
shrink_json(
    original_path=original_questions_path,
    output_path=mini_questions_path,
    key_name='questions',
    num_items=NUM_ITEMS
)

# 标注文件
original_annotations_path = os.path.join(DATA_DIR, 'v2_mscoco_val2014_annotations.json')
mini_annotations_path = os.path.join(OUTPUT_DIR, 'mini_v2_mscoco_val2014_annotations.json')
shrink_json(
    original_path=original_annotations_path,
    output_path=mini_annotations_path,
    key_name='annotations',
    num_items=NUM_ITEMS
)