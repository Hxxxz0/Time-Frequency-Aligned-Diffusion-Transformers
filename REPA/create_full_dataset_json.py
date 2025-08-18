import json
import os
import glob

# 获取所有图像文件
image_dir = "full_dataset/images"
image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

# 创建标签字典
labels = []
for idx, img_path in enumerate(image_files):
    # 获取文件名
    filename = os.path.basename(img_path)
    # 为每个图像分配一个假标签（0-999之间）
    label = idx % 1000
    labels.append([filename, label])

# 保存到dataset.json
dataset_dict = {"labels": labels}
output_path = "full_dataset/images/dataset.json"
with open(output_path, 'w') as f:
    json.dump(dataset_dict, f, indent=2)

print(f"Created dataset.json with {len(labels)} entries") 