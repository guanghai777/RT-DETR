import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import ROOT

import os
import shutil
import cv2
from lxml import etree
from tqdm import tqdm

# ========== 路径可在 project_paths.py 中统一修改 ==========
DATASET_ROOT = str(ROOT / 'datasets/insulator')
IMAGE_DIR = 'images'          # 里面包含 train 和 val 子目录
LABEL_DIR = 'labels'          # 里面包含 train 和 val 子目录（存放生成的txt）

# 类别映射（必须与 data.yaml 中的 names 顺序完全一致）
class_mapping = {
    'insulator': 0,
    '110_two_hight_glass': 1,
    'broken_disc': 2,
    'Glass_Dirty':3,
    'Glass_Loss': 4,
    'pollution-flashover': 5,
    'Polyme_Dirty': 6,
    # 如果你的数据集有 'insulator'，加在这里，编号继续 6
}

# ==========================================

def convert_one_xml(xml_path, img_path, txt_path):
    """将单个 XML 转换为 YOLO TXT"""
    # 解析 XML
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像尺寸（优先从 XML 读取，没有则从图片读取）
    size = root.find('size')
    if size is not None and size.find('width') is not None:
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
    else:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片：{img_path}")
            return False
        img_h, img_w = img.shape[:2]
    
    objects = root.findall('object')
    if not objects:
        return False  # 没有目标，不生成 txt
    
    lines = []
    for obj in objects:
        name = obj.find('name').text
        if name not in class_mapping:
            print(f"警告：类别 '{name}' 未在映射表中，跳过该目标（文件 {os.path.basename(xml_path)}）")
            continue
        class_id = class_mapping[name]
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 归一化
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    if not lines:
        return False
    
    # 写入 txt
    with open(txt_path, 'w') as f:
        f.writelines(lines)
    return True


def process_split(split_name):
    """处理 train 或 val 子集"""
    img_dir = os.path.join(DATASET_ROOT, IMAGE_DIR, split_name)
    xml_dir = os.path.join(DATASET_ROOT, LABEL_DIR, split_name)   # 原 XML 存放位置
    txt_dir = os.path.join(DATASET_ROOT, LABEL_DIR, split_name)   # TXT 输出到同一目录（覆盖原 XML？注意：我们同时保留 XML，但框架只认 TXT）

    if not os.path.exists(img_dir):
        print(f"跳过 {split_name}：图片目录不存在 {img_dir}")
        return

    # 获取所有图片文件（支持 .jpg .png）
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print(f"跳过 {split_name}：图片目录为空")
        return

    print(f"处理 {split_name} 集，共 {len(image_files)} 张图片...")
    success_count = 0
    for img_file in tqdm(image_files):
        base_name = os.path.splitext(img_file)[0]
        xml_file = base_name + '.xml'
        xml_path = os.path.join(xml_dir, xml_file)
        if not os.path.exists(xml_path):
            print(f"警告：找不到 XML 文件 {xml_path}，跳过")
            continue
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(txt_dir, base_name + '.txt')
        if convert_one_xml(xml_path, img_path, txt_path):
            success_count += 1
        else:
            # 转换失败（可能是空标注），生成一个空 txt 或跳过？这里不生成
            pass

    print(f"{split_name} 完成，成功生成 {success_count} 个 TXT 文件（共 {len(image_files)} 张图片）")


if __name__ == '__main__':
    # 分别处理 train 和 val
    process_split('train')
    process_split('val')
    process_split('test')
    print("\n所有转换完成！请检查 labels 目录下是否正确生成了 .txt 文件。")
    print("别忘了删除旧的 .cache 缓存文件，然后重新训练。")