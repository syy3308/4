import json
import os
import cv2
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def find_images_directory(dataset_root):
    """
    在 dataset_root 中查找图像存放目录，优先查找 "Images"，其次 "images"、"images_test"。
    """
    possible_dirs = ["Images", "images", "images_test"]
    for d in possible_dirs:
        candidate = os.path.join(dataset_root, d)
        if os.path.exists(candidate):
            logging.info(f"Found images directory: {candidate}")
            return candidate
    logging.error("未找到图像目录，请检查 dataset_root 结构。")
    return None

def load_annotations(annotation_file):
    """
    加载标注文件。
    若文件以 '[' 开头，则当作 JSON 数组加载；
    否则每行解析为一个 JSON 对象（JSON Lines 格式）。
    """
    annotations = []
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                annotations = json.load(f)
            else:
                for idx, line in enumerate(f):
                    try:
                        anno = json.loads(line.strip())
                        annotations.append(anno)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON 解析错误，第 {idx+1} 行: {e}")
    except Exception as e:
        logging.error(f"加载标注文件 {annotation_file} 失败: {e}")
    return annotations

def is_valid_media_path(media_path):
    """
    判断媒体路径是否有效：长度大于3且以常见图片扩展名结尾。
    """
    if not media_path or len(media_path) < 3:
        return False
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in valid_exts:
        if media_path.lower().endswith(ext):
            return True
    return False

def load_image_cv2(image_path):
    """
    尝试使用 OpenCV 加载图像。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread 返回 None")
        return img
    except Exception as e:
        logging.error(f"使用 cv2.imread 加载图像 {image_path} 时出错: {e}")
        return None

def load_image_pillow(image_path):
    """
    当 OpenCV 加载失败时，尝试使用 Pillow 加载图像。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = np.array(img)
        return img
    except Exception as e:
        logging.error(f"使用 Pillow 加载图像 {image_path} 时出错: {e}")
        return None

def preprocess_data(dsdl_annotation_root, dataset_root):
    """
    数据预处理函数：
      - 从 dsdl_annotation_root 下的 set-train 目录加载标注文件（假设文件名为 train_samples.json）
      - 在 dataset_root 中查找图像目录（使用 Images 文件夹）
      - 从标注数据中读取 "img_file"（优先）或 fallback 到 "media" 或 "ID" 字段构造图像文件名
      - 加载图像，并返回 (图像, 检测框列表) 数据对
    """
    annotation_file = os.path.join(dsdl_annotation_root, "set-train", "train_samples.json")
    if not os.path.exists(annotation_file):
        logging.error(f"标注文件不存在: {annotation_file}")
        return []

    annotations = load_annotations(annotation_file)
    if not annotations:
        logging.error("未加载到任何标注数据！")
        return []

    images_dir = find_images_directory(dataset_root)
    if not images_dir:
        return []

    data = []
    for sample in annotations:
        # 优先使用 "img_file" 字段构造图像文件名；如果没有，再尝试 "media" 或 "ID" 字段
        img_filename = sample.get("img_file", "").strip()
        if not img_filename:
            if "media" in sample:
                img_filename = sample.get("media", {}).get("media_path", "").strip()
            elif "ID" in sample:
                img_filename = sample.get("ID", "").strip() + ".jpg"
            else:
                logging.warning(f"标注中缺少构造图像路径的字段: {sample}")
                continue

        if not is_valid_media_path(img_filename):
            logging.warning(f"无效的图像路径: {img_filename}")
            continue

        full_image_path = os.path.join(images_dir, img_filename)
        logging.info(f"加载图像: {full_image_path}")
        if not os.path.exists(full_image_path):
            logging.warning(f"图像不存在: {full_image_path}")
            continue

        img = load_image_cv2(full_image_path)
        if img is None:
            logging.info("尝试使用 Pillow 加载图像")
            img = load_image_pillow(full_image_path)
        if img is None:
            logging.warning(f"无法加载图像: {full_image_path}")
            continue

        gtboxes = sample.get("gtboxes", [])
        data.append((img, gtboxes))
    logging.info(f"预处理了 {len(data)} 个样本。")
    return data

if __name__ == "__main__":
    # 根据实际项目调整 dsdl_annotation_root 和 dataset_root 的路径
    dsdl_annotation_root = r"D:\ProgramData\PyCharm Community Edition 2024.3.5\PycharmProjects\PythonProject2\OpenDataLab___CrowdHuman\dsdl\dsdl_root"
    dataset_root = r"D:\ProgramData\PyCharm Community Edition 2024.3.5\PycharmProjects\PythonProject2\OpenDataLab___CrowdHuman\dsdl\dataset_root"
    train_data = preprocess_data(dsdl_annotation_root, dataset_root)