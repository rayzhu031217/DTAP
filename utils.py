import cv2
import numpy as np
import requests
import json
import re
import os
import bisect
from PIL import Image
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility

# ========== 视频处理（已弃用，保留但不再使用）==========
def extract_video_clips(video_path, clip_duration, fps):
    """原函数不再使用，保留占位"""
    raise NotImplementedError("Use load_tum_sequence instead")

def get_position_at_time(log_path, target_time):
    """原函数不再使用，位置直接从groundtruth获取"""
    raise NotImplementedError("Use pose from groundtruth")

# ========== TUM RGB-D 序列加载 ==========
def load_tum_sequence(rgb_txt, depth_txt, groundtruth_txt=None,
                      rgb_root='rgb', depth_root='depth', depth_scale=0.001):
    """
    加载 TUM 格式 RGB-D 序列。
    参数：
        rgb_txt: rgb.txt 路径
        depth_txt: depth.txt 路径
        groundtruth_txt: groundtruth.txt 路径（可选）
        rgb_root: RGB 图像所在文件夹（相对于工作目录）
        depth_root: 深度图像所在文件夹
        depth_scale: 深度图缩放因子
    返回：
        frames: 列表，每个元素为字典，包含：
            - timestamp: 时间戳（float）
            - rgb: RGB 图像 (H,W,3)
            - depth: 深度图 (H,W) 单位为米
            - pose: 位姿元组 (tx, ty, tz, qx, qy, qz, qw) 或 None
    """
    # 读取 rgb.txt
    rgb_entries = []
    with open(rgb_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                t = float(parts[0])
                filename = parts[1]
                rgb_entries.append((t, os.path.join(rgb_root, filename)))

    # 读取 depth.txt
    depth_entries = []
    with open(depth_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                t = float(parts[0])
                filename = parts[1]
                depth_entries.append((t, os.path.join(depth_root, filename)))

    rgb_entries.sort(key=lambda x: x[0])
    depth_entries.sort(key=lambda x: x[0])

    # 读取 groundtruth 位姿
    pose_dict = {}
    if groundtruth_txt and os.path.exists(groundtruth_txt):
        with open(groundtruth_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 8:  # timestamp tx ty tz qx qy qz qw
                    t = float(parts[0])
                    pose = tuple(map(float, parts[1:8]))
                    pose_dict[t] = pose

    # 对齐 RGB 与深度，并匹配位姿
    frames = []
    depth_timestamps = [d[0] for d in depth_entries]
    depth_paths = [d[1] for d in depth_entries]

    for t_rgb, rgb_path in rgb_entries:
        # 在深度时间戳中找最近邻
        idx = bisect.bisect_left(depth_timestamps, t_rgb)
        best_idx = None
        if idx > 0 and idx < len(depth_timestamps):
            if abs(depth_timestamps[idx] - t_rgb) < abs(depth_timestamps[idx-1] - t_rgb):
                best_idx = idx
            else:
                best_idx = idx-1
        elif idx == 0 and len(depth_timestamps) > 0:
            best_idx = 0
        elif idx == len(depth_timestamps) and len(depth_timestamps) > 0:
            best_idx = len(depth_timestamps) - 1
        else:
            continue

        t_depth = depth_timestamps[best_idx]
        depth_path = depth_paths[best_idx]

        if abs(t_depth - t_rgb) > 0.05:  # 阈值 50ms
            continue

        # 加载图像
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Warning: cannot read {rgb_path}")
            continue

        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            try:
                depth_img = np.load(depth_path.replace('.png', '.npy'))
            except:
                print(f"Warning: cannot read depth {depth_path}")
                continue

        # 深度图转换为米
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) * depth_scale

        # 获取对应位姿
        pose = None
        if pose_dict:
            pose_timestamps = sorted(pose_dict.keys())
            idx_pose = bisect.bisect_left(pose_timestamps, t_rgb)
            best_pose_t = None
            if idx_pose > 0 and idx_pose < len(pose_timestamps):
                if abs(pose_timestamps[idx_pose] - t_rgb) < abs(pose_timestamps[idx_pose-1] - t_rgb):
                    best_pose_t = pose_timestamps[idx_pose]
                else:
                    best_pose_t = pose_timestamps[idx_pose-1]
            elif idx_pose == 0 and len(pose_timestamps) > 0:
                best_pose_t = pose_timestamps[0]
            elif idx_pose == len(pose_timestamps) and len(pose_timestamps) > 0:
                best_pose_t = pose_timestamps[-1]

            if best_pose_t is not None and abs(best_pose_t - t_rgb) <= 0.05:
                pose = pose_dict[best_pose_t]

        frames.append({
            'timestamp': t_rgb,
            'rgb': rgb_img,
            'depth': depth_img,
            'pose': pose
        })

    return frames

# ========== VLM 调用 ==========
def generate_caption(frames):
    """使用VLM生成视频片段的描述，frames为RGB图像列表"""
    import base64
    from io import BytesIO
    images_b64 = []
    for frame in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images_b64.append(img_b64)

    payload = {
        "model": "llava:7b",
        "prompt": "Describe what is happening in this sequence of images in one sentence.",
        "images": images_b64,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
    except Exception as e:
        print(f"VLM error: {e}")
    return ""

def has_event(caption):
    """判断描述是否包含重要事件（关键词匹配）"""
    keywords = ["fall", "drop", "enter", "leave", "open", "close", "collide", "stop", "move"]
    caption_lower = caption.lower()
    for kw in keywords:
        if kw in caption_lower:
            return True
    return False

# ========== 物体检测（带深度过滤）==========
def detect_objects(image, depth, classes, max_distance=5.0):
    """
    使用Grounding DINO等检测物体，并利用深度图滤除距离过远的物体。
    返回检测到的类别集合。
    """
    # 实际检测实现（这里模拟返回空集，但预留深度过滤逻辑）
    # 假设检测结果返回列表，每个元素为 (class_name, distance)
    detected = []  # 调用真实检测模型得到
    # 模拟：假设检测到一些物体
    # detected = [("chair", 2.0), ("table", 6.0), ("person", 1.5)]
    filtered = [obj for obj in detected if obj[1] <= max_distance]
    return set([obj[0] for obj in filtered])

# ========== 帧差异 ==========
def frame_difference(frame1, frame2):
    """计算两帧之间的均方误差"""
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff)

def depth_difference(depth1, depth2):
    """计算两帧深度图的平均绝对差异（忽略无效值）"""
    valid_mask = (depth1 > 0) & (depth2 > 0)
    if np.sum(valid_mask) == 0:
        return 0.0
    return np.mean(np.abs(depth2[valid_mask] - depth1[valid_mask]))

# ========== 嵌入 ==========
def embed_text(text, model="nomic-embed-text"):
    """通过Ollama获取文本嵌入向量"""
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json=payload, timeout=10)
        if response.status_code == 200:
            return np.array(response.json()["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
    return np.zeros(config.VECTOR_DIM)

# ========== LLM 调用 ==========
def call_llm(messages, model, temperature=0.2, max_tokens=200):
    """调用Ollama生成文本"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["message"]["content"]
    except Exception as e:
        print(f"LLM error: {e}")
    return ""

def parse_function_call(response):
    """解析LLM响应中的函数调用（JSON格式）"""
    json_pattern = r'\{.*"name".*"arguments".*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            func_call = json.loads(match.group())
            return func_call
        except:
            pass
    return None

# ========== Milvus 初始化 ==========
def init_milvus_collection(collection_name, dim):
    """创建或获取集合，并创建索引"""
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name='caption', dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name='position_x', dtype=DataType.FLOAT),
        FieldSchema(name='position_y', dtype=DataType.FLOAT),
        FieldSchema(name='timestamp', dtype=DataType.FLOAT),
        FieldSchema(name='value', dtype=DataType.FLOAT)
    ]
    schema = CollectionSchema(fields)
    collection = Collection(collection_name, schema)
    index_params = {
        'metric_type': 'IP',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 128}
    }
    collection.create_index('vector', index_params)
    return collection