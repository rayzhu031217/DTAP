import cv2
import numpy as np
import requests
import json
import re
import os
import bisect
from PIL import Image
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility
import config

# -------------------------- 可选依赖导入与回退 --------------------------

# Grounding DINO 相关
try:
    import torch
    from groundingdino.util.inference import load_model, predict
    _GROUNDING_DINO_AVAILABLE = True
except ImportError:
    _GROUNDING_DINO_AVAILABLE = False
    print("Warning: Grounding DINO not installed. Object detection will be disabled.")

# Habitat 相关
try:
    import habitat_sim
    from habitat_sim.utils import common as habitat_utils
    _HABITAT_AVAILABLE = True
except ImportError:
    _HABITAT_AVAILABLE = False
    print("Warning: Habitat-sim not installed. R2R data loading will be disabled.")

# -------------------------- 通用辅助函数 --------------------------

def frame_difference(frame1, frame2):
    """计算两帧 RGB 的均方误差"""
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff)

def depth_difference(depth1, depth2):
    """计算两帧深度图的平均绝对差异（忽略无效值）"""
    valid_mask = (depth1 > 0) & (depth2 > 0)
    if np.sum(valid_mask) == 0:
        return 0.0
    return np.mean(np.abs(depth2[valid_mask] - depth1[valid_mask]))

def has_event(caption):
    """判断描述是否包含重要事件（关键词匹配）"""
    keywords = ["fall", "drop", "enter", "leave", "open", "close", "collide", "stop", "move"]
    caption_lower = caption.lower()
    return any(kw in caption_lower for kw in keywords)

# -------------------------- VLM / LLM / 嵌入 --------------------------

def generate_caption(frames):
    """使用 VLM（Ollama LLaVA）生成视频片段的描述，frames为RGB图像列表"""
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

def embed_text(text, model="nomic-embed-text"):
    """通过 Ollama 获取文本嵌入向量"""
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

def call_llm(messages, model, temperature=0.2, max_tokens=200):
    """调用 Ollama 生成文本（用于记忆查询）"""
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
    """解析 LLM 响应中的函数调用（JSON格式）"""
    json_pattern = r'\{.*"name".*"arguments".*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

# -------------------------- 物体检测（Grounding DINO + 深度过滤）--------------------------

# 全局缓存模型
_grounding_model = None
_grounding_device = None

def _load_grounding_dino():
    """延迟加载 Grounding DINO 模型"""
    global _grounding_model, _grounding_device
    if _grounding_model is not None:
        return _grounding_model, _grounding_device
    if not _GROUNDING_DINO_AVAILABLE:
        return None, None

    # 配置模型路径（可修改为你的本地路径）
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint = "groundingdino_swint_ogc.pth"

    if not os.path.exists(config_file) or not os.path.exists(checkpoint):
        print("Grounding DINO model files not found. Please download them.")
        return None, None

    _grounding_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _grounding_model = load_model(config_file, checkpoint, device=_grounding_device)
    return _grounding_model, _grounding_device

def detect_objects(image, depth, classes, max_distance=5.0):
    """
    使用 Grounding DINO 检测物体，并用深度图过滤超出距离的物体。
    参数：
        image: RGB 图像 (H,W,3) BGR 格式（OpenCV）
        depth: 深度图 (H,W) 单位：米
        classes: 感兴趣类别列表，如 ['person', 'chair']
        max_distance: 最大保留距离（米）
    返回：
        检测到的物体类别集合（字符串）
    """
    if not _GROUNDING_DINO_AVAILABLE:
        return set()

    model, device = _load_grounding_dino()
    if model is None:
        return set()

    # 构建文本提示：类别用点号分隔（Grounding DINO 常用格式）
    text_prompt = ". ".join(classes) + "."

    # 转换为 PIL RGB
    rgb_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 预测
    boxes, logits, phrases = predict(
        model=model,
        image=rgb_pil,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device=device
    )

    detected_classes = set()
    h_img, w_img = image.shape[:2]

    for box, score, phrase in zip(boxes, logits, phrases):
        cx, cy, w, h = box  # 归一化坐标 (x_center, y_center, width, height)
        # 转换到像素坐标
        x1 = int((cx - w/2) * w_img)
        y1 = int((cy - h/2) * h_img)
        x2 = int((cx + w/2) * w_img)
        y2 = int((cy + h/2) * h_img)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        depth_region = depth[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]
        if len(valid_depths) == 0:
            # 没有有效深度，默认保留
            detected_classes.add(phrase)
            continue

        obj_distance = np.median(valid_depths)
        if obj_distance <= max_distance:
            detected_classes.add(phrase)

    return detected_classes

# -------------------------- TUM RGB-D 序列加载 --------------------------

def load_tum_sequence(rgb_txt, depth_txt, groundtruth_txt=None,
                      rgb_root='rgb', depth_root='depth', depth_scale=0.001):
    """
    加载 TUM 格式 RGB-D 序列。
    返回 frames 列表，每个元素为字典，包含：
        timestamp, rgb (BGR), depth (float32, 单位米), pose (7 元组或 None)
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
                if len(parts) >= 8:
                    t = float(parts[0])
                    pose = tuple(map(float, parts[1:8]))
                    pose_dict[t] = pose

    frames = []
    depth_timestamps = [d[0] for d in depth_entries]
    depth_paths = [d[1] for d in depth_entries]

    for t_rgb, rgb_path in rgb_entries:
        # 找最近邻深度
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

        if abs(t_depth - t_rgb) > 0.05:   # 50ms 阈值
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

        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) * depth_scale

        # 获取位姿
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

# -------------------------- R2R 数据加载器（基于 Habitat）--------------------------

class R2RDataLoader:
    """
    加载 R2R 数据集，并通过 Habitat 模拟器获取 RGB-D 图像和位姿。
    需要安装 habitat-sim 并准备好 Matterport3D 数据。
    """
    def __init__(self, r2r_json_path, mp3d_data_dir, scene_list=None,
                 width=640, height=480, hfov=90):
        if not _HABITAT_AVAILABLE:
            raise ImportError("Habitat-sim is required for R2R data loading.")

        self.r2r_json_path = r2r_json_path
        self.mp3d_data_dir = mp3d_data_dir
        self.scene_list = scene_list
        self.width = width
        self.height = height
        self.hfov = hfov

        # 读取 R2R JSON
        with open(r2r_json_path, 'r') as f:
            self.data = json.load(f)

        # 按场景分组
        self.paths_by_scene = {}
        for item in self.data:
            scene_id = item['scene_id']
            if scene_list is None or scene_id in scene_list:
                self.paths_by_scene.setdefault(scene_id, []).append(item)

    def __iter__(self):
        """迭代所有路径，每条路径返回一个帧序列（与 load_tum_sequence 格式一致）"""
        for scene_id, paths in self.paths_by_scene.items():
            sim = self._init_simulator(scene_id)
            for path_item in paths:
                frames = self._traverse_path(sim, path_item)
                if frames:
                    yield frames
            sim.close()

    def _init_simulator(self, scene_id):
        """为指定场景创建 Habitat 模拟器实例"""
        scene_file = f"{self.mp3d_data_dir}/{scene_id}/{scene_id}.glb"
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_file

        # 定义传感器
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.height, self.width]
        rgb_spec.position = [0, 1.5, 0]
        rgb_spec.orientation = [0, 0, 0]
        rgb_spec.hfov = self.hfov

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.height, self.width]
        depth_spec.position = [0, 1.5, 0]
        depth_spec.orientation = [0, 0, 0]
        depth_spec.hfov = self.hfov

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.action_space = {}  # 不需要动作，直接设置位姿

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        return habitat_sim.Simulator(sim_cfg)

    def _traverse_path(self, sim, path_item):
        """
        在模拟器中按路径行走，收集每一帧的 RGB-D 和位姿。
        假设 path_item 中包含 'path' 字段，每个节点为 {'pose': [tx,ty,tz,qx,qy,qz,qw]}
        若节点只有 viewpoint ID，则需要先通过 connectivity 映射到位姿，此处简化。
        """
        frames = []
        path = path_item.get('path', [])
        if not path:
            # 尝试获取其他可能字段（如 'trajectory'）
            path = path_item.get('trajectory', [])

        for idx, node in enumerate(path):
            # 提取位姿
            if 'pose' in node:
                pose = node['pose']
            elif 'position' in node and 'rotation' in node:
                # 可能格式不同
                pos = node['position']
                rot = node['rotation']
                pose = [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]]
            else:
                continue

            # 设置 agent 状态
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([pose[0], pose[1], pose[2]])
            agent_state.rotation = habitat_utils.quat_from_coeff(pose[3:7])  # [x,y,z,w]
            sim.agents[0].set_state(agent_state)

            # 获取观测
            obs = sim.get_sensor_observations()
            rgb = obs['rgb']          # (H,W,3) 0-255
            depth = obs['depth']      # (H,W) 单位米

            frames.append({
                'timestamp': idx,      # 用索引作为时间（实际可改为时间戳）
                'rgb': rgb,
                'depth': depth,
                'pose': pose
            })

        return frames

# -------------------------- Milvus 初始化 --------------------------

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
