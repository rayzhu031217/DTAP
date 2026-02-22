import os

# Milvus配置
MILVUS_HOST = 'localhost'
MILVUS_PORT = 19530
COLLECTION_NAME = 'dtap_value'  # 修改为 dtap_value
VECTOR_DIM = 768  # 与嵌入模型输出维度一致，例如mxbai-embed-large-v1为1024，这里用768示意

# 视频处理配置
VIDEO_FPS = 2               # 处理帧率（VILA输入帧率）
CLIP_DURATION = 3           # 每个片段时长（秒）
CLIP_FRAMES = VIDEO_FPS * CLIP_DURATION  # 每个片段帧数

# 价值函数权重
WEIGHT_SALIENCY = 0.3
WEIGHT_NEW_OBJECTS = 0.4
WEIGHT_EVENT = 0.3
VALUE_THRESHOLD = 0.5       # 存储阈值

# 物体检测配置（Grounding DINO等）
OBJECT_DETECTION_MODEL = "grounding-dino-base"  # 假设模型名称
OBJECT_CLASSES = ["person", "chair", "table", "door", "car", "box"]  # 感兴趣类别

# 嵌入模型（用于标题文本）
EMBED_MODEL = "nomic-embed-text"  # Ollama嵌入模型
OLLAMA_BASE_URL = "http://localhost:11434"

# 大模型配置
LLM_MODEL = "llava:7b"      # 或 "llama3.1:8b" 用于纯文本
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 200

# 查询阶段参数
RETRIEVAL_K = 20            # 初次检索条数
FINAL_K = 5                 # 重排序后保留条数
MAX_ITERATIONS = 3          # 最大检索迭代次数