import time
import cv2
from pymilvus import Collection, utility, connections
from value_scorer import ValueScorer
from utils import (
    load_tum_sequence, generate_caption, embed_text,
    init_milvus_collection
)
import config

class MemoryBuilder:
    def __init__(self):
        connections.connect(alias='default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = init_milvus_collection(config.COLLECTION_NAME, config.VECTOR_DIM)
        self.scorer = ValueScorer(config)

    def build(self, rgb_txt, depth_txt, groundtruth_txt=None):
        """
        从 TUM 格式 RGB-D 序列构建记忆
        rgb_txt: rgb.txt 文件路径
        depth_txt: depth.txt 文件路径
        groundtruth_txt: groundtruth.txt 文件路径（可选）
        """
        # 加载所有帧
        all_frames = load_tum_sequence(
            rgb_txt, depth_txt, groundtruth_txt,
            rgb_root='rgb', depth_root='depth',
            depth_scale=config.DEPTH_SCALE
        )
        if not all_frames:
            print("No frames loaded.")
            return

        # 按固定时间跨度划分片段
        duration = config.CLIP_DURATION
        clips = []
        i = 0
        while i < len(all_frames):
            start_time = all_frames[i]['timestamp']
            end_time = start_time + duration
            clip_frames = []
            while i < len(all_frames) and all_frames[i]['timestamp'] <= end_time:
                clip_frames.append(all_frames[i])
                i += 1
            if len(clip_frames) >= config.MIN_CLIP_FRAMES:
                clips.append((clip_frames, start_time))
            # 如果希望片段之间有重叠，可调整 i 的步进（这里为连续不重叠）

        # 处理每个片段
        batch = []  # 用于批量插入
        for clip_frames, start_time in clips:
            # 提取 RGB 列表用于 VLM
            rgb_list = [f['rgb'] for f in clip_frames]
            caption = generate_caption(rgb_list)

            # 获取位置（取中间帧的位姿）
            mid_idx = len(clip_frames) // 2
            pose = clip_frames[mid_idx].get('pose')
            if pose is None:
                pos = (0.0, 0.0)
            else:
                # 取 tx, ty 作为平面位置
                pos = (pose[0], pose[1])

            # 计算价值
            value = self.scorer.compute(clip_frames, caption, start_time)

            if value >= config.VALUE_THRESHOLD:
                emb = embed_text(caption, model=config.EMBED_MODEL)
                metadata = {
                    'caption': caption,
                    'position_x': pos[0],
                    'position_y': pos[1],
                    'timestamp': start_time,
                    'value': value
                }
                batch.append((emb, metadata))
                print(f"Stored memory at {start_time:.3f}s, value={value:.2f}: {caption[:50]}...")
                # 批量插入，每10条一次
                if len(batch) >= 10:
                    self._batch_insert(batch)
                    batch = []
            else:
                print(f"Skipped memory at {start_time:.3f}s, value={value:.2f}")

        # 插入剩余批次
        if batch:
            self._batch_insert(batch)

        self.collection.flush()
        print("Memory building completed.")

    def _batch_insert(self, batch):
        """批量插入记忆到 Milvus"""
        ids = []
        vectors = []
        captions = []
        pos_x = []
        pos_y = []
        timestamps = []
        values = []
        import random
        for emb, meta in batch:
            entity_id = int(time.time() * 1000) + random.randint(0, 1000)
            ids.append(entity_id)
            vectors.append(emb.tolist())
            captions.append(meta['caption'])
            pos_x.append(meta['position_x'])
            pos_y.append(meta['position_y'])
            timestamps.append(meta['timestamp'])
            values.append(meta['value'])
        entities = [ids, vectors, captions, pos_x, pos_y, timestamps, values]
        self.collection.insert(entities)