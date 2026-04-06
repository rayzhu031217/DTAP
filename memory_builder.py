import time
import cv2
import config
from pymilvus import Collection, connections
from value_scorer import ValueScorer
from utils import (
    load_tum_sequence, embed_text, init_milvus_collection,
    generate_caption, R2RDataLoader
)

class MemoryBuilder:
    def __init__(self):
        connections.connect(alias='default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = init_milvus_collection(config.COLLECTION_NAME, config.VECTOR_DIM)
        self.scorer = ValueScorer(config)

    def build(self, rgb_txt=None, depth_txt=None, groundtruth_txt=None) -> int:
        """
        根据 config.DATA_SOURCE 选择数据加载方式。
        返回处理的总帧数。
        """
        if config.DATA_SOURCE == 'tum':
            if rgb_txt is None or depth_txt is None:
                raise ValueError("For TUM mode, rgb_txt and depth_txt must be provided.")
            total_frames = self._build_from_tum(rgb_txt, depth_txt, groundtruth_txt)
        elif config.DATA_SOURCE == 'r2r':
            total_frames = self._build_from_r2r()
        else:
            raise ValueError(f"Unknown DATA_SOURCE: {config.DATA_SOURCE}")
        return total_frames

    def _build_from_tum(self, rgb_txt, depth_txt, groundtruth_txt) -> int:
        """从 TUM 格式数据构建记忆，返回总帧数"""
        all_frames = load_tum_sequence(
            rgb_txt, depth_txt, groundtruth_txt,
            rgb_root='rgb', depth_root='depth',
            depth_scale=config.DEPTH_SCALE
        )
        if not all_frames:
            print("No frames loaded from TUM data.")
            return 0

        total_frames = len(all_frames)
        clips = self._split_into_clips(all_frames)
        self._process_clips(clips)
        return total_frames

    def _build_from_r2r(self) -> int:
        """从 R2R 数据集构建记忆，返回总帧数"""
        try:
            loader = R2RDataLoader(
                config.R2R_JSON_PATH,
                config.MP3D_DATA_DIR,
                scene_list=getattr(config, 'MP3D_SCENE_LIST', None),
                width=config.SIMULATOR_FRAME_WIDTH,
                height=config.SIMULATOR_FRAME_HEIGHT,
                hfov=90
            )
        except ImportError:
            print("Error: Habitat-sim not installed. Cannot load R2R data.")
            return 0

        total_frames = 0
        for frames in loader:
            total_frames += len(frames)
            if len(frames) >= config.MIN_CLIP_FRAMES:
                self._process_clip(frames)
            else:
                print(f"Skipped short path with {len(frames)} frames.")
        return total_frames

    def _split_into_clips(self, all_frames):
        """将连续帧序列按时间窗口切分为片段（用于 TUM 数据）"""
        clips = []
        duration = config.CLIP_DURATION
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
        return clips

    def _process_clips(self, clips):
        """批量处理 TUM 数据划分出的多个片段"""
        batch = []
        for clip_frames, start_time in clips:
            self._process_clip(clip_frames, start_time, batch)
        if batch:
            self._batch_insert(batch)

    def _process_clip(self, clip_frames, start_time=None, batch=None):
        """
        处理单个片段：
        - 生成描述
        - 计算价值
        - 若价值高于阈值，则嵌入并加入插入批次
        """
        if start_time is None:
            start_time = clip_frames[0]['timestamp'] if clip_frames else 0

        # 生成描述
        rgb_list = [f['rgb'] for f in clip_frames]
        caption = generate_caption(rgb_list)
        if not caption:
            caption = "No caption generated."

        # 获取位置（取中间帧的位姿）
        mid_idx = len(clip_frames) // 2
        pose = clip_frames[mid_idx].get('pose')
        if pose is None:
            pos = (0.0, 0.0)
        else:
            pos = (pose[0], pose[1])  # 取 tx, ty

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
            if batch is None:
                self._batch_insert([(emb, metadata)])
            else:
                batch.append((emb, metadata))
                if len(batch) >= 10:
                    self._batch_insert(batch)
                    batch.clear()
            print(f"Stored memory at {start_time:.3f}s, value={value:.2f}: {caption[:50]}...")
        else:
            print(f"Skipped memory at {start_time:.3f}s, value={value:.2f}")

    def _batch_insert(self, batch):
        """批量插入记忆到 Milvus"""
        if not batch:
            return
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
