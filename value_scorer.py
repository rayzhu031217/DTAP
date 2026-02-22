import cv2
import numpy as np
from utils import detect_objects, has_event, frame_difference, depth_difference

class ValueScorer:
    def __init__(self, config):
        self.config = config
        self.seen_objects = set()  # 全局已见物体集合（跨片段）

    def compute(self, clip_frames, caption, timestamp):
        """
        计算视频片段的价值分数
        clip_frames: 片段内的帧列表，每帧为字典 {'rgb':..., 'depth':...}
        caption: VLM生成的描述字符串
        timestamp: 片段起始时间戳（秒）
        返回: 价值分数 (float)
        """
        # 提取 RGB 和深度列表
        rgb_frames = [f['rgb'] for f in clip_frames]
        depth_frames = [f['depth'] for f in clip_frames]

        # 1. 视觉显著性：结合 RGB 差异和深度差异
        rgb_saliency = frame_difference(rgb_frames[0], rgb_frames[-1])
        depth_saliency = depth_difference(depth_frames[0], depth_frames[-1])
        saliency = 0.5 * rgb_saliency + 0.5 * depth_saliency
        # 归一化（可改进为自适应）
        saliency = min(saliency / 100.0, 1.0)

        # 2. 物体新颖性：利用深度过滤远处物体
        objects = detect_objects(
            rgb_frames[-1],
            depth_frames[-1],
            self.config.OBJECT_CLASSES,
            max_distance=self.config.OBJECT_MAX_DISTANCE
        )
        new_objects = objects - self.seen_objects
        novelty = len(new_objects) / 10.0  # 假设最多10个新物体
        novelty = min(novelty, 1.0)
        self.seen_objects.update(objects)

        # 3. 事件显著性：用描述判断
        event_score = 1.0 if has_event(caption) else 0.0

        # 加权求和
        value = (self.config.WEIGHT_SALIENCY * saliency +
                 self.config.WEIGHT_NEW_OBJECTS * novelty +
                 self.config.WEIGHT_EVENT * event_score)
        return value