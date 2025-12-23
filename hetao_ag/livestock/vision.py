# -*- coding: utf-8 -*-
"""
hetao_ag.livestock.vision - 动物视觉检测

基于深度学习的动物检测和计数。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class Detection:
    """检测结果"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str


class AnimalDetector:
    """动物检测器
    
    基于YOLO的动物检测,支持边缘设备部署。
    
    示例:
        >>> detector = AnimalDetector()
        >>> detections = detector.detect("farm_image.jpg")
        >>> cows = [d for d in detections if d.label == "cow"]
    """
    
    SUPPORTED_ANIMALS = ["cow", "sheep", "goat", "horse", "pig", "chicken"]
    
    def __init__(
        self,
        model_name: str = "yolov5s",
        confidence_threshold: float = 0.5,
        use_gpu: bool = True
    ):
        """初始化检测器
        
        参数:
            model_name: 模型名称
            confidence_threshold: 置信度阈值
            use_gpu: 是否使用GPU
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.model = None
        self._model_loaded = False
    
    def load_model(self):
        """加载检测模型"""
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
            
            if self.use_gpu and torch.cuda.is_available():
                self.model.to('cuda')
                try:
                    self.model.half()
                except:
                    pass
            
            self._model_loaded = True
        except ImportError:
            print("警告: PyTorch未安装,使用模拟模式")
            self._model_loaded = False
    
    def detect(self, image) -> List[Detection]:
        """检测图像中的动物
        
        参数:
            image: 图像路径、numpy数组或URL
            
        返回:
            Detection列表
        """
        if not self._model_loaded:
            # 模拟检测结果
            return self._simulate_detection()
        
        results = self.model(image)
        detections = []
        
        for *bbox, conf, cls_idx in results.xyxy[0].tolist():
            label = results.names[int(cls_idx)]
            
            if conf >= self.confidence_threshold:
                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=int(cls_idx),
                    label=label
                ))
        
        return detections
    
    def _simulate_detection(self) -> List[Detection]:
        """模拟检测结果(用于测试)"""
        return [
            Detection((100, 100, 300, 300), 0.95, 0, "cow"),
            Detection((400, 150, 550, 350), 0.87, 0, "cow"),
        ]
    
    def count_animals(self, image, species: Optional[str] = None) -> Dict[str, int]:
        """统计动物数量
        
        参数:
            image: 输入图像
            species: 指定物种(可选)
            
        返回:
            各物种数量字典
        """
        detections = self.detect(image)
        
        counts = {}
        for det in detections:
            if species and det.label != species:
                continue
            counts[det.label] = counts.get(det.label, 0) + 1
        
        return counts


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """计算两个边界框的IoU
    
    参数:
        box1, box2: (x1, y1, x2, y2)格式的边界框
        
    返回:
        IoU值(0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    print("=" * 50)
    print("动物视觉检测演示")
    print("=" * 50)
    
    detector = AnimalDetector(confidence_threshold=0.5)
    
    # 模拟检测
    detections = detector.detect("test_image.jpg")
    
    print(f"\n检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"  {det.label}: 置信度={det.confidence:.2f}, 位置={det.bbox}")
    
    counts = detector.count_animals("test_image.jpg")
    print(f"\n动物计数: {counts}")
