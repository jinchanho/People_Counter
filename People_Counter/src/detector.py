"""
YOLOv8 기반 사람 감지 모듈
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class PersonDetector:
    """YOLOv8을 사용한 사람 감지 클래스"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        Args:
            model_path: YOLOv8 모델 경로
            conf_threshold: 신뢰도 임계값
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO 데이터셋에서 사람 클래스 ID는 0
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        프레임에서 사람을 감지
        
        Args:
            frame: 입력 프레임 (BGR)
            
        Returns:
            List of (x1, y1, x2, y2, confidence) bounding boxes
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 사람 클래스만 필터링
                if int(box.cls) == self.person_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return detections
