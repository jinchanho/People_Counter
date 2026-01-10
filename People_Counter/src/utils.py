"""
유틸리티 함수들
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict
import time


def draw_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
              track_id: int, attributes: Dict, color: Tuple[int, int, int] = None) -> np.ndarray:
    """바운딩 박스와 정보 그리기"""
    x1, y1, x2, y2 = bbox
    
    # 색상 결정
    if color is None:
        # Track ID 기반 색상
        color = generate_color(track_id)
    
    # 바운딩 박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 라벨 생성
    label_parts = [f"ID: {track_id}"]
    if attributes:
        if 'top_color' in attributes and attributes['top_color'] != 'unknown':
            label_parts.append(f"Top: {attributes['top_color']}")
        if 'bottom_color' in attributes and attributes['bottom_color'] != 'unknown':
            label_parts.append(f"Bottom: {attributes['bottom_color']}")
        # 기존 color 속성도 지원 (하위 호환성)
        if 'color' in attributes and attributes['color'] != 'unknown':
            label_parts.append(f"Color: {attributes['color']}")
        if 'gender' in attributes and attributes['gender'] != 'unknown':
            label_parts.append(f"Gender: {attributes['gender']}")
        if 'bag' in attributes and attributes['bag'] != 'unknown':
            bag_text = "Bag" if attributes['bag'] == 'yes' else "No Bag"
            label_parts.append(bag_text)
        if 'mask' in attributes and attributes['mask'] != 'unknown':
            mask_text = "Mask" if attributes['mask'] == 'yes' else "No Mask"
            label_parts.append(mask_text)
    
    label = " | ".join(label_parts)
    
    # 라벨 배경
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame, 
        (x1, y1 - label_height - 10), 
        (x1 + label_width, y1), 
        color, 
        -1
    )
    
    # 라벨 텍스트
    cv2.putText(
        frame, 
        label, 
        (x1, y1 - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (255, 255, 255), 
        1
    )
    
    return frame


def generate_color(track_id: int) -> Tuple[int, int, int]:
    """Track ID 기반 색상 생성"""
    np.random.seed(track_id)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color


def draw_counting_line(frame: np.ndarray, line: Tuple[int, int, int, int], 
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """카운팅 라인 그리기"""
    x1, y1, x2, y2 = line
    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, 
        "Counting Line", 
        (x1, y1 - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        color, 
        2
    )
    return frame


def draw_statistics(frame: np.ndarray, counts: Dict, 
                   position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """통계 정보 그리기"""
    x, y = position
    line_height = 25
    
    # 총 카운트
    cv2.putText(
        frame, 
        f"Total Count: {counts.get('total', 0)}", 
        (x, y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )
    y += line_height
    
    # 속성별 카운트
    by_attr = counts.get('by_attribute', {})
    if by_attr:
        cv2.putText(
            frame, 
            "By Attribute:", 
            (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
        y += line_height
        
        for key, value in sorted(by_attr.items()):
            text = f"  {key}: {value}"
            cv2.putText(
                frame, 
                text, 
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (200, 200, 200), 
                1
            )
            y += line_height
    
    # 조합별 카운트
    by_comb = counts.get('by_combination', {})
    if by_comb:
        cv2.putText(
            frame, 
            "By Combination:", 
            (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
        y += line_height
        
        for key, value in sorted(by_comb.items()):
            text = f"  {key}: {value}"
            cv2.putText(
                frame, 
                text, 
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (200, 200, 200), 
            1
            )
            y += line_height
    
    return frame


class FPSMeter:
    """FPS 측정 클래스"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
    
    def update(self):
        """FPS 업데이트"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
    
    def get_fps(self) -> float:
        """현재 FPS 반환"""
        return self.fps
    
    def reset(self):
        """리셋"""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
