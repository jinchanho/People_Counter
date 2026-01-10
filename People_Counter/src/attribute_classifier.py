"""
속성 분류 모듈 (옷색깔, 성별, 가방, 마스크)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from collections import Counter


class ColorClassifier:
    """기본 색상 분류 클래스 (공통 메서드)"""
    
    def __init__(self):
        # 주요 색상 카테고리 정의
        self.color_categories = {
            'black': ([0, 0, 0], [180, 255, 30]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'purple': ([130, 100, 100], [160, 255, 255]),
            'pink': ([160, 100, 100], [180, 255, 255]),
            'gray': ([0, 0, 30], [180, 30, 200]),
            'brown': ([10, 100, 20], [20, 255, 100])
        }
    
    def _classify_region(self, roi: np.ndarray) -> str:
        """ROI 영역의 색상 분류 (공통 메서드)"""
        if roi.size == 0:
            return 'unknown'
        
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 각 색상 카테고리와의 매칭 점수 계산
        color_scores = {}
        for color_name, (lower, upper) in self.color_categories.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            score = np.sum(mask > 0) / mask.size
            color_scores[color_name] = score
        
        # 가장 높은 점수의 색상 반환
        if color_scores:
            dominant_color = max(color_scores, key=color_scores.get)
            if color_scores[dominant_color] > 0.05:  # 최소 임계값
                return dominant_color
        
        return 'unknown'


class TopColorClassifier(ColorClassifier):
    """상의 색상 분류 클래스"""
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """
        바운딩 박스 영역의 상의 색상 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            상의 색상 카테고리
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 바운딩 박스 영역 추출
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 상의 영역 (전체 높이의 상단 30-50% 부분)
        # 목 부분은 제외하고 가슴/배 부분만 사용
        body_height = y2 - y1
        top_y1 = y1 + int(body_height * 0.25)  # 목 부분 제외
        top_y2 = y1 + int(body_height * 0.50)  # 허리 위까지
        
        roi = frame[top_y1:top_y2, x1:x2]
        
        return self._classify_region(roi)


class BottomColorClassifier(ColorClassifier):
    """하의 색상 분류 클래스"""
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """
        바운딩 박스 영역의 하의 색상 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            하의 색상 카테고리
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 바운딩 박스 영역 추출
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 하의 영역 (전체 높이의 하단 50-90% 부분)
        # 허리 아래부터 다리 부분
        body_height = y2 - y1
        bottom_y1 = y1 + int(body_height * 0.50)  # 허리 아래
        bottom_y2 = y1 + int(body_height * 0.90)  # 무릎 위까지
        
        roi = frame[bottom_y1:bottom_y2, x1:x2]
        
        return self._classify_region(roi)


class GenderClassifier:
    """성별 분류 클래스 (간단한 휴리스틱 기반)"""
    
    def __init__(self):
        # 실제 프로덕션에서는 딥러닝 모델 사용 권장
        # 여기서는 간단한 휴리스틱 사용 (실제로는 학습된 모델 필요)
        pass
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """
        바운딩 박스 영역의 성별 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            'male', 'female', 또는 'unknown'
        """
        # 실제 구현에서는 학습된 모델 사용
        # 여기서는 랜덤 또는 휴리스틱 기반 분류
        # 실제 프로젝트에서는 ResNet 기반 분류 모델 사용 권장
        
        # 간단한 휴리스틱: 키와 체형 기반 (실제로는 부정확할 수 있음)
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / width if width > 0 else 1.0
        
        # 매우 간단한 휴리스틱 (실제로는 모델 필요)
        # 높이와 종횡비를 기반으로 추정
        if aspect_ratio > 2.2:
            return 'male'  # 일반적으로 남성이 더 키가 크고 마름
        elif aspect_ratio < 1.8:
            return 'female'
        else:
            return 'unknown'


class BagClassifier:
    """가방 유무 분류 클래스"""
    
    def __init__(self):
        pass
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """
        바운딩 박스 영역의 가방 유무 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            'yes' (가방 있음), 'no' (가방 없음), 또는 'unknown'
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 바운딩 박스 영역 추출
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 가방은 보통 옆구리나 등쪽에 있으므로, 좌우 측면 영역과 하단 영역 검사
        body_height = y2 - y1
        body_width = x2 - x1
        
        # 옆구리 영역 (좌우 20% 영역)
        side_width = int(body_width * 0.2)
        left_side = frame[y1:y2, x1:x1+side_width]
        right_side = frame[y1:y2, x2-side_width:x2]
        
        # 하단 30% 영역 (가방이 매달려 있을 수 있는 부분)
        bottom_y1 = y1 + int(body_height * 0.7)
        bottom_region = frame[bottom_y1:y2, x1:x2]
        
        # 간단한 휴리스틱: 측면이나 하단에 특정 패턴이 있는지 확인
        # 실제 구현에서는 딥러닝 모델 사용 권장
        
        # 옆구리 영역의 텍스처 복잡도 확인 (가방이 있으면 패턴이 복잡함)
        def calculate_complexity(roi):
            if roi.size == 0:
                return 0
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            # 라플라시안 필터로 엣지 검출하여 복잡도 측정
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
        
        left_complexity = calculate_complexity(left_side)
        right_complexity = calculate_complexity(right_side)
        bottom_complexity = calculate_complexity(bottom_region)
        
        # 복잡도 임계값 (경험적 값, 실제로는 학습 필요)
        complexity_threshold = 50.0
        max_complexity = max(left_complexity, right_complexity, bottom_complexity)
        
        # 측면이나 하단에 복잡한 패턴이 있으면 가방이 있을 가능성
        if max_complexity > complexity_threshold:
            # 추가: 색상 일관성 체크 (가방은 보통 단색이나 명확한 패턴을 가짐)
            if max_complexity > complexity_threshold * 1.5:
                return 'yes'
            elif max_complexity > complexity_threshold:
                return 'yes'  # 가능성 있음
            else:
                return 'no'
        else:
            return 'no'


class MaskClassifier:
    """마스크 착용 여부 분류 클래스"""
    
    def __init__(self):
        pass
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """
        바운딩 박스 영역의 마스크 착용 여부 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            'yes' (마스크 착용), 'no' (미착용), 또는 'unknown'
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 바운딩 박스 영역 추출
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 얼굴 영역 (상단 30-50% 영역)
        body_height = y2 - y1
        face_y1 = y1 + int(body_height * 0.3)
        face_y2 = y1 + int(body_height * 0.5)
        
        # 중앙 영역만 사용 (얼굴은 보통 중앙에 위치)
        body_width = x2 - x1
        face_x1 = x1 + int(body_width * 0.2)
        face_x2 = x2 - int(body_width * 0.2)
        
        face_roi = frame[face_y1:face_y2, face_x1:face_x2]
        
        if face_roi.size == 0:
            return 'unknown'
        
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # 마스크는 보통 파란색, 흰색, 검은색, 회색 계열
        # 피부색과는 다른 색상 영역이 얼굴 하단에 있는지 확인
        
        # 피부색 범위 (HSV)
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        
        # 마스크 색상 범위 (흰색, 파란색, 검은색, 회색)
        mask_colors = [
            ([0, 0, 200], [180, 30, 255]),  # 흰색
            ([100, 50, 50], [130, 255, 255]),  # 파란색
            ([0, 0, 0], [180, 255, 30]),  # 검은색
            ([0, 0, 30], [180, 30, 150]),  # 회색
        ]
        
        # 얼굴 영역을 상하로 나눔 (마스크는 하단에)
        face_height = face_y2 - face_y1
        upper_face = face_roi[:face_height//2, :]
        lower_face = face_roi[face_height//2:, :]
        
        # 상단 얼굴의 피부색 비율
        if upper_face.size > 0:
            upper_hsv = cv2.cvtColor(upper_face, cv2.COLOR_BGR2HSV)
            upper_skin_mask = cv2.inRange(upper_hsv, skin_lower, skin_upper)
            upper_skin_ratio = np.sum(upper_skin_mask > 0) / upper_skin_mask.size
        else:
            upper_skin_ratio = 0
        
        # 하단 얼굴의 피부색 비율
        if lower_face.size > 0:
            lower_hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            lower_skin_mask = cv2.inRange(lower_hsv, skin_lower, skin_upper)
            lower_skin_ratio = np.sum(lower_skin_mask > 0) / lower_skin_mask.size
            
            # 마스크 색상 비율 계산
            mask_color_ratio = 0
            for lower, upper in mask_colors:
                mask_mask = cv2.inRange(lower_hsv, np.array(lower, dtype=np.uint8), 
                                       np.array(upper, dtype=np.uint8))
                mask_color_ratio += np.sum(mask_mask > 0) / mask_mask.size
        else:
            lower_skin_ratio = 0
            mask_color_ratio = 0
        
        # 휴리스틱: 상단은 피부색이 많고, 하단은 피부색이 적고 마스크 색상이 많으면 마스크 착용
        if upper_skin_ratio > 0.3 and lower_skin_ratio < 0.2 and mask_color_ratio > 0.3:
            return 'yes'
        elif upper_skin_ratio > 0.2 and lower_skin_ratio < 0.15:
            return 'yes'  # 가능성 있음
        elif lower_skin_ratio > 0.4:
            return 'no'  # 하단에도 피부색이 많으면 마스크 미착용
        else:
            return 'unknown'


class AttributeClassifier:
    """통합 속성 분류 클래스"""
    
    def __init__(self):
        self.top_color_classifier = TopColorClassifier()
        self.bottom_color_classifier = BottomColorClassifier()
        self.gender_classifier = GenderClassifier()
        self.bag_classifier = BagClassifier()
        self.mask_classifier = MaskClassifier()
    
    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> Dict[str, str]:
        """
        바운딩 박스의 모든 속성 분류
        
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            frame: 입력 프레임 (BGR)
            
        Returns:
            속성 딕셔너리 {'top_color': 'red', 'bottom_color': 'blue', 'gender': 'male', 'bag': 'yes', 'mask': 'yes', ...}
        """
        top_color = self.top_color_classifier.classify(bbox, frame)
        bottom_color = self.bottom_color_classifier.classify(bbox, frame)
        gender = self.gender_classifier.classify(bbox, frame)
        bag = self.bag_classifier.classify(bbox, frame)
        mask = self.mask_classifier.classify(bbox, frame)
        
        return {
            'top_color': top_color,
            'bottom_color': bottom_color,
            'gender': gender,
            'bag': bag,
            'mask': mask
        }
