"""
DeepSORT 기반 추적 모듈
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import cv2


class KalmanFilter:
    """간단한 칼만 필터 구현"""
    
    def __init__(self):
        # 상태 벡터: [cx, cy, s, r, vx, vy, vs]
        # cx, cy: 중심점, s: 면적, r: 종횡비, vx, vy, vs: 속도
        self.ndim = 7
        self.dt = 1.0
        
        # 상태 전이 행렬
        self.F = np.eye(7)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        
        # 관측 행렬 (중심점, 면적, 종횡비만 관측)
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
        # 공분산 행렬
        self.P = np.eye(7) * 1000
        self.Q = np.eye(7) * 0.03
        self.R = np.eye(4) * 1.0
        
        self.x = np.zeros((7, 1))
    
    def predict(self):
        """예측 단계"""
        new_x = self.F @ self.x
        new_P = self.F @ self.P @ self.F.T + self.Q
        
        # 예측된 상태가 유효한지 확인
        if np.any(np.isnan(new_x)) or np.any(np.isinf(new_x)):
            # 유효하지 않은 예측, 상태 리셋
            self.x = np.zeros((7, 1))
            self.P = np.eye(7) * 1000
            return self.x
        
        if np.any(np.isnan(new_P)) or np.any(np.isinf(new_P)):
            # 유효하지 않은 공분산, 리셋
            self.P = np.eye(7) * 1000
        else:
            self.P = new_P
        
        self.x = new_x
        return self.x
    
    def update(self, z):
        """업데이트 단계"""
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # S 행렬이 역행렬을 가질 수 있는지 확인
        try:
            S_inv = np.linalg.inv(S)
            # NaN이나 무한대 체크
            if np.any(np.isnan(S_inv)) or np.any(np.isinf(S_inv)):
                return  # 업데이트 실패, 상태 유지
            
            K = self.P @ self.H.T @ S_inv
            new_x = self.x + K @ y
            
            # 업데이트된 상태가 유효한지 확인
            if np.any(np.isnan(new_x)) or np.any(np.isinf(new_x)):
                return  # 유효하지 않은 업데이트, 상태 유지
            
            self.x = new_x
            self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            # 역행렬 계산 실패 시 업데이트 건너뜀
            pass
    
    def get_state(self):
        """현재 상태 반환"""
        return self.x.flatten()


class Track:
    """단일 추적 객체"""
    
    def __init__(self, bbox: Tuple[int, int, int, int], track_id: int, 
                 attributes: Optional[Dict] = None):
        """
        Args:
            bbox: (x1, y1, x2, y2) 바운딩 박스
            track_id: 추적 ID
            attributes: 속성 딕셔너리
        """
        self.track_id = track_id
        self.bbox = bbox
        self.attributes = attributes or {}
        self.kalman = KalmanFilter()
        
        # 바운딩 박스를 칼만 필터 상태로 변환
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
        
        self.kalman.x = np.array([[cx], [cy], [s], [r], [0], [0], [0]])
        
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
    
    def predict(self):
        """예측"""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: Tuple[int, int, int, int], attributes: Optional[Dict] = None):
        """업데이트"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
        
        z = np.array([cx, cy, s, r])
        self.kalman.update(z)
        
        self.bbox = bbox
        if attributes:
            self.attributes = attributes
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """현재 상태를 바운딩 박스로 반환"""
        state = self.kalman.get_state()
        cx, cy, s, r = state[0], state[1], state[2], state[3]
        
        # NaN 값 체크 및 처리
        if np.isnan(cx) or np.isnan(cy) or np.isnan(s) or np.isnan(r):
            # 유효하지 않은 상태일 경우, 마지막 유효한 bbox 반환
            return self.bbox
        
        # 면적(s)과 종횡비(r)가 유효한지 확인
        if s <= 0 or r <= 0 or np.isinf(s) or np.isinf(r):
            # 유효하지 않은 경우, 마지막 유효한 bbox 반환
            return self.bbox
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        
        # w 또는 h가 NaN이나 무한대인지 확인
        if np.isnan(w) or np.isnan(h) or np.isinf(w) or np.isinf(h):
            return self.bbox
        
        # 계산된 값이 유효한지 확인 후 정수로 변환
        try:
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # 최종 결과가 NaN인지 확인
            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                return self.bbox
            
            return (x1, y1, x2, y2)
        except (ValueError, OverflowError):
            # 변환 실패 시 마지막 유효한 bbox 반환
            return self.bbox


class Tracker:
    """DeepSORT 스타일 추적기"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: 추적이 사라지기 전 최대 프레임 수
            min_hits: 추적이 확정되기 전 최소 히트 수
            iou_threshold: IOU 임계값
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_id = 1
    
    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """IOU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _associate_detections_to_tracks(self, detections: List[Tuple], 
                                       tracks: List[Track]) -> Tuple[List, List, List]:
        """감지와 추적 매칭"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # IOU 매트릭스 계산
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._iou(det[:4], track.get_state())
        
        # 헝가리안 알고리즘 (간단한 그리디 매칭)
        matched_indices = []
        unmatched_detections = []
        unmatched_tracks = []
        
        # IOU가 임계값 이상인 매칭 찾기
        if iou_matrix.size > 0:
            for d in range(len(detections)):
                best_iou = 0
                best_track = -1
                for t in range(len(tracks)):
                    if iou_matrix[d, t] > best_iou and iou_matrix[d, t] > self.iou_threshold:
                        best_iou = iou_matrix[d, t]
                        best_track = t
                
                if best_track >= 0:
                    matched_indices.append((d, best_track))
                else:
                    unmatched_detections.append(d)
            
            matched_tracks = set([t for _, t in matched_indices])
            unmatched_tracks = [t for t in range(len(tracks)) if t not in matched_tracks]
        else:
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(tracks)))
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Tuple], attributes_list: List[Dict]) -> List[Track]:
        """
        추적 업데이트
        
        Args:
            detections: List of (x1, y1, x2, y2, conf) 바운딩 박스
            attributes_list: 각 감지에 대한 속성 리스트
            
        Returns:
            활성 추적 리스트
        """
        self.frame_count += 1
        
        # 모든 추적에 대해 예측
        for track in self.tracks:
            track.predict()
        
        # 매칭
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # 매칭된 추적 업데이트
        for d_idx, t_idx in matched:
            det = detections[d_idx]
            attrs = attributes_list[d_idx] if d_idx < len(attributes_list) else {}
            self.tracks[t_idx].update(det[:4], attrs)
        
        # 매칭되지 않은 감지에 대해 새 추적 생성
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            attrs = attributes_list[d_idx] if d_idx < len(attributes_list) else {}
            new_track = Track(det[:4], self.next_id, attrs)
            self.next_id += 1
            self.tracks.append(new_track)
        
        # 오래된 추적 제거
        self.tracks = [t for t in self.tracks 
                      if not (t.time_since_update > self.max_age)]
        
        # 활성 추적만 반환 (min_hits 이상)
        active_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        
        return active_tracks
