"""
속성별 선별 카운팅 모듈
"""
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np


class AttributeCounter:
    """속성별 카운터"""
    
    def __init__(self, counting_line: Tuple[int, int, int, int] = None):
        """
        Args:
            counting_line: (x1, y1, x2, y2) 카운팅 라인 (None이면 전체 프레임)
        """
        self.counting_line = counting_line
        self.total_count = 0
        self.attribute_counts = defaultdict(int)
        self.crossed_tracks: Set[int] = set()
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # 속성 조합별 카운트
        self.combined_counts = defaultdict(int)
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                                line: Tuple[int, int, int, int]) -> float:
        """점에서 선까지의 거리 계산"""
        px, py = point
        x1, y1, x2, y2 = line
        
        # 선분의 방향 벡터
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # 점인 경우
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # 선분 위의 가장 가까운 점 찾기
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def _has_crossed_line(self, track_id: int, current_center: Tuple[float, float]) -> bool:
        """추적이 카운팅 라인을 넘었는지 확인"""
        if self.counting_line is None:
            return False
        
        if track_id not in self.track_history:
            return False
        
        history = self.track_history[track_id]
        if len(history) < 2:
            return False
        
        # 이전 위치와 현재 위치
        prev_center = history[-2]
        curr_center = current_center
        
        x1, y1, x2, y2 = self.counting_line
        
        # 선분과 두 점의 교차 확인
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        line_start = (x1, y1)
        line_end = (x2, y2)
        
        return intersect(prev_center, curr_center, line_start, line_end)
    
    def update(self, tracks: List, frame_center: Tuple[int, int] = None):
        """
        추적 업데이트 및 카운팅
        
        Args:
            tracks: 활성 추적 리스트
            frame_center: 프레임 중심 (라인이 없을 때 사용)
        """
        current_track_ids = set()
        
        for track in tracks:
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # 바운딩 박스 중심 계산
            x1, y1, x2, y2 = track.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # 히스토리 업데이트
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:  # 최근 30프레임만 유지
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # 카운팅 라인 교차 확인
            if track_id not in self.crossed_tracks:
                if self.counting_line:
                    if self._has_crossed_line(track_id, center):
                        self._count_person(track)
                        self.crossed_tracks.add(track_id)
                elif frame_center:
                    # 라인이 없으면 프레임 중심을 지나면 카운트
                    cx, cy = frame_center
                    if len(self.track_history[track_id]) >= 5:
                        prev_center = self.track_history[track_id][-5]
                        if (prev_center[0] < cx < center[0] or prev_center[0] > cx > center[0]):
                            self._count_person(track)
                            self.crossed_tracks.add(track_id)
        
        # 사라진 추적의 히스토리 정리
        disappeared_tracks = set(self.track_history.keys()) - current_track_ids
        for track_id in disappeared_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
    
    def _count_person(self, track):
        """사람 카운팅"""
        self.total_count += 1
        
        attributes = track.attributes
        
        # 개별 속성 카운트
        for attr_name, attr_value in attributes.items():
            if attr_value and attr_value != 'unknown':
                self.attribute_counts[f"{attr_name}_{attr_value}"] += 1
        
        # 속성 조합 카운트
        top_color = attributes.get('top_color', 'unknown')
        bottom_color = attributes.get('bottom_color', 'unknown')
        color = attributes.get('color', 'unknown')  # 하위 호환성
        gender = attributes.get('gender', 'unknown')
        
        # 상의와 하의 색상 조합
        if top_color != 'unknown' and bottom_color != 'unknown':
            combined_key = f"{top_color}_{bottom_color}"
            self.combined_counts[combined_key] += 1
        elif top_color != 'unknown':
            self.combined_counts[f"top_{top_color}"] += 1
        elif bottom_color != 'unknown':
            self.combined_counts[f"bottom_{bottom_color}"] += 1
        
        # 기존 color 속성 지원 (하위 호환성)
        if color != 'unknown' and gender != 'unknown':
            combined_key = f"{color}_{gender}"
            self.combined_counts[combined_key] += 1
        elif color != 'unknown':
            self.combined_counts[f"{color}"] += 1
        
        if gender != 'unknown':
            if top_color != 'unknown' or bottom_color != 'unknown':
                # 상의/하의와 성별 조합
                if top_color != 'unknown' and bottom_color != 'unknown':
                    combined_key = f"{top_color}_{bottom_color}_{gender}"
                    self.combined_counts[combined_key] += 1
                elif top_color != 'unknown':
                    combined_key = f"{top_color}_{gender}"
                    self.combined_counts[combined_key] += 1
                elif bottom_color != 'unknown':
                    combined_key = f"{bottom_color}_{gender}"
                    self.combined_counts[combined_key] += 1
    
    def get_counts(self) -> Dict:
        """현재 카운트 반환"""
        return {
            'total': self.total_count,
            'by_attribute': dict(self.attribute_counts),
            'by_combination': dict(self.combined_counts)
        }
    
    def reset(self):
        """카운터 리셋"""
        self.total_count = 0
        self.attribute_counts.clear()
        self.combined_counts.clear()
        self.crossed_tracks.clear()
        self.track_history.clear()