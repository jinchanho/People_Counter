"""
성능 평가 모듈
"""
import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class PerformanceEvaluator:
    """성능 평가 클래스"""
    
    def __init__(self):
        self.detection_times = []
        self.tracking_times = []
        self.attribute_times = []
        self.total_times = []
        self.frame_count = 0
        self.detection_counts = []
        self.track_counts = []
    
    def record_detection_time(self, elapsed: float, num_detections: int):
        """감지 시간 기록"""
        self.detection_times.append(elapsed)
        self.detection_counts.append(num_detections)
    
    def record_tracking_time(self, elapsed: float, num_tracks: int):
        """추적 시간 기록"""
        self.tracking_times.append(elapsed)
        self.track_counts.append(num_tracks)
    
    def record_attribute_time(self, elapsed: float):
        """속성 분류 시간 기록"""
        self.attribute_times.append(elapsed)
    
    def record_total_time(self, elapsed: float):
        """전체 처리 시간 기록"""
        self.total_times.append(elapsed)
        self.frame_count += 1
    
    def get_statistics(self) -> Dict:
        """통계 반환"""
        stats = {}
        
        if self.detection_times:
            stats['detection'] = {
                'mean_time': np.mean(self.detection_times),
                'std_time': np.std(self.detection_times),
                'min_time': np.min(self.detection_times),
                'max_time': np.max(self.detection_times),
                'mean_fps': 1.0 / np.mean(self.detection_times) if np.mean(self.detection_times) > 0 else 0,
                'mean_detections': np.mean(self.detection_counts) if self.detection_counts else 0
            }
        
        if self.tracking_times:
            stats['tracking'] = {
                'mean_time': np.mean(self.tracking_times),
                'std_time': np.std(self.tracking_times),
                'min_time': np.min(self.tracking_times),
                'max_time': np.max(self.tracking_times),
                'mean_fps': 1.0 / np.mean(self.tracking_times) if np.mean(self.tracking_times) > 0 else 0,
                'mean_tracks': np.mean(self.track_counts) if self.track_counts else 0
            }
        
        if self.attribute_times:
            stats['attribute'] = {
                'mean_time': np.mean(self.attribute_times),
                'std_time': np.std(self.attribute_times),
                'min_time': np.min(self.attribute_times),
                'max_time': np.max(self.attribute_times),
                'mean_fps': 1.0 / np.mean(self.attribute_times) if np.mean(self.attribute_times) > 0 else 0
            }
        
        if self.total_times:
            stats['total'] = {
                'mean_time': np.mean(self.total_times),
                'std_time': np.std(self.total_times),
                'min_time': np.min(self.total_times),
                'max_time': np.max(self.total_times),
                'mean_fps': 1.0 / np.mean(self.total_times) if np.mean(self.total_times) > 0 else 0,
                'total_frames': self.frame_count
            }
        
        return stats
    
    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()
        
        print("\n=== Performance Statistics ===")
        
        if 'detection' in stats:
            d = stats['detection']
            print(f"\nDetection:")
            print(f"  Mean Time: {d['mean_time']*1000:.2f} ms")
            print(f"  Mean FPS: {d['mean_fps']:.2f}")
            print(f"  Mean Detections per Frame: {d['mean_detections']:.2f}")
        
        if 'tracking' in stats:
            t = stats['tracking']
            print(f"\nTracking:")
            print(f"  Mean Time: {t['mean_time']*1000:.2f} ms")
            print(f"  Mean FPS: {t['mean_fps']:.2f}")
            print(f"  Mean Tracks per Frame: {t['mean_tracks']:.2f}")
        
        if 'attribute' in stats:
            a = stats['attribute']
            print(f"\nAttribute Classification:")
            print(f"  Mean Time: {a['mean_time']*1000:.2f} ms")
            print(f"  Mean FPS: {a['mean_fps']:.2f}")
        
        if 'total' in stats:
            tot = stats['total']
            print(f"\nTotal Processing:")
            print(f"  Mean Time: {tot['mean_time']*1000:.2f} ms")
            print(f"  Mean FPS: {tot['mean_fps']:.2f}")
            print(f"  Total Frames: {tot['total_frames']}")
    
    def reset(self):
        """리셋"""
        self.detection_times.clear()
        self.tracking_times.clear()
        self.attribute_times.clear()
        self.total_times.clear()
        self.frame_count = 0
        self.detection_counts.clear()
        self.track_counts.clear()
