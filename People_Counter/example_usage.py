"""
사용 예제 스크립트
"""
import cv2
from src.detector import PersonDetector
from src.tracker import Tracker
from src.attribute_classifier import AttributeClassifier
from src.counter import AttributeCounter
from src.utils import draw_bbox, draw_counting_line, draw_statistics, FPSMeter


def example_webcam():
    """웹캠 사용 예제"""
    print("웹캠 예제 시작...")
    
    # 모듈 초기화
    detector = PersonDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    attribute_classifier = AttributeClassifier()
    counter = AttributeCounter(counting_line=None)
    fps_meter = FPSMeter()
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counting_line = (0, height // 2, width, height // 2)
    counter.counting_line = counting_line
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 처리 파이프라인
        detections = detector.detect(frame)
        
        attributes_list = []
        for det in detections:
            bbox = det[:4]
            attributes = attribute_classifier.classify(bbox, frame)
            attributes_list.append(attributes)
        
        tracks = tracker.update(detections, attributes_list)
        counter.update(tracks, (width // 2, height // 2))
        
        # 시각화
        frame = draw_counting_line(frame, counting_line)
        for track in tracks:
            frame = draw_bbox(frame, track.bbox, track.track_id, track.attributes)
        
        counts = counter.get_counts()
        frame = draw_statistics(frame, counts)
        
        fps_meter.update()
        fps_text = f"FPS: {fps_meter.get_fps():.1f}"
        cv2.putText(frame, fps_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('People Counter Example', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 최종 통계
    final_counts = counter.get_counts()
    print("\n=== 최종 통계 ===")
    print(f"총 카운트: {final_counts.get('total', 0)}")
    print("\n속성별:")
    for key, value in sorted(final_counts.get('by_attribute', {}).items()):
        print(f"  {key}: {value}")


def example_video_file(video_path: str):
    """비디오 파일 사용 예제"""
    print(f"비디오 파일 예제 시작: {video_path}")
    
    # 모듈 초기화
    detector = PersonDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    attribute_classifier = AttributeClassifier()
    counter = AttributeCounter(counting_line=None)
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counting_line = (0, height // 2, width, height // 2)
    counter.counting_line = counting_line
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 처리 파이프라인
        detections = detector.detect(frame)
        
        attributes_list = []
        for det in detections:
            bbox = det[:4]
            attributes = attribute_classifier.classify(bbox, frame)
            attributes_list.append(attributes)
        
        tracks = tracker.update(detections, attributes_list)
        counter.update(tracks, (width // 2, height // 2))
        
        # 진행 상황 출력
        if frame_count % 30 == 0:
            counts = counter.get_counts()
            print(f"Frame {frame_count}: Total = {counts.get('total', 0)}")
    
    cap.release()
    
    # 최종 통계
    final_counts = counter.get_counts()
    print("\n=== 최종 통계 ===")
    print(f"총 프레임: {frame_count}")
    print(f"총 카운트: {final_counts.get('total', 0)}")
    print("\n속성별:")
    for key, value in sorted(final_counts.get('by_attribute', {}).items()):
        print(f"  {key}: {value}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 비디오 파일 경로가 제공된 경우
        example_video_file(sys.argv[1])
    else:
        # 웹캠 사용
        example_webcam()
