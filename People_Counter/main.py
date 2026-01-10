"""
메인 애플리케이션: 속성 기반 피플 카운터
"""
import cv2
import argparse
import numpy as np
from src.detector import PersonDetector
from src.tracker import Tracker
from src.attribute_classifier import AttributeClassifier
from src.counter import AttributeCounter
from src.utils import draw_bbox, draw_counting_line, draw_statistics, FPSMeter


def main():
    parser = argparse.ArgumentParser(description='People Counter with Attribute-based Filtering')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model path')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--line', type=str, default=None,
                       help='Counting line coordinates (x1,y1,x2,y2)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Display video')
    parser.add_argument('--display_size', type=str, default=None,
                       help='Display window size (e.g., "0.5" for 50%, or "640,480" for specific pixels)')
    
    args = parser.parse_args()
    
    # 비디오 소스 설정
    if args.source == '0' or args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 카운팅 라인 설정
    counting_line = None
    if args.line:
        coords = list(map(int, args.line.split(',')))
        if len(coords) == 4:
            counting_line = tuple(coords)
    
    # 모듈 초기화
    detector = PersonDetector(model_path=args.model, conf_threshold=args.conf)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    attribute_classifier = AttributeClassifier()
    counter = AttributeCounter(counting_line=counting_line)
    fps_meter = FPSMeter()
    
    # 비디오 캡처
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # 비디오 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 디스플레이 크기 설정
    display_width = width
    display_height = height
    if args.display_size:
        if ',' in args.display_size:
            # "width,height" 형식
            try:
                w, h = map(int, args.display_size.split(','))
                display_width = w
                display_height = h
            except ValueError:
                print("Warning: Invalid display_size format. Using original size.")
        else:
            # 비율 형식
            try:
                scale = float(args.display_size)
                display_width = int(width * scale)
                display_height = int(height * scale)
            except ValueError:
                print("Warning: Invalid display_size format. Using original size.")
    
    # 출력 비디오 설정
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps if fps > 0 else 30.0, (width, height))
    
    # 기본 카운팅 라인 (설정되지 않은 경우)
    if counting_line is None:
        # 화면 중앙에 수직선 (세로선)
        counting_line = (width // 2, 0, width // 2, height)
    
    print("Starting people counter...")
    print("Press 'q' to quit, 'r' to reset counter")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 사람 감지
            detections = detector.detect(frame)
            
            # 속성 분류
            attributes_list = []
            for det in detections:
                bbox = det[:4]
                attributes = attribute_classifier.classify(bbox, frame)
                attributes_list.append(attributes)
            
            # 추적 업데이트
            tracks = tracker.update(detections, attributes_list)
            
            # 카운팅 업데이트
            frame_center = (width // 2, height // 2)
            counter.update(tracks, frame_center)
            
            # 시각화
            # 카운팅 라인 그리기
            frame = draw_counting_line(frame, counting_line)
            
            # 추적된 사람 그리기
            for track in tracks:
                frame = draw_bbox(frame, track.bbox, track.track_id, track.attributes)
            
            # 통계 정보 그리기
            counts = counter.get_counts()
            frame = draw_statistics(frame, counts)
            
            # FPS 표시
            fps_meter.update()
            fps_text = f"FPS: {fps_meter.get_fps():.1f}"
            cv2.putText(
                frame, 
                fps_text, 
                (width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            
            # 출력 비디오에 쓰기
            if out:
                out.write(frame)
            
            # 화면에 표시
            if args.display:
                display_frame = frame
                if display_width != width or display_height != height:
                    display_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow('People Counter', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    counter.reset()
                    print("Counter reset")
            
            # 진행 상황 출력 (매 30프레임마다)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Total count = {counts.get('total', 0)}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # 정리
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 최종 통계 출력
        final_counts = counter.get_counts()
        print("\n=== Final Statistics ===")
        print(f"Total Count: {final_counts.get('total', 0)}")
        print("\nBy Attribute:")
        for key, value in sorted(final_counts.get('by_attribute', {}).items()):
            print(f"  {key}: {value}")
        print("\nBy Combination:")
        for key, value in sorted(final_counts.get('by_combination', {}).items()):
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
