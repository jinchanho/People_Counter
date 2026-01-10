# People Counter with Attribute-based Filtering

특정 속성(옷색깔, 성별 등)을 선별하여 카운팅하는 피플 카운터 시스템입니다. 단일 카메라를 사용하며, ReID(Re-identification) 기반 추적과 속성 분류를 통해 정확한 카운팅을 수행합니다.

![Result2](https://github.com/user-attachments/assets/66ba4a51-151b-4c23-942e-f8e5d760df6f)
![Result3](https://github.com/user-attachments/assets/f00462cf-0a53-4921-be60-440a6619c483)

## 주요 기능

- **사람 감지**: YOLOv8 기반 실시간 사람 감지
- **추적**: DeepSORT 스타일 추적 알고리즘으로 사람 추적
- **속성 분류**: 
  - 상의 색상 분류 (11가지 색상: black, white, red, blue, green, yellow, orange, purple, pink, gray, brown)
  - 하의 색상 분류 (11가지 색상: black, white, red, blue, green, yellow, orange, purple, pink, gray, brown)
  - 성별 분류 (male, female, unknown)
  - 가방 유무 분류 (yes, no, unknown)
  - 마스크 착용 여부 분류 (yes, no, unknown)
- **속성별 선별 카운팅**: 특정 속성 조합별로 사람 수 카운팅
- **실시간 시각화**: 바운딩 박스, 추적 ID, 속성 정보, 통계 표시
- **성능 측정**: FPS 및 처리 시간 측정

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU도 동작 가능)
- 웹캠 또는 비디오 파일

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/jinchanho/People_Counter.git
cd People_Counter
```

### 2. 가상환경 생성 및 활성화 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. YOLOv8 모델 다운로드

YOLOv8 모델은 첫 실행 시 자동으로 다운로드됩니다. 또는 수동으로 다운로드할 수 있습니다:

```bash
# Ultralytics에서 자동 다운로드 (기본값: yolov8n.pt)
# 또는 다른 모델 사용: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## 사용 방법

### 기본 사용 (웹캠)

```bash
python main.py
```

### 비디오 파일 사용

```bash
python main.py --source path/to/video.mp4
```

### 카운팅 라인 설정

```bash
# 카운팅 라인 좌표 지정 (x1,y1,x2,y2)
# 세로선 예시: 화면 왼쪽에서 1/3 위치 (x1=x2, y1=0, y2=height)
python main.py --line 320,0,320,480

# 가로선 예시 (필요시): 화면 중앙에 수평선 (x1=0, x2=width, y1=y2)
python main.py --line 0,240,640,240
```

### 출력 비디오 저장

```bash
python main.py --source input.mp4 --output output.mp4
```

### 화면 크기 조절

```bash
# 비율로 조절 (50% 크기)
python main.py --source video.mp4 --display_size 0.5

# 픽셀 크기로 지정 (너비, 높이)
python main.py --source video.mp4 --display_size 640,480
```

### 고급 옵션

```bash
python main.py \
    --source 0 \
    --model yolov8n.pt \
    --conf 0.5 \
    --line 320,0,320,480 \
    --output result.mp4 \
    --display \
    --display_size 0.7
```

### 명령줄 인자 설명

- `--source`: 비디오 소스 (0=웹캠, 또는 비디오 파일 경로)
- `--model`: YOLOv8 모델 경로 (기본값: yolov8n.pt)
- `--conf`: 신뢰도 임계값 (0.0-1.0, 기본값: 0.5)
- `--line`: 카운팅 라인 좌표 (x1,y1,x2,y2 형식, 기본값: 화면 중앙 수직선)
- `--output`: 출력 비디오 파일 경로
- `--display`: 비디오 표시 여부 (기본값: True)
- `--display_size`: 화면 표시 크기
  - 비율 형식: `0.5` (50% 크기), `0.7` (70% 크기) 등
  - 픽셀 형식: `640,480` (너비, 높이)
  - 지정하지 않으면 원본 크기로 표시

### 키보드 단축키

- `q`: 프로그램 종료
- `r`: 카운터 리셋

## 프로젝트 구조

```
People_Counter/
├── main.py                 # 메인 애플리케이션
├── requirements.txt        # Python 의존성
├── README.md              # 프로젝트 문서
├── .gitignore             # Git 무시 파일
└── src/                   # 소스 코드
    ├── __init__.py
    ├── detector.py        # YOLOv8 기반 사람 감지
    ├── tracker.py         # DeepSORT 스타일 추적
    ├── attribute_classifier.py  # 속성 분류 (색상, 성별)
    ├── counter.py         # 속성별 카운팅 로직
    ├── evaluator.py       # 성능 평가 모듈
    └── utils.py           # 유틸리티 함수
```

## 개발 과정

### 1. 아키텍처 설계

시스템은 다음 모듈로 구성됩니다:

1. **Detector (detector.py)**: YOLOv8을 사용한 사람 감지
2. **Tracker (tracker.py)**: 칼만 필터 기반 추적 알고리즘
3. **Attribute Classifier (attribute_classifier.py)**: 
   - HSV 색공간 기반 색상 분류
   - 휴리스틱 기반 성별 분류
   - 텍스처 복잡도 기반 가방 분류
   - 얼굴 영역 색상 패턴 기반 마스크 분류
4. **Counter (counter.py)**: 카운팅 라인 교차 감지 및 속성별 카운팅
5. **Utils (utils.py)**: 시각화 및 유틸리티 함수

### 2. 핵심 알고리즘

#### 사람 감지
- **YOLOv8**: Ultralytics의 최신 객체 감지 모델 사용
- COCO 데이터셋의 person 클래스 (ID: 0)만 필터링
- 신뢰도 임계값 기반 필터링

#### 추적 알고리즘
- **칼만 필터**: 상태 벡터 [cx, cy, s, r, vx, vy, vs] 사용
  - cx, cy: 중심점
  - s: 면적
  - r: 종횡비
  - vx, vy, vs: 속도
- **IOU 기반 매칭**: 감지와 추적 간 매칭
- **추적 생명주기 관리**: max_age, min_hits 파라미터로 추적 품질 제어

#### 속성 분류

**색상 분류**:
- HSV 색공간 변환
- 11가지 색상 카테고리 정의
- 상체 영역(전체 높이의 상단 60%)만 분석하여 옷 색상에 집중
- 각 색상 카테고리와의 매칭 점수 계산

**성별 분류**:
- 휴리스틱 기반 (키와 체형 비율)

**가방 분류**:
- 텍스처 복잡도 기반 휴리스틱 (측면 및 하단 영역 분석)

**마스크 분류**:
- 얼굴 하단 영역의 색상 패턴 분석 (피부색 vs 마스크 색상)

#### 카운팅 로직
- **카운팅 라인 교차 감지**: 선분 교차 알고리즘 사용
- **중복 카운팅 방지**: 추적 ID 기반 중복 제거
- **속성별 통계**: 개별 속성 및 속성 조합별 카운팅

### 3. 성능 최적화

- **배치 처리**: 여러 감지를 한 번에 처리
- **효율적인 IOU 계산**: NumPy 벡터화 연산
- **히스토리 제한**: 추적 히스토리를 최근 30프레임으로 제한

### 4. 시각화

- 실시간 바운딩 박스 및 추적 ID 표시
- 속성 정보 오버레이
- 통계 정보 패널
- FPS 표시
- 카운팅 라인 시각화

## 속성 정의

### 색상 카테고리 (11가지)

| 색상 | HSV 범위 |
|------|----------|
| Black | [0,0,0] - [180,255,30] |
| White | [0,0,200] - [180,30,255] |
| Red | [0,100,100] - [10,255,255] |
| Blue | [100,100,100] - [130,255,255] |
| Green | [40,100,100] - [80,255,255] |
| Yellow | [20,100,100] - [30,255,255] |
| Orange | [10,100,100] - [20,255,255] |
| Purple | [130,100,100] - [160,255,255] |
| Pink | [160,100,100] - [180,255,255] |
| Gray | [0,0,30] - [180,30,200] |
| Brown | [10,100,20] - [20,255,100] |

### 성별 카테고리

- **Male**: 남성
- **Female**: 여성
- **Unknown**: 분류 불가

### 가방 카테고리

- **Yes**: 가방을 메고 있음
- **No**: 가방 없음
- **Unknown**: 분류 불가

### 마스크 카테고리

- **Yes**: 마스크 착용
- **No**: 마스크 미착용
- **Unknown**: 분류 불가

### 속성 조합

시스템은 다음과 같은 조합으로도 카운팅합니다:
- `{top_color}_{bottom_color}`: 예) "red_blue" (빨간 상의 + 파란 하의)
- `{top_color}_{bottom_color}_{gender}`: 예) "red_blue_male"
- `{top_color}_{gender}`: 예) "red_male"
- `{bottom_color}_{gender}`: 예) "blue_female"
- `top_{color}`: 상의 색상만 알려진 경우
- `bottom_{color}`: 하의 색상만 알려진 경우
- `{gender}`: 성별만 알려진 경우

## 성능 평가

### 정확성
- YOLOv8의 높은 감지 정확도 활용
- 칼만 필터 기반 안정적인 추적
- 중복 카운팅 방지 메커니즘

### 추론 속도
- YOLOv8n (nano): ~30-50 FPS (GPU)
- YOLOv8s (small): ~20-30 FPS (GPU)
- CPU 모드: ~5-15 FPS (모델 크기에 따라 다름)

### 기능
- 실시간 속성 분류
- 다중 속성 조합 지원
- 실시간 통계 업데이트

## 개선 가능 사항

1. **성별 분류 모델**: 현재 휴리스틱 기반 → 딥러닝 모델로 교체
2. **ReID 모델 통합**: FastReID 등 전문 ReID 모델 통합
3. **추가 속성**: 나이, 체형, 가방 유무 등
4. **다중 카메라 지원**: 여러 카메라 동시 처리
5. **데이터베이스 연동**: 카운팅 결과 저장 및 분석

## 문제 해결

### 웹캠이 열리지 않는 경우
- 다른 프로그램이 웹캠을 사용 중인지 확인
- `--source` 인자로 다른 카메라 인덱스 시도 (1, 2, ...)

### 낮은 FPS
- 더 작은 YOLOv8 모델 사용 (yolov8n.pt)
- `--conf` 값을 높여서 감지 수 감소
- GPU 사용 확인

### 속성 분류가 부정확한 경우
- 조명 조건 확인
- 색상 분류 HSV 범위 조정 필요 시 `attribute_classifier.py` 수정
- 성별 분류는 실제 모델로 교체 권장

## 라이선스

이 프로젝트는 오픈소스 라이선스를 따릅니다.

## 참고 자료

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [OpenCV Documentation](https://docs.opencv.org/)
