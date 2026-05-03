# AI 기반 한인 디아스포라 기록 유산 디지털화 시스템
## 이미지 처리 모듈 v1.0.0

박사논문 "AI 기반 한인 디아스포라 기록 유산 디지털화 및 실감형 콘텐츠 생성 모델 연구"의 
6장 구현 코드입니다.

---

## 📋 개요

이 시스템은 노후화된 디아스포라 기록 유산 이미지를 AI 기술을 활용하여 자동으로 복원하고 분석합니다.

### 주요 기능

| 모듈 | 기술 | 기능 |
|------|------|------|
| 초해상도 복원 | Real-ESRGAN | 저해상도 이미지 4배 업스케일링 |
| 얼굴 향상 | GFPGAN | 손상된 얼굴 복원 및 선명화 |
| 흑백 컬러화 | DeOldify | 흑백 사진 자동 컬러화 |
| 이미지 분석 | BLIP, CLIP, face_recognition | 캡션 생성, 장면 분류, 얼굴 감지 |

---

## 🚀 설치

### 요구사항

- Python 3.9+
- CUDA 11.8+ (GPU 사용 시)
- 8GB+ GPU 메모리 권장

### 설치 단계

```bash
# 1. 저장소 클론 또는 디렉토리 이동
cd diaspora_image_processing

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 프로젝트 구조 생성
bash setup_project.sh

# 5. 사전학습 모델 다운로드
python download_models.py --all
```

---

## 💻 사용법

### 기본 사용

```bash
# 단일 이미지 처리
python main.py -i input.jpg -o output.jpg

# 배치 처리 (디렉토리)
python main.py -i ./input_dir -o ./output_dir --batch

# 설정 파일 사용
python main.py -i input.jpg -o output.jpg --config config.yaml
```

### 옵션

```bash
# 특정 단계 비활성화
python main.py -i input.jpg -o output.jpg --no-sr      # 초해상도 제외
python main.py -i input.jpg -o output.jpg --no-face    # 얼굴향상 제외
python main.py -i input.jpg -o output.jpg --no-color   # 컬러화 제외

# 중간 결과 저장
python main.py -i input.jpg -o output.jpg --save-intermediate

# 처리 보고서 생성
python main.py -i input.jpg -o output.jpg --report report.json

# CPU 모드
python main.py -i input.jpg -o output.jpg --device cpu
```

### Python API 사용

```python
from src import ImageRestorationPipeline, ProcessingOptions

# 파이프라인 초기화
pipeline = ImageRestorationPipeline(device='cuda')

# 이미지 처리
result = pipeline.process("old_photo.jpg", "restored_photo.jpg")

# 결과 확인
print(f"캡션: {result.caption}")
print(f"장면: {result.scenes}")
print(f"얼굴 수: {result.face_count}")
print(f"처리 시간: {result.total_time:.2f}초")
```

### 개별 모듈 사용

```python
# 초해상도만 적용
from src.modules import SuperResolutionModule
sr = SuperResolutionModule(device='cuda')
result = sr.enhance("low_res.jpg", outscale=4)
result['enhanced'].save("high_res.jpg")

# 얼굴 향상만 적용
from src.modules import FaceEnhancementModule
face = FaceEnhancementModule(device='cuda')
result = face.enhance("portrait.jpg")
result['enhanced'].save("enhanced_portrait.jpg")

# 컬러화만 적용
from src.modules import ColorizationModule
colorizer = ColorizationModule(device='cuda')
result = colorizer.colorize("bw_photo.jpg")
result['colorized'].save("color_photo.jpg")

# 이미지 분석만 수행
from src.modules import ImageAnalysisModule
analyzer = ImageAnalysisModule(device='cuda')
result = analyzer.analyze("photo.jpg")
print(result['caption'])
print(result['metadata'])
```

---

## 📁 프로젝트 구조

```
diaspora_image_processing/
├── config.yaml              # 설정 파일
├── requirements.txt         # 의존성 목록
├── main.py                  # 메인 실행 파일
├── download_models.py       # 모델 다운로드 스크립트
├── test_pipeline.py         # 테스트 스크립트
├── setup_project.sh         # 프로젝트 초기화 스크립트
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # 통합 파이프라인
│   └── modules/
│       ├── __init__.py
│       ├── super_resolution.py   # Real-ESRGAN
│       ├── face_enhancement.py   # GFPGAN
│       ├── colorization.py       # DeOldify
│       └── image_analysis.py     # BLIP, CLIP, face_recognition
│
├── models/                  # 사전학습 모델 저장
│   ├── realesrgan/
│   ├── gfpgan/
│   └── deoldify/
│
├── data/
│   ├── input/               # 입력 이미지
│   ├── output/              # 출력 이미지
│   └── test_images/         # 테스트 이미지
│
└── logs/                    # 로그 파일
```

---

## ⚙️ 설정 (config.yaml)

```yaml
# 시스템 설정
system:
  device: "cuda"
  gpu_id: 0

# 초해상도 설정
super_resolution:
  enabled: true
  model_name: "RealESRGAN_x4plus"
  scale: 4
  auto_trigger:
    min_resolution: 512
    max_resolution: 2048

# 얼굴 향상 설정
face_enhancement:
  enabled: true
  model_name: "GFPGANv1.4"
  upscale: 2

# 컬러화 설정
colorization:
  enabled: true
  model_type: "artistic"
  render_factor: 35

# 이미지 분석 설정
image_analysis:
  captioning:
    enabled: true
    model_name: "Salesforce/blip-image-captioning-large"
  scene_classification:
    enabled: true
  face_detection:
    enabled: true

# 파이프라인 설정
pipeline:
  conditional_execution: true
  save_intermediate: false
```

---

## 📊 평가 지표

| 지표 | 설명 | 목표값 |
|------|------|--------|
| PSNR | 피크 신호 대 잡음비 | > 25 dB |
| SSIM | 구조적 유사도 | > 0.8 |
| LPIPS | 지각적 유사도 | < 0.3 |
| 처리 시간 | 이미지당 평균 | < 10초 |

---

## 🔬 기술 스택

### AI/ML 라이브러리
- **PyTorch** 2.1+: 딥러닝 프레임워크
- **Real-ESRGAN**: 초해상도 복원 (Wang et al., 2021)
- **GFPGAN**: 얼굴 복원 (Wang et al., 2021)
- **DeOldify**: 흑백 컬러화 (Antic, 2019)
- **BLIP**: 이미지 캡셔닝 (Li et al., 2022)
- **CLIP**: 제로샷 분류 (Radford et al., 2021)
- **face_recognition**: 얼굴 감지/인코딩

### 유틸리티
- OpenCV, Pillow, NumPy
- transformers (HuggingFace)
- scikit-image, lpips (평가)

---

## 📚 참고문헌

1. Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. ICCVW.
2. Wang, X., et al. (2021). Towards Real-World Blind Face Restoration with Generative Facial Prior. CVPR.
3. Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML.
4. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.

---

## 📝 라이선스

이 프로젝트는 학술 연구 목적으로 개발되었습니다.
각 오픈소스 라이브러리의 라이선스를 준수하시기 바랍니다.

---

## 👤 저자

**김용환 (YongHwan Kim)**
- 박사논문: AI 기반 한인 디아스포라 기록 유산 디지털화 및 실감형 콘텐츠 생성 모델 연구
- 2026
