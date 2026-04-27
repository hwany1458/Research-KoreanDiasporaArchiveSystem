# AI 기반 한인 디아스포라 기록유산 디지털화 시스템 — 이미지 처리 모듈

> 박사학위 논문 연구 — 한인 디아스포라 기록유산의 AI 기반 디지털화 및 실감형 콘텐츠 생성 시스템

본 저장소는 위 시스템의 **이미지 처리 모듈** 구현 코드를 포함합니다. 한인 디아스포라 아카이브(KADA, USC 등)의 흑백·노후 사진을 자동 분석·복원·컬러화하는 4단계 파이프라인을 제공합니다.

---

## 주요 기능

본 모듈은 입력 이미지에 다음 4단계 파이프라인을 자동 적용합니다.

| 단계 | 모듈 | 사용 모델 | 주요 역할 |
|------|------|-----------|-----------|
| 1 | 초해상도 복원 | Real-ESRGAN ×4 | 4배 업스케일링 및 디테일 복원 |
| 2 | 얼굴 향상 | GFPGAN v1.4 | 검출된 얼굴의 blind face restoration |
| 3 | 흑백 컬러화 | DDColor (ICCV 2023) | 흑백 사진의 사실적 컬러 복원 |
| 4 | 이미지 분석 | BLIP, CLIP, RetinaFace+HOG | 캡션, 장면 분류, 얼굴 검출 |

### 설계 특징

- **모듈 책임 분리**: SR과 얼굴 복원의 책임을 분리하여 처리 시간 75% 단축, 색상 편향 완화
- **이중 얼굴 검출기**: RetinaFace + HOG fallback으로 흑백·노후 사진의 robustness 확보
- **다중 컬러화 백엔드**: DDColor(주) → DeOldify(보조) → Sepia(fallback)의 graceful degradation
- **자동 흑백 판정**: 컬러 사진은 컬러화 자동 스킵, 원본 색상 충실도 보존
- **메모리 안전 임계값**: SR 결과 24MP 초과 시 자동 스킵으로 GPU OOM 방지
- **한글 경로 완전 지원**: Windows 한글 파일명/경로 안전 처리

---

## 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (CUDA 11.8 이상 호환), 최소 8GB VRAM 권장
- 본 시스템은 RTX 4080 (16GB VRAM) 환경에서 검증됨
- CPU 모드도 지원하나 처리 시간이 매우 길어짐 (수십 배)

### 소프트웨어
- **OS**: Windows 11 (검증됨), Linux/macOS도 동작 가능
- **Python**: 3.10 또는 3.11
- **CUDA**: 12.1 (검증됨)
- **PyTorch**: 2.1.x (cu121 빌드)

---

## 설치

상세 설치 가이드는 [`INSTALL.md`](./INSTALL.md)를 참고하세요. 아무것도 설치되지 않은 노트북에 처음부터 환경을 구축하는 방법이 단계별로 정리되어 있습니다.

### 빠른 설치 (이미 Python/CUDA가 준비된 경우)

```bash
# 1. conda 환경 생성
conda create -n diaspora python=3.11 -y
conda activate diaspora

# 2. PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3. 핵심 패키지
pip install realesrgan basicsr gfpgan facexlib face_recognition
pip install modelscope datasets simplejson addict sortedcontainers oss2 timm
pip install transformers opencv-python Pillow pyyaml

# 4. 가중치 다운로드 (weights/ 디렉토리에)
mkdir weights && cd weights
curl.exe -L -o RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
curl.exe -L -o RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
curl.exe -L -o GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
cd ..
# DDColor 가중치는 최초 실행 시 modelscope이 자동 다운로드
```

---

## 사용법

### 단일 이미지 처리

```bash
python main.py --input data/input/photo.jpg --output data/output/photo_restored.jpg --verbose
```

주요 옵션:
- `--no-sr`: 초해상도 비활성화
- `--no-face`: 얼굴 향상 비활성화
- `--no-color`: 컬러화 비활성화
- `--no-analysis`: 이미지 분석 비활성화
- `--device cpu`: CPU 모드 강제
- `--config config.yaml`: 설정 파일 사용

### 일괄 처리

```bash
# data/input/ 의 모든 이미지를 처리
python batch_process.py

# 처음 3장만 (테스트용)
python batch_process.py --limit 3

# 비교 이미지 생략 (속도 향상)
python batch_process.py --no-comparison
```

일괄 처리 결과는 `data/output/batch_<timestamp>/` 디렉토리에 다음 구조로 생성됩니다.

```
batch_20260427_225720/
├── restored/             # 복원된 결과 이미지
├── comparisons/          # 원본+결과 나란히 비교 이미지 (학위논문 figure용)
├── reports/              # 각 이미지 상세 JSON 리포트
├── batch_summary.json    # 전체 일괄 처리 요약 (JSON)
└── batch_summary.md      # 사람이 읽기 좋은 요약 (Markdown)
```

---

## 디렉토리 구조

```
03_diaspora_image_processing/
├── main.py                       # 단일 이미지 처리 진입점
├── batch_process.py              # 일괄 처리 스크립트
├── config.yaml                   # 처리 설정 (선택)
├── requirements.txt              # Python 의존성
├── README.md                     # 본 파일
├── INSTALL.md                    # 상세 설치 가이드
├── src/
│   ├── pipeline.py               # ImageRestorationPipeline (메인 파이프라인)
│   └── modules/
│       ├── super_resolution.py   # Real-ESRGAN 모듈
│       ├── face_enhancement.py   # GFPGAN + 이중 검출기 모듈
│       ├── colorization.py       # DDColor + 다중 백엔드 모듈
│       └── image_analysis.py     # BLIP/CLIP/face_recognition 분석 모듈
├── weights/                      # 모델 가중치 (.pth 파일)
└── data/
    ├── input/                    # 입력 이미지
    └── output/                   # 처리 결과
```

---

## 검증 데이터셋

본 시스템은 다음 공식 한인 디아스포라 아카이브의 자료로 검증되었습니다.

- **KADA** (Korean American Digital Archive) — UC Irvine 도서관
- **USC Korean American Digital Archive** — University of Southern California 도서관

두 아카이브는 1900년대 초 한인 미국 이민사를 기록한 대표적 공식 자료원입니다.

---

## 검증 결과 요약

RTX 4080 환경에서의 단계별 평균 처리 시간 (n=3, KADA/USC 표본 기준):

| 단계 | 평균 시간 | 비고 |
|------|-----------|------|
| Super-resolution | 3.29초 | tile_size=1024 기준 |
| Face enhancement | 9.33초 | upscale=1, 단체사진 시 증가 |
| Colorization | 0.79초 | DDColor의 GPU 가속 |
| Analysis | 3.97초 | BLIP+CLIP 추론 포함 |
| **총계** | **약 23.8초/장** | 모델 로딩 후 평균 |

처리 사례:
- KADA 단체사진(1200×844, 흑백) → 4800×3376, **14명 얼굴 검출**, 26.3초
- USC 집회사진(1200×845, 흑백) → 4800×3380, 5명 검출, 19.7초
- KADA 결혼사진(727×1200, 흑백) → 2908×4800, 2명 검출, 53.2초

---

## 알려진 한계

1. **GFPGAN의 데이터셋 편향**: FFHQ로 학습되어 동아시아인 얼굴 복원 시 일부 영역(특히 귀)에 색상 편향. 본 시스템은 `upscale=1` 정책으로 부분적으로 완화.
2. **표본 크기**: 현재 검증은 n=3 수준의 사례 분석. 통계적 유의성을 위한 30장 이상 표본 확장은 향후 작업.
3. **RetinaFace 라이브러리 호환성**: PyTorch 2.x 환경에서 facexlib 내부 오류 발생. HOG fallback으로 우회 처리됨.

---

## 인용

본 시스템에서 사용된 주요 모델의 인용:

- **Real-ESRGAN**: Wang, X., et al. (2021). *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*. ICCVW.
- **GFPGAN**: Wang, X., et al. (2021). *Towards Real-World Blind Face Restoration with Generative Facial Prior*. CVPR.
- **DDColor**: Kang, X., et al. (2023). *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders*. ICCV.
- **RetinaFace**: Deng, J., et al. (2020). *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild*. CVPR.
- **HOG**: Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.

---

## 라이선스

본 저장소의 코드는 학술 연구 목적으로 공개됩니다. 사용된 외부 모델·라이브러리의 라이선스는 각 프로젝트의 정책을 따릅니다.

---

## 작성자

YongHwan Kim (이용환) — 박사학위 연구  
2026
