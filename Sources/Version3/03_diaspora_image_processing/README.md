# Diaspora Image Restoration Pipeline

> 한인 디아스포라 기록유산을 위한 AI 기반 이미지 복원 파이프라인  
> AI-based Image Restoration Pipeline for Korean Diaspora Archival Heritage

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

흑백 가족사진 등 디아스포라 아카이브 이미지를 4단계 AI 파이프라인 (초해상도 → 얼굴 향상 → 컬러화 → 의미 분석) 으로 복원합니다. 8GB VRAM 노트북에서도 동작 가능하도록 모듈별 lazy load · 입력 크기 정규화 · LAB 공간 색상 합성 등의 메모리 최적화 기법을 적용했습니다.

## ✨ 주요 특징

- **4개 SOTA AI 모듈 통합**: Real-ESRGAN (SR) · GFPGAN (Face) · DDColor (Color) · BLIP+CLIP (Analysis)
- **8GB VRAM 노트북 호환**: lazy load + unload, 입력 크기 정규화, expandable_segments
- **모듈 실행 순서 ablation 지원**: `--pipeline-order` CLI 옵션으로 5가지 순열 실험 가능
- **한인 디아스포라 컨텍스트 인식**: 20개 한국 카테고리 (제사 · 한복 · 명절 등) 장면 분류
- **재현성 강화**: LAB 공간 색상 합성으로 디테일 무손실 보존, 단계별 중간 결과 저장
- **HOG 검출기 fallback**: RetinaFace 실패 시 자동 전환

## 🎬 빠른 시작

```bash
# 1. 환경 설정 (자세한 설치는 INSTALL.md 참고)
conda create -n diaspora python=3.11 -y
conda activate diaspora
pip install -r requirements.txt

# 2. 가중치 다운로드 (한 번만)
mkdir weights
# RealESRGAN_x4plus.pth, GFPGANv1.4.pth 다운로드 (INSTALL.md 참고)

# 3. 실행
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_restored.jpg \
    --low-vram --vram-verbose --save-intermediate
```

## 📊 성능 (RTX 4070 Laptop, 8GB VRAM 기준)

| 입력 크기 | 처리 시간 | 출력 크기 | VRAM 사용량 |
|---|---|---|---|
| 1200 × 844 | 47 ~ 60초 | 4800 × 3376 | < 1.5 GB |
| 800 × 600 | 25 ~ 35초 | 3200 × 2400 | < 1.2 GB |

자세한 벤치마크는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고.

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  입력: 흑백 가족사진 (예: 1200 × 844 JPEG)                    │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Super Resolution (Real-ESRGAN x4plus)                     │
│     1200 × 844 → 4800 × 3376                                  │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Face Enhancement (GFPGAN v1.4 + RetinaFace + HOG)         │
│     얼굴 14개 검출 · 512×512 정렬 후 복원 · paste-back        │
│     ※ 1536px 다운스케일 후 처리하여 OOM 회피                    │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Colorization (DDColor, ICCV 2023)                         │
│     1536px 다운스케일 → DDColor → LAB 공간에서 ab 채널만      │
│     원본 해상도로 업스케일 후 원본 L 채널과 합성               │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Analysis (BLIP-large + CLIP-large + face_recognition)     │
│     캡션 생성 · 20개 한국 카테고리 장면 분류 · 얼굴 인코딩     │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  출력: 4800 × 3376 컬러 이미지 + 메타데이터                   │
└──────────────────────────────────────────────────────────────┘
```

`--pipeline-order` 옵션으로 1·2·3 단계의 순서를 변경할 수 있습니다 (4번 분석은 항상 마지막). 5가지 순열의 ablation 결과는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고.

## 📁 프로젝트 구조

```
diaspora-image-pipeline/
├── README.md                      # 이 파일
├── INSTALL.md                     # 상세 설치 가이드
├── EXPERIMENTS.md                 # 8GB 노트북 검증 결과 + ablation
├── LICENSE                        # MIT License
├── CITATION.cff                   # 학술 인용 정보
├── requirements.txt               # Python 의존성
├── config.yaml                    # 설정 파일 예시
├── .gitignore                     # 가중치 · 데이터 제외
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                # 메인 파이프라인 + CLI
│   └── modules/
│       ├── __init__.py
│       ├── super_resolution.py    # Real-ESRGAN
│       ├── face_enhancement.py    # GFPGAN + 1536px 패치
│       ├── colorization.py        # DDColor + LAB 합성 패치
│       └── image_analysis.py      # BLIP + CLIP + face_recognition
│
├── weights/                       # 모델 가중치 (별도 다운로드)
│   └── README.md                  # 다운로드 가이드만
│
├── data/
│   ├── input/                     # 입력 이미지 (gitignore)
│   └── output/                    # 출력 결과 (gitignore)
│
└── docs/
    └── system_flow_diagram.html   # 시각적 시스템 다이어그램
```

## 🔬 사용 예시

### 기본 사용 (default 순서: SR → Face → Color → Analysis)

```bash
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_restored.jpg \
    --low-vram --save-intermediate
```

### 권장 ablation 순서 (Color → SR → Face)

연구·실험 결과 가장 균형 잡힌 결과를 내는 순서:

```bash
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionE.jpg \
    --pipeline-order "color,sr,face" \
    --low-vram --save-intermediate
```

### config.yaml 사용

```bash
python -m src.pipeline --config config.yaml \
    --input data/input/photo.jpg \
    --output data/output/photo_config.jpg
```

### 일부 단계만 실행

```bash
# SR 끄고 Color + Face만 (1200x844 결과, 색상 가장 자연스러움)
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_no_sr.jpg \
    --no-sr --pipeline-order "color,face"
```

자세한 옵션은 `python -m src.pipeline --help` 또는 [INSTALL.md](INSTALL.md) 참고.

## 🔍 핵심 기여

본 시스템은 다중 AI 모듈을 단순 직렬 연결할 때 발생하는 누적 오차를 다음 세 가지 메커니즘으로 해결합니다:

1. **모듈별 입력 정규화**: face/color 모듈에 `max_input_dim=1536` 옵션을 도입해 GFPGAN의 RetinaFace OOM과 DDColor의 색상 cast를 동시에 완화
2. **LAB 공간 색상 합성**: 컬러화 결과의 ab(색상) 채널만 사용하고 L(휘도) 채널은 SR 결과를 보존하여 디테일 무손실
3. **모듈 순서 ablation 시스템**: `--pipeline-order` CLI 옵션으로 5가지 순열을 1줄 변경으로 실험 가능

8GB VRAM 노트북 환경에서 5회 ablation 실험을 통해 위 메커니즘의 효과를 정량 검증했습니다 ([EXPERIMENTS.md](EXPERIMENTS.md)).

## 📖 인용

본 시스템 또는 코드를 학술 연구에 사용한 경우 다음을 인용해 주세요:

```bibtex
@phdthesis{kim2026diaspora,
  title={한인 디아스포라 기록유산을 위한 AI 기반 디지털화 및 몰입형 콘텐츠 생성 시스템},
  author={[저자명]},
  year={2026},
  school={[소속 대학]},
  type={박사학위논문}
}
```

(학위 취득 후 정식 논문이 게재되면 BibTeX를 업데이트합니다)

## 📚 참고 문헌

핵심 모델들의 원논문:

- **Real-ESRGAN**: Wang, X. et al. (2021). "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data." *ICCVW*.
- **GFPGAN**: Wang, X. et al. (2021). "Towards Real-World Blind Face Restoration with Generative Facial Prior." *CVPR*.
- **DDColor**: Kang, X. et al. (2023). "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders." *ICCV*.
- **BLIP**: Li, J. et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." *ICML*.
- **CLIP**: Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.

## 📝 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참고. 단, 본 시스템이 사용하는 외부 모델(Real-ESRGAN, GFPGAN, DDColor, BLIP, CLIP)들은 각 모델의 라이선스를 따릅니다.

## 🙏 감사의 말

본 시스템은 다음 오픈소스 프로젝트 위에 구축되었습니다:
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)
- [piddnad/DDColor](https://github.com/piddnad/DDColor)
- [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- [salesforce/BLIP](https://github.com/salesforce/BLIP)
- [openai/CLIP](https://github.com/openai/CLIP)

## 📮 문의

이슈 및 질문은 [GitHub Issues](../../issues)로 등록해 주세요.
