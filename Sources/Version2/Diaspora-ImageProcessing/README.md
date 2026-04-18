# AI 기반 한인 디아스포라 기록 유산 디지털화 시스템
## Image Processing Module

> **박사학위 논문 구현 코드**  
> "AI 기반 한인 디아스포라 기록 유산 디지털화 및 실감형 콘텐츠 생성 모델 연구"  
> 원광대학교 게임콘텐츠학과

[![Python](https://img.shields.io/badge/Python-3.10.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2+cu121-ee4c2c)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 개요

본 모듈은 한인 디아스포라 관련 흑백/저화질 사진 자료를 AI 기반으로 자동 복원하고
Dublin Core 표준 메타데이터를 생성하는 이미지 처리 파이프라인입니다.

### 처리 파이프라인

```
입력 이미지 (흑백/저화질 역사 사진)
    │
    ▼
[1단계] 초해상도 복원 ── Real-ESRGAN (x4 업스케일)
    │                    812×1200 → 3248×4800
    ▼
[2단계] 얼굴 향상 ────── GFPGAN v1.4
    │                    RetinaFace 기반 얼굴 감지 및 복원
    ▼
[3단계] 흑백 컬러화 ──── DDColor (ICCV 2023)
    │                    HuggingFace: piddnad/ddcolor_modelscope
    ▼
[4단계] 이미지 분석 ──── BLIP (캡션) + CLIP (장면분류) + face_recognition
    │                    Dublin Core 메타데이터 자동 생성
    ▼
출력: 고해상도 컬러 이미지 + JSON 메타데이터
```

### 처리 결과 예시

| 항목 | 결과 |
|------|------|
| 원본 크기 | 812 × 1200 px |
| 처리 후 크기 | 3248 × 4800 px |
| BLIP 캡션 | "there is a man sitting on a bench in front of a building" |
| CLIP 장면 분류 | portrait photo (85.6%) |
| 감지된 얼굴 | 1명 |
| 처리 시간 | 약 19~62초/장 (GPU) |

---

## 🛠️ 기술 스택

| 모듈 | 기술 | 버전/모델 |
|------|------|----------|
| 초해상도 | Real-ESRGAN | RealESRGAN_x4plus |
| 얼굴 향상 | GFPGAN | GFPGANv1.4 |
| 흑백 컬러화 | DDColor | piddnad/ddcolor_modelscope |
| 이미지 캡션 | BLIP | Salesforce/blip-image-captioning-large |
| 장면 분류 | CLIP | openai/clip-vit-large-patch14 |
| 얼굴 감지 | face_recognition | HOG 모델 |
| 메타데이터 | Dublin Core JSON | --report 옵션 |

---

## 📁 프로젝트 구조

```
diaspora_image_processing/
├── main.py                      # CLI 진입점
├── config.yaml                  # 설정 파일
├── requirements.txt             # 패키지 목록
├── download_models.py           # 모델 다운로드 스크립트
├── test_pipeline.py             # 파이프라인 테스트
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # 통합 파이프라인 (핵심)
│   └── modules/
│       ├── __init__.py
│       ├── super_resolution.py  # Real-ESRGAN 초해상도
│       ├── face_enhancement.py  # GFPGAN 얼굴 향상
│       ├── colorization.py      # DDColor 컬러화
│       └── image_analysis.py    # BLIP/CLIP/face_recognition 분석
│
├── data/
│   ├── input/                   # 입력 이미지 (gitignored)
│   └── output/                  # 출력 결과 (gitignored)
│
├── models/                      # 사전학습 모델 (gitignored)
│   ├── realesrgan/
│   │   ├── RealESRGAN_x4plus.pth
│   │   └── RealESRGAN_x2plus.pth
│   ├── gfpgan/
│   │   └── GFPGANv1.4.pth
│   └── facexlib/
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
│
└── logs/                        # 처리 로그
```

---

## ⚙️ 설치 방법

### 환경 요구사항

- **OS**: Windows 10/11 (Linux/Mac도 가능)
- **Python**: 3.10.x (3.11+ 미검증)
- **GPU**: CUDA 12.1 지원 NVIDIA GPU (RTX 계열 권장)
- **VRAM**: 최소 6GB (8GB+ 권장)
- **RAM**: 최소 16GB

---

### Step 1: 저장소 클론 및 가상환경 생성

```powershell
git clone https://github.com/[YOUR_USERNAME]/diaspora-image-processing.git
cd diaspora-image-processing

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

---

### Step 2: PyTorch 설치 (CUDA 12.1)

```powershell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **중요**: torch를 먼저 설치해야 합니다. 다른 패키지가 numpy를 최신 버전으로 올릴 수 있습니다.

---

### Step 3: numpy 고정 (필수)

```powershell
pip install "numpy==1.26.4" --force-reinstall
```

---

### Step 4: 핵심 패키지 설치

```powershell
# basicsr, realesrgan은 반드시 --no-deps로 설치
pip install basicsr==1.4.2 --no-deps
pip install realesrgan==0.3.0 --no-deps

# GFPGAN
pip install gfpgan>=1.3.8

# transformers (5.x 사용 불가 - torch 2.4+ 요구)
pip install transformers==4.37.2 tokenizers==0.15.2 "huggingface-hub==0.20.3" --no-deps

# 나머지 패키지
pip install face_recognition Pillow opencv-python tqdm requests pyyaml colorama scikit-image
```

---

### Step 5: DDColor 설치 (컬러화 모듈)

DDColor는 pip 패키지가 없어 GitHub에서 직접 설치합니다.

```powershell
# 1. 소스 클론
git clone https://github.com/piddnad/DDColor.git ddcolor_src

# 2. basicsr 아키텍처 파일 복사 (필수!)
$src = "ddcolor_src\basicsr\archs"
$dst = "venv\Lib\site-packages\basicsr\archs"
Copy-Item "$src\ddcolor_arch.py" "$dst\ddcolor_arch.py" -Force
Copy-Item "$src\ddcolor_arch_utils" "$dst\ddcolor_arch_utils" -Recurse -Force
Copy-Item "$src\vgg_arch.py" "$dst\vgg_arch.py" -Force

# 3. 설치
cd ddcolor_src
pip install -e . --no-build-isolation
cd ..

# 4. numpy, basicsr 재고정 (DDColor가 버전을 바꿔버림)
pip install "numpy==1.26.4" --force-reinstall
pip install basicsr==1.4.2 --no-deps
pip install realesrgan==0.3.0 --no-deps
```

---

### Step 6: 모델 다운로드

```powershell
python download_models.py --all
```

다운로드되는 모델:

| 모델 | 크기 | 저장 위치 |
|------|------|----------|
| RealESRGAN_x4plus.pth | 67MB | models/realesrgan/ |
| RealESRGAN_x2plus.pth | 67MB | models/realesrgan/ |
| GFPGANv1.4.pth | 348MB | models/gfpgan/ |
| detection_Resnet50_Final.pth | 109MB | models/facexlib/ |
| parsing_parsenet.pth | 85MB | models/facexlib/ |
| ddcolor_modelscope (HuggingFace) | 912MB | ~/.cache/huggingface/ |
| blip-image-captioning-large (HF) | 1.88GB | ~/.cache/huggingface/ |
| clip-vit-large-patch14 (HF) | 1.71GB | ~/.cache/huggingface/ |

> HuggingFace 모델은 첫 실행 시 자동 다운로드됩니다.

---

### Step 7: 설치 확인

```powershell
python test_pipeline.py --test-imports
```

아래와 같이 나오면 성공:
```
✓ PyTorch 2.1.2+cu121
✓ CUDA 사용 가능
✓ Real-ESRGAN
✓ GFPGAN
✓ Transformers (BLIP, CLIP)
✓ face_recognition
```

---

## 🚀 사용법

### 기본 실행 (단일 이미지)

```powershell
python main.py --input data\input\photo.jpg --output data\output\result.jpg
```

### 메타데이터 JSON 포함

```powershell
python main.py \
  --input data\input\photo.jpg \
  --output data\output\result.jpg \
  --report data\output\metadata.json
```

### 배치 처리 (폴더 전체)

```powershell
python main.py \
  --input data\input \
  --output data\output\batch \
  --batch \
  --report data\output\batch_report.json
```

### 중간 단계 결과 저장

```powershell
python main.py \
  --input data\input\photo.jpg \
  --output data\output\result.jpg \
  --save-intermediate
```

중간 결과는 `data\output\result_stages\` 에 저장됩니다:
```
result_stages/
├── 01_super_resolution.jpg
├── 02_face_enhancement.jpg
└── 03_colorization.jpg
```

### 특정 단계 비활성화

```powershell
# 초해상도 없이
python main.py --input ... --output ... --no-sr

# 컬러화 없이
python main.py --input ... --output ... --no-color

# 분석 없이 (빠른 처리)
python main.py --input ... --output ... --no-analysis
```

---

## 📊 출력 형식

### 이미지

- 포맷: JPEG (quality=95)
- 해상도: 원본 × 4배 (Real-ESRGAN x4plus 기준)

### JSON 메타데이터

```json
{
  "generated_at": "2026-04-18T06:30:33",
  "total_images": 1,
  "successful": 1,
  "failed": 0,
  "results": [
    {
      "input": "data/input/photo.jpg",
      "output": "data/output/result.jpg",
      "success": true,
      "original_size": [812, 1200],
      "final_size": [3248, 4800],
      "stages_applied": [
        "super_resolution",
        "face_enhancement",
        "colorization",
        "analysis"
      ],
      "caption": "there is a man sitting on a bench in front of a building",
      "scenes": [["portrait photo", 0.856]],
      "face_count": 1,
      "metadata": {
        "dc:type": "Image",
        "dc:format": "image/jpeg",
        "dc:created": "2026-04-18T06:30:32",
        "dc:description": "there is a man sitting on a bench...",
        "dc:subject": ["portrait photo"]
      },
      "total_time": 19.45,
      "stage_times": {
        "super_resolution": 1.65,
        "face_enhancement": 2.04,
        "colorization": 1.43,
        "analysis": 5.89
      }
    }
  ]
}
```

---

## ⚠️ 알려진 이슈 및 해결책

### numpy 버전 충돌
```
RuntimeError: Numpy is not available
```
**해결**: `pip install numpy==1.26.4 --force-reinstall`

### Real-ESRGAN import 실패
```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```
**해결**: torchvision 버전 확인 후 `pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121`

### DDColor 모델 로드 실패
```
No module named 'basicsr.archs.ddcolor_arch_utils'
```
**해결**: Step 5의 basicsr 아키텍처 파일 복사 단계 재실행

### BLIP/CLIP 초기화 실패
```
BlipForConditionalGeneration requires PyTorch >= 2.4
```
**해결**: `pip install transformers==4.37.2 tokenizers==0.15.2 "huggingface-hub==0.20.3" --no-deps`

### 얼굴 감지 실패 (고해상도)
고해상도 이미지(4000px 이상)에서 얼굴 감지가 실패하는 경우, `face_enhancement.py`의 `max_size` 파라미터가 자동으로 리사이즈를 처리합니다. (기본값: 2048px)

---

## 📦 확정 패키지 버전

재현 가능한 환경을 위한 확정 버전입니다.

| 패키지 | 버전 | 설치 방법 |
|--------|------|----------|
| torch | 2.1.2+cu121 | `--index-url .../cu121` |
| torchvision | 0.16.2+cu121 | `--index-url .../cu121` |
| numpy | 1.26.4 | `--force-reinstall` |
| basicsr | 1.4.2 | `--no-deps` |
| realesrgan | 0.3.0 | `--no-deps` |
| gfpgan | 1.3.8 | 일반 설치 |
| transformers | 4.37.2 | `--no-deps` |
| tokenizers | 0.15.2 | `--no-deps` |
| huggingface-hub | 0.20.3 | `--no-deps` |

---

## 📖 논문 참고문헌

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{wang2021realesrgan,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={ICCVW},
  year={2021}
}

@inproceedings{wang2021gfpgan,
  title={Towards Real-World Blind Face Restoration with Generative Facial Prior},
  author={Wang, Xintao and Li, Yu and Zhang, Honglun and Shan, Ying},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={ICML},
  year={2022}
}

@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  booktitle={ICML},
  year={2021}
}
```

---

## 🔬 개발 환경

- **OS**: Windows 11
- **Python**: 3.10.11 (pyenv-win)
- **GPU**: NVIDIA (CUDA 12.1)
- **IDE**: VS Code

---

## 📄 라이선스

본 코드는 학술 연구 목적으로 작성되었습니다.  
각 AI 모델의 라이선스는 원 저작자를 따릅니다:
- Real-ESRGAN: BSD 3-Clause
- GFPGAN: Apache 2.0
- DDColor: Apache 2.0
- BLIP: BSD 3-Clause
- CLIP: MIT

---

## 👤 저자

**김용환 (YongHwan Kim)**  
원광대학교 게임콘텐츠학과  
2026
