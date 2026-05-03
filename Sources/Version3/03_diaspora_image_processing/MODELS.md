# 모델 파일 안내 (Models Guide)

> 본 시스템이 사용하는 모든 사전 훈련된 모델 가중치 파일에 대한 안내입니다.
> **GitHub repo에는 모델 파일이 포함되어 있지 않습니다.** 본 문서를 참고하여
> 사용자가 직접 다운로드하거나 자동 다운로드를 통해 확보해야 합니다.

## 목차

- [1. 개요](#1-개요)
- [2. 모델 파일 위치 — 3개 폴더](#2-모델-파일-위치--3개-폴더)
- [3. weights/ — 사용자 직접 다운로드](#3-weights--사용자-직접-다운로드)
- [4. gfpgan/weights/ — facexlib 자동 다운로드](#4-gfpganweights--facexlib-자동-다운로드)
- [5. models/ — DeOldify 자동 다운로드 (선택)](#5-models--deoldify-자동-다운로드-선택)
- [6. 시스템 캐시 — 라이브러리 자동 관리](#6-시스템-캐시--라이브러리-자동-관리)
- [7. 다운로드 검증 및 트러블슈팅](#7-다운로드-검증-및-트러블슈팅)
- [8. 모델 라이선스 정보](#8-모델-라이선스-정보)

---

## 1. 개요

### 모델 파일이 GitHub에 없는 이유

1. **파일 크기**: 핵심 가중치 합계 약 400MB (GitHub 단일 파일 100MB 제한 초과)
2. **저작권**: 각 모델은 원 저자/단체의 라이선스를 따르며, 재배포는 라이선스를 별도로 검토해야 함
3. **표준 관행**: 학술 코드 공개 시 가중치는 별도 다운로드로 안내하는 것이 일반적

### 본 시스템이 사용하는 모델 (총 7개)

| # | 모델 | 역할 | 파일명 | 크기 | 다운로드 방식 |
|---|---|---|---|---|---|
| 1 | Real-ESRGAN x4plus | 초해상도 (4배) | `RealESRGAN_x4plus.pth` | 64 MB | 사용자 직접 |
| 2 | GFPGAN v1.4 | 얼굴 향상 | `GFPGANv1.4.pth` | 348 MB | 사용자 직접 |
| 3 | Real-ESRGAN x2plus | 배경 업샘플러 (선택) | `RealESRGAN_x2plus.pth` | 64 MB | 사용자 직접 (선택) |
| 4 | RetinaFace ResNet50 | 얼굴 검출 | `detection_Resnet50_Final.pth` | 110 MB | facexlib 자동 |
| 5 | Parsing Parsenet | 얼굴 파싱 | `parsing_parsenet.pth` | 85 MB | facexlib 자동 |
| 6 | DDColor | 컬러화 (주 백엔드) | `pytorch_model.pt` | 870 MB | modelscope 자동 |
| 7 | BLIP-large | 캡션 생성 | (HF Hub) | 1.8 GB | transformers 자동 |
| 8 | CLIP ViT-Large/14 | 장면 분류 | (HF Hub) | 1.7 GB | transformers 자동 |
| 9 | face_recognition_models | 얼굴 인코딩 | (패키지 내장) | 100 MB | pip 설치 시 자동 |

### 다운로드 합계

- **사용자가 직접 다운로드**: 약 412 MB (필수 2개) ~ 476 MB (선택 포함)
- **첫 실행 시 자동 다운로드**: 약 4.6 GB
- **총 디스크 사용량**: 약 5.0 GB

---

## 2. 모델 파일 위치 — 3개 폴더

본 시스템 실행 시 모델 파일은 **3개의 다른 위치**에 분산됩니다. 이는 각 라이브러리가 자체적으로 강제하는 경로 때문이며, 단일 폴더로 통합하기 어려운 구조적 이유가 있습니다.

### 폴더 구조

```
03_diaspora_image_processing/
├── weights/                      # 사용자 관리
│   ├── README.md                 # 다운로드 가이드
│   ├── RealESRGAN_x4plus.pth     # 사용자 직접 다운로드
│   ├── GFPGANv1.4.pth            # 사용자 직접 다운로드
│   └── RealESRGAN_x2plus.pth     # (선택) 사용자 직접 다운로드
│
├── gfpgan/                       # facexlib 자동 관리
│   └── weights/
│       ├── detection_Resnet50_Final.pth   # 자동 다운로드
│       └── parsing_parsenet.pth           # 자동 다운로드
│
└── models/                       # DeOldify 자동 관리 (선택)
    └── ColorizeArtistic_gen.pth  # DeOldify 사용 시만 (선택)
```

### 폴더별 관리 주체

| 폴더 | 관리 주체 | 본 시스템 코드와의 관계 |
|---|---|---|
| `weights/` | **사용자** | 본 코드의 `_resolve_weight_path()`가 직접 참조 |
| `gfpgan/` | **facexlib 라이브러리** | 본 코드는 직접 다루지 않음, GFPGAN 라이브러리가 자체 관리 |
| `models/` | **DeOldify 라이브러리** | 본 코드는 직접 다루지 않음, DeOldify가 자체 관리 |

> 💡 **중요**: `gfpgan/`과 `models/` 폴더의 위치는 라이브러리가 작업 디렉토리(CWD) 기준으로 자동 생성합니다. 즉 프로젝트 루트(03_diaspora_image_processing/)에서 파이프라인을 실행해야 이 위치에 생성됩니다.

### 시스템 캐시 폴더 (별도 위치)

위 3개 외에도 다음 시스템 캐시 폴더에 모델이 다운로드됩니다:

```
~/.cache/
├── modelscope/hub/damo/cv_ddcolor_image-colorization/   # DDColor
└── huggingface/hub/                                      # BLIP, CLIP
    ├── models--Salesforce--blip-image-captioning-large/
    └── models--openai--clip-vit-large-patch14/
```

(Windows: `C:\Users\<USER>\.cache\`, Linux/macOS: `~/.cache/`)

---

## 3. weights/ — 사용자 직접 다운로드

### 3.1 RealESRGAN_x4plus.pth (필수)

**역할**: 4배 초해상도. 1200×844 → 4800×3376 변환의 핵심.

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 64 MB (정확히 67,040 KB) |
| 저장 경로 | `weights/RealESRGAN_x4plus.pth` |
| 파일 형식 | PyTorch state_dict (`.pth`) |
| 모델 아키텍처 | RRDBNet (num_block=23, num_feat=64, scale=4) |
| 학습 데이터 | DF2K (DIV2K + Flickr2K) + OST |
| 발표 | Wang et al., *ICCVW 2021* |

**다운로드 URL**:
```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

**Windows PowerShell**:
```powershell
Invoke-WebRequest `
    -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" `
    -OutFile "weights\RealESRGAN_x4plus.pth"
```

**Linux/macOS**:
```bash
wget -P weights/ \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

**확인 명령**:
```bash
# 파일 크기 확인 (Windows)
Get-Item weights\RealESRGAN_x4plus.pth | Select-Object Name, Length

# 파일 크기 확인 (Linux/macOS)
ls -lh weights/RealESRGAN_x4plus.pth
```

### 3.2 GFPGANv1.4.pth (필수)

**역할**: 얼굴 영역 정밀 복원. 14명 얼굴까지 안정 처리 가능.

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 348 MB (정확히 333 MB) |
| 저장 경로 | `weights/GFPGANv1.4.pth` |
| 파일 형식 | PyTorch state_dict (`.pth`) |
| 모델 아키텍처 | StyleGAN2 prior + degradation removal |
| 학습 데이터 | FFHQ (Flickr-Faces-HQ) |
| 발표 | Wang et al., *CVPR 2021* |

**다운로드 URL**:
```
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

**Windows PowerShell**:
```powershell
Invoke-WebRequest `
    -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" `
    -OutFile "weights\GFPGANv1.4.pth"
```

**Linux/macOS**:
```bash
wget -P weights/ \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

> 💡 **버전 선택**: 본 코드는 v1.4를 default로 사용합니다. v1.3 (GFPGANv1.3.pth)도 지원하지만, v1.4가 더 자연스러운 결과를 냅니다. 두 버전 모두 동일한 URL 패턴으로 다운로드 가능 (파일명만 변경).

### 3.3 RealESRGAN_x2plus.pth (선택)

**역할**: 얼굴 향상의 배경 업샘플러용. 본 시스템 default 설정에서는 사용되지 않음.

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 64 MB |
| 저장 경로 | `weights/RealESRGAN_x2plus.pth` |
| 사용 시기 | `face_enhancement.py`에서 `bg_upsampler='realesrgan'` 활성화 시 |

**다운로드 URL**:
```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

**언제 필요한가**: `FaceEnhancementModule(bg_upsampler='realesrgan')`로 초기화할 때만. 현재 본 시스템은 SR을 별도 단계로 처리하므로 이 옵션이 비활성화되어 있어 **다운로드하지 않아도 됩니다**.

---

## 4. gfpgan/weights/ — facexlib 자동 다운로드

### 4.1 detection_Resnet50_Final.pth (자동)

**역할**: GFPGAN 내부의 얼굴 검출(RetinaFace).

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 110 MB |
| 저장 경로 | `gfpgan/weights/detection_Resnet50_Final.pth` |
| 다운로드 시점 | 첫 실행 시 GFPGAN이 facexlib을 통해 자동 다운로드 |
| 출처 | facexlib 라이브러리 (Tencent ARC Lab) |

### 4.2 parsing_parsenet.pth (자동)

**역할**: 얼굴 영역 파싱(눈/코/입/머리카락 등 분할).

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 85 MB |
| 저장 경로 | `gfpgan/weights/parsing_parsenet.pth` |
| 다운로드 시점 | 첫 실행 시 자동 다운로드 |
| 출처 | facexlib 라이브러리 |

### 4.3 자동 다운로드 작동 방식

GFPGAN을 처음 호출할 때 다음 동작이 자동으로 일어납니다:

```
1. 본 시스템이 FaceEnhancementModule.enhance() 호출
2. GFPGAN이 facexlib.detection.RetinaFace 인스턴스화
3. facexlib이 gfpgan/weights/ 폴더 검사
4. 파일이 없으면 GitHub Releases에서 자동 다운로드
5. 다운로드 완료 후 정상 동작
```

이 과정은 **첫 실행 시에만** 일어나며, 그 후에는 캐시된 파일을 사용합니다.

> ⚠️ **경로 주의**: `gfpgan/` 폴더는 facexlib이 **현재 작업 디렉토리(CWD)** 를 기준으로 생성합니다. 따라서 `python -m src.pipeline ...` 실행 시 반드시 프로젝트 루트(`03_diaspora_image_processing/`)에서 실행해야 일관된 위치에 생성됩니다.

### 4.4 수동 다운로드 (인터넷 차단 환경용)

회사망 등 인터넷 접속이 제한된 환경에서는 수동 다운로드가 필요할 수 있습니다:

```
detection_Resnet50_Final.pth:
https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth

parsing_parsenet.pth:
https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
```

다운로드 후 `gfpgan/weights/` 폴더에 직접 배치하면 자동 다운로드가 스킵됩니다.

---

## 5. models/ — DeOldify 자동 다운로드 (선택)

> ⚠️ **본 시스템의 default 백엔드는 DDColor입니다.** DeOldify는 보조 백엔드로, 명시적으로 활성화한 경우에만 다운로드됩니다.

### 5.1 ColorizeArtistic_gen.pth (선택)

**역할**: DeOldify의 흑백 컬러화 (Artistic 모드).

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 800 MB |
| 저장 경로 | `models/ColorizeArtistic_gen.pth` |
| 다운로드 시점 | DeOldify 백엔드 활성화 시 첫 호출에서 자동 다운로드 |
| 발표 | Antic, *DeOldify 2019* |

**활성화 방법** (DeOldify 사용 시):

```python
from src.modules.colorization import ColorizationModule
module = ColorizationModule(device='cuda', backend='deoldify')
```

또는 `config.yaml`:
```yaml
colorization:
  backend: "deoldify"
```

### 5.2 ColorizeStable_gen.pth (선택, 더 보조적)

**역할**: DeOldify의 Stable 모드 (보수적 컬러화).

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 800 MB |
| 저장 경로 | `models/ColorizeStable_gen.pth` |
| 활성화 | `model_type='stable'` 설정 시 |

### 5.3 수동 다운로드 URL (필요시)

```
ColorizeArtistic_gen.pth:
https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth

ColorizeStable_gen.pth:
https://data.deepai.org/deoldify/ColorizeStable_gen.pth
```

> 💡 **DeOldify는 별도 설치 필요**: DeOldify 라이브러리 자체가 본 시스템의 requirements.txt에 포함되지 않습니다. `pip install deoldify` 별도 실행 필요. 또한 fastai 1.x 의존성으로 환경 충돌 위험이 있어 권장하지 않습니다.

---

## 6. 시스템 캐시 — 라이브러리 자동 관리

다음 모델들은 시스템의 사용자 캐시 폴더에 자동 저장됩니다. **본 시스템 폴더 외부**이므로 별도 관리가 필요 없으며, 디스크 정리 시 주의해야 합니다.

### 6.1 DDColor (modelscope 캐시)

| 항목 | 값 |
|---|---|
| 모델 ID | `damo/cv_ddcolor_image-colorization` |
| 파일 크기 | 약 870 MB |
| 저장 경로 (Windows) | `C:\Users\<USER>\.cache\modelscope\hub\damo\cv_ddcolor_image-colorization\` |
| 저장 경로 (Linux/macOS) | `~/.cache/modelscope/hub/damo/cv_ddcolor_image-colorization/` |
| 주요 파일 | `pytorch_model.pt`, `configuration.json` |
| 발표 | Kang et al., *ICCV 2023* |

다운로드는 첫 실행 시 자동으로 일어나며, modelscope 라이브러리가 관리합니다.

### 6.2 BLIP-large (HuggingFace 캐시)

| 항목 | 값 |
|---|---|
| 모델 ID | `Salesforce/blip-image-captioning-large` |
| 파일 크기 | 약 1.8 GB |
| 저장 경로 (Windows) | `C:\Users\<USER>\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-large\` |
| 저장 경로 (Linux/macOS) | `~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/` |
| 발표 | Li et al., *ICML 2022* |

### 6.3 CLIP ViT-Large/14 (HuggingFace 캐시)

| 항목 | 값 |
|---|---|
| 모델 ID | `openai/clip-vit-large-patch14` |
| 파일 크기 | 약 1.7 GB |
| 저장 경로 | `~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/` |
| 발표 | Radford et al., *ICML 2021* |

### 6.4 face_recognition_models (패키지 내장)

| 항목 | 값 |
|---|---|
| 파일 크기 | 약 100 MB |
| 저장 경로 | `<conda_env>/Lib/site-packages/face_recognition_models/models/` |
| 다운로드 시점 | `pip install face_recognition_models` 실행 시 |
| 주요 파일 | `dlib_face_recognition_resnet_model_v1.dat`, `shape_predictor_*.dat` |

### 6.5 캐시 관리 팁

**캐시 위치 변경**: HuggingFace 캐시 위치를 다른 드라이브로 옮기려면:

```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\hf_cache', 'User')
```

```bash
# Linux/macOS
export HF_HOME="/path/to/cache"
```

**캐시 정리**: 디스크 공간이 부족하면:

```powershell
# Windows — 전체 HF 캐시 제거 (다음 실행 시 재다운로드됨)
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\modelscope
```

---

## 7. 다운로드 검증 및 트러블슈팅

### 7.1 전체 검증 스크립트

다음 스크립트로 모든 모델 파일이 올바르게 배치됐는지 확인할 수 있습니다:

```python
"""check_models.py — 모델 파일 검증 스크립트"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
HOME = Path.home()

# 1. weights/ — 사용자 직접 다운로드
print("=" * 60)
print("1. weights/ (사용자 관리)")
print("=" * 60)
required = {
    'RealESRGAN_x4plus.pth': 64 * 1024**2,    # 64 MB
    'GFPGANv1.4.pth':         348 * 1024**2,  # 348 MB
}
optional = {
    'RealESRGAN_x2plus.pth':  64 * 1024**2,
}
for name, expected_size in required.items():
    path = PROJECT_ROOT / 'weights' / name
    if path.exists():
        size_mb = path.stat().st_size / 1024**2
        print(f"  ✓ {name} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {name} — 파일 없음 (필수)")

for name, expected_size in optional.items():
    path = PROJECT_ROOT / 'weights' / name
    status = "✓" if path.exists() else "○"
    print(f"  {status} {name} (선택)")

# 2. gfpgan/weights/ — 자동 다운로드
print("\n" + "=" * 60)
print("2. gfpgan/weights/ (facexlib 자동 관리)")
print("=" * 60)
gfpgan_files = ['detection_Resnet50_Final.pth', 'parsing_parsenet.pth']
for name in gfpgan_files:
    path = PROJECT_ROOT / 'gfpgan' / 'weights' / name
    if path.exists():
        size_mb = path.stat().st_size / 1024**2
        print(f"  ✓ {name} ({size_mb:.1f} MB)")
    else:
        print(f"  ○ {name} — 첫 실행 시 자동 다운로드됨")

# 3. 시스템 캐시 — 자동 관리
print("\n" + "=" * 60)
print("3. 시스템 캐시 (라이브러리 자동 관리)")
print("=" * 60)
cache_paths = {
    "DDColor (modelscope)": HOME / ".cache/modelscope/hub/damo/cv_ddcolor_image-colorization",
    "BLIP-large (HF)":      HOME / ".cache/huggingface/hub/models--Salesforce--blip-image-captioning-large",
    "CLIP-large (HF)":      HOME / ".cache/huggingface/hub/models--openai--clip-vit-large-patch14",
}
for name, path in cache_paths.items():
    status = "✓" if path.exists() else "○ (첫 실행 시 자동)"
    print(f"  {status} {name}")
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024**2
        print(f"     → {size:.1f} MB")

print("\n" + "=" * 60)
print("범례: ✓ = 다운로드 완료 / ○ = 미다운로드 (정상)")
print("=" * 60)
```

저장 위치: `check_models.py` (프로젝트 루트). 실행:

```bash
python check_models.py
```

### 7.2 자주 발생하는 문제

#### 문제 1: "가중치 파일을 찾을 수 없습니다"

```
FileNotFoundError: 가중치 파일을 찾을 수 없습니다:
  .../weights/RealESRGAN_x4plus.pth
```

**원인**: 사용자 직접 다운로드 누락.

**해결**: [3장](#3-weights--사용자-직접-다운로드) 참고하여 다운로드.

#### 문제 2: facexlib 다운로드 실패

```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**원인**: 회사망 SSL 인증서 차단, 또는 인터넷 접속 제한.

**해결**:
1. VPN 또는 다른 네트워크 사용
2. 또는 [4.4절](#44-수동-다운로드-인터넷-차단-환경용) 참고하여 수동 다운로드

#### 문제 3: HuggingFace 다운로드 rate limit

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

**원인**: 익명 다운로드의 rate limit.

**해결**: HF Token 발급 후 환경변수 설정:

```powershell
# Windows
$env:HF_TOKEN="hf_xxxxxxxxxxxx"
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_xxxxxxxxxxxx', 'User')
```

토큰 발급: https://huggingface.co/settings/tokens

#### 문제 4: DDColor 다운로드 끊김

modelscope의 한국 서버 접속 문제일 수 있습니다.

**해결**: 환경변수로 미러 사용:

```bash
export MODELSCOPE_DOMAIN="https://modelscope.cn"
```

#### 문제 5: gfpgan/ 폴더가 다른 위치에 생성됨

**원인**: 작업 디렉토리(CWD) 차이.

**해결**: 반드시 프로젝트 루트에서 실행:

```bash
cd /path/to/03_diaspora_image_processing
python -m src.pipeline ...
```

### 7.3 디스크 공간 부족 시

본 시스템은 약 5GB의 디스크 공간을 사용합니다. 부족하면:

| 절약 가능 항목 | 절약량 | 효과 |
|---|---|---|
| `RealESRGAN_x2plus.pth` 미다운로드 | 64 MB | 영향 없음 (default 미사용) |
| DeOldify 미설치 | 800 MB+ | DeOldify 백엔드만 사용 불가, DDColor는 정상 |
| BLIP/CLIP 캐시 다른 드라이브 | 3.5 GB | `HF_HOME` 환경변수로 이전 가능 |

---

## 8. 모델 라이선스 정보

각 모델은 원 프로젝트의 라이선스를 따릅니다. **상업적 사용 시 반드시 각 라이선스를 확인하세요.**

| 모델 | 라이선스 | 출처 |
|---|---|---|
| Real-ESRGAN | BSD 3-Clause | https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE |
| GFPGAN | Apache License 2.0 | https://github.com/TencentARC/GFPGAN/blob/master/LICENSE |
| DDColor | Apache License 2.0 | https://github.com/piddnad/DDColor/blob/master/LICENSE |
| facexlib (RetinaFace, Parsing) | MIT | https://github.com/xinntao/facexlib/blob/master/LICENSE |
| BLIP | BSD 3-Clause | https://github.com/salesforce/BLIP/blob/main/LICENSE.txt |
| CLIP | MIT | https://github.com/openai/CLIP/blob/main/LICENSE |
| face_recognition | MIT | https://github.com/ageitgey/face_recognition/blob/master/LICENSE |
| DeOldify | MIT | https://github.com/jantic/DeOldify/blob/master/LICENSE |

본 시스템 코드 자체는 [MIT License](LICENSE)로 배포되며, 모델 가중치들은 위의 각 라이선스를 따릅니다.

---

## 9. 인용

본 시스템 사용 시 위의 모든 모델들도 함께 인용해 주세요. 핵심 BibTeX:

```bibtex
@inproceedings{wang2021realesrgan,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author={Wang, Xintao and others},
  booktitle={ICCVW},
  year={2021}
}

@inproceedings{wang2021gfpgan,
  title={Towards Real-World Blind Face Restoration with Generative Facial Prior},
  author={Wang, Xintao and others},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and others},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and others},
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

**문의**: 모델 다운로드 관련 문제는 [GitHub Issues](../../issues)로 보고해 주세요.

**최종 업데이트**: 2026-05-03
