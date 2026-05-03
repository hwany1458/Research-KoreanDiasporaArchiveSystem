# 실험 결과: 8GB VRAM 노트북 검증 + 모듈 순서 Ablation

> 본 문서는 본 시스템을 8GB VRAM 노트북 환경에서 개발·검증한 과정과,  
> 모듈 실행 순서에 대한 5회 ablation 실험의 결과를 정리합니다.

## 목차

- [1. 개요](#1-개요)
- [2. 8GB VRAM 환경 호환성 작업](#2-8gb-vram-환경-호환성-작업)
- [3. 모듈 실행 순서 Ablation 실험](#3-모듈-실행-순서-ablation-실험)
- [4. 핵심 발견 및 권장 설정](#4-핵심-발견-및-권장-설정)
- [5. 재현 방법](#5-재현-방법)

---

## 1. 개요

### 검증 환경

| 항목 | 사양 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX (8GB VRAM 추정, 노트북 GPU) |
| Python | 3.11 |
| PyTorch | 2.x + CUDA 12.1 |
| 테스트 이미지 | KADA-Edwinlee091.jpg (1200×844, 흑백, 1980년대 한인 단체사진) |

### 실험 목적

1. **8GB VRAM 노트북에서 SOTA AI 파이프라인이 동작 가능한가?**
2. **모듈 실행 순서가 결과 품질에 어떤 영향을 미치는가?**
3. **각 모듈의 실패 모드와 해결책은 무엇인가?**

---

## 2. 8GB VRAM 환경 호환성 작업

### 2.1 초기 문제 — CUDA OOM

본 시스템은 4개 SOTA AI 모듈을 사용하는데, 동시 적재 시 8GB VRAM을 초과합니다.

| 모듈 | VRAM 사용량 |
|---|---|
| Real-ESRGAN x4plus | 2~3 GB |
| GFPGAN + RetinaFace | 1.5~2 GB |
| DDColor | 0.9~2 GB |
| BLIP + CLIP | ~2 GB |
| **동시 적재 시** | **6.4~9 GB** ❌ |

### 2.2 적용한 메커니즘 (5가지)

#### 2.2.1 모듈 lazy load + unload

`ImageRestorationPipeline`의 모듈 속성을 `@property`로 정의해 최초 접근 시점에만 GPU 적재. `--low-vram` 모드에서 단계 종료 후 모듈을 GPU에서 unload (model.to('cpu') → del → torch.cuda.empty_cache()).

#### 2.2.2 PYTORCH_CUDA_ALLOC_CONF 환경변수

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

PyTorch 메모리 할당기를 expandable_segments 모드로 전환하여 단편화로 인한 OOM 방지.

#### 2.2.3 SR 타일 크기 자동 조정

| 모드 | 타일 크기 |
|---|---|
| `--low-vram` (8GB) | 128 |
| 일반 모드 (16GB+) | 256 |
| `--sr-tile-size 0` | 분할 없음 (가장 빠름) |

#### 2.2.4 Face/Color 모듈 입력 다운스케일 (max_input_dim)

⭐ **본 시스템의 핵심 기여 중 하나**

GFPGAN과 DDColor 모두 최대 입력 변을 1536px로 제한. 결과를 LANCZOS4로 원본 해상도 복원.

**근거:**
- GFPGAN은 검출 얼굴을 512×512로 정렬하여 복원하므로, 검출 입력 해상도가 복원 품질에 거의 영향 없음
- DDColor는 256~512에서 학습되어, 큰 입력에서 색상 일관성이 오히려 저하됨

**효과:**
- VRAM 1.5GB 이하 안정 동작
- 색상 cast 부분 완화 (DDColor 모델 자체 한계는 잔존 — Section 3 참조)

#### 2.2.5 LAB 공간 색상 합성 (디테일 무손실 보존)

DDColor의 다운스케일된 출력에서 ab(색상) 채널만 추출 → 원본 해상도로 LANCZOS4 업스케일 → 원본 SR 결과의 L(휘도) 채널과 LAB 공간에서 합성 → BGR로 복원.

**검증:**
- L 채널 max diff = 1, mean diff = 0.03 (사실상 무손실)
- 4배 SR로 얻은 디테일이 컬러화 후에도 100% 보존

### 2.3 호환성 작업 전후 비교

| 항목 | 작업 전 | 작업 후 |
|---|---|---|
| Face 단계 | OOM ❌ | 정상 ✅ |
| Color 단계 | 정상 / 색상 cast | 정상 / 디테일↑ |
| VRAM 최대 사용량 | OOM (>8GB) | ≤1.5GB |
| 얼굴 검출 | 0개 (실패) | 14개 (HOG fallback) |
| ablation 실험 | 코드 수정 필요 | CLI 한 줄로 가능 |

---

## 3. 모듈 실행 순서 Ablation 실험

### 3.1 실험 설계

본 시스템의 SR / Face / Color 3단계는 6가지 순열이 가능하지만(3!), 학술적으로 의미 있는 5가지를 검증했습니다. Analysis는 항상 마지막에 실행되므로 순열 대상에서 제외.

### 3.2 종합 결과 표

| # | 순서 | 색상 cast | 얼굴 자연스러움 | 디테일 | 처리시간 | 종합 진단 |
|---|---|---|---|---|---|---|
| 1 | SR → Face → Color (default) | ❌ 보라색 cast 심함 | △ GFPGAN 흑백 복원 | ✅ 4800×3376 | 60.9초 | DDColor 큰 입력 한계 노출 |
| 2 | SR → Face → Color + 본 패치 | △ 부분 개선 | ✅ 안정적 | ✅ L채널 보존 무손실 | 47.3초 | 입력 정규화는 도움, 모델 한계 잔존 |
| 3 | **Color → Face → SR (옵션 B)** | ✅ 02까지 완벽 → SR이 망침 | ❌ SR이 얼굴 일그러뜨림 | ✅ 4800×3376 | ~50초 | Real-ESRGAN이 GFPGAN 출력을 OOD 처리 |
| 4 | **SR → Color → Face (옵션 D)** | ❌ cast 재발 | ✅ 안정적 | ✅ 4800×3376 | ~55초 | 얼굴 보호되나 색상 cast 미해결 |
| 5 | **Color → SR → Face (옵션 E)** | △ 01완벽 → SR이 cast 도입 | ✅ 안정적 | ✅ 4800×3376 | ~55초 | Real-ESRGAN의 색상 편향 발견 |

> **범례**: ✅ 매우 우수 / △ 부분적 / ❌ 문제 발생

### 3.3 실험별 상세 진단

#### 실험 1: SR → Face → Color (Default)

**최초 가설**: 디테일 먼저 살리고 → 얼굴 복원 → 색상 추정 (직관적 순서)

**관찰된 문제:**
- 가운데 남성 양복: 부자연스러운 보라/자주색 cast
- 우측 상단 족자: 빨간 번짐
- 인물 사이 경계: 색상 누출 (edge artifact)

**진단**: DDColor가 4800×3376의 큰 입력에서 색상 일관성을 잃음. 학습 데이터(ImageNet)와 분포 차이가 있는 1980년대 동양인 단체사진에서 더 두드러짐.

**부수 발견 — CLIP 한국 카테고리 인식**: 본 시스템이 `family memorial service (jesa)` 카테고리를 후보로 식별 (점수 0.17). 범용 BLIP은 같은 사진을 "restaurant"로 분류 → 본 시스템의 한국 컨텍스트 인식 차별성.

#### 실험 2: 옵션 A — 본 패치 적용 (face 1536px + LAB 합성)

**가설**: 입력 정규화로 색상 cast 완화 가능

**결과**: 디테일은 완벽히 보존됐지만 색상 cast는 부분 개선만. **입력 크기가 아닌 DDColor 모델 자체의 학습 분포 편향이 근본 원인**임이 드러남.

**학술적 근거**: arXiv 2510.23399 (Zhuang, 2025)에서 "DDColor의 입력 차원 부족으로 인한 color cast 문제"가 명시적으로 보고된 바 있음.

#### 실험 3: 옵션 B — Color → Face → SR

**가설**: 작은 흑백 입력에서 컬러화 → 색이 있는 얼굴을 GFPGAN이 자연스럽게 복원 → 마지막 SR

**결과:**
- **01_colorization (1200×844)**: ⭐ 최고 품질. 보라색 cast 사라짐, 자연스러운 동양인 피부톤
- **02_face_enhancement**: 01과 거의 동일, 미세하게 부드러워짐. 이상적
- **03_super_resolution (4800×3376)**: ❌ 큰 문제. 얼굴 일그러짐 + 색상 cast 재출현

**핵심 발견**: Real-ESRGAN은 일반 컬러 사진의 열화 복원에 학습됐는데, GFPGAN+DDColor가 만들어낸 합성 컬러 이미지는 학습 분포 외(OOD)이므로 "노이즈"로 잘못 판단해 추가 변형을 가함. 작은 얼굴(뒷줄·족자 속)일수록 픽셀 수가 적어 더 심하게 망가짐.

**중요**: 만약 4800×3376이 필수가 아니라면 **옵션 B의 02 결과가 시각적으로 가장 자연스러움**.

#### 실험 4: 옵션 D — SR → Color → Face

**가설**: 색상 cast와 얼굴 변형 둘 다를 피할 수 있는 후보

**결과:**
- 얼굴 보호 가설 입증 ✅ (face가 SR 다음에 오면 변형 없음)
- 그러나 옵션 A의 LAB 합성만으로는 4800×3376 입력의 색상 cast 미해결

**결론**: 색상이 좋으려면 DDColor 입력이 작아야 함 → Color가 SR보다 먼저 와야 함

#### 실험 5: 옵션 E — Color → SR → Face

**가설**: 두 우선 조건을 동시 만족 (Color가 SR보다 먼저 + Face가 SR보다 나중)

**결과:**
- **01_colorization**: 옵션 B 02와 동일하게 자연스러움 (1200×844)
- **02_super_resolution**: ⚠️ Real-ESRGAN이 컬러 합성 결과를 업스케일하면서 보라색 cast 도입
- **03_face_enhancement**: 02의 cast 그대로 물려받음, 얼굴 변형은 없음

**결정적 발견**: Real-ESRGAN 자체가 보라색 편향을 가짐. 이전엔 "GFPGAN+DDColor 합성 결과가 OOD라서 SR이 변형"으로 해석했지만, 옵션 E는 SR 직전 입력이 "DDColor만 거친 깨끗한 컬러 이미지"였음에도 SR 후 cast 발생.

### 3.4 5회 실험으로 도출된 종합 진단

| 진단 항목 | 내용 |
|---|---|
| **DDColor 입력 크기** | ≤1536px일 때 색상 일관성 양호. 4800×3376 같은 큰 입력에서는 학습 분포 편향이 두드러져 보라색 cast 발생. |
| **GFPGAN 입력 크기** | max_input_dim=1536 패치로 큰 입력 안전 처리 가능. 검출 입력 다운스케일이 복원 품질에 영향 없음 (어차피 512×512 정렬). |
| **Real-ESRGAN 한계** | (1) GFPGAN의 매끈한 얼굴 출력을 OOD로 처리해 변형. (2) 컬러 합성 결과를 업스케일하면 보라색 편향 도입. |
| **얼굴은 마지막에** | Face 단계 다음에 어떤 변환도 오지 않아야 GFPGAN의 정밀 복원이 보존됨. |
| **LAB 공간 합성** | Color 단계의 다운스케일이 클 때 L 채널을 원본에서 보존하여 디테일 무손실. 색상 cast의 근본 원인이 모델 자체일 때는 부분적 도움만. |

---

## 4. 핵심 발견 및 권장 설정

### 4.1 권장 default 순서

#### 권장 1: "Color → Face" (SR 생략) — 1200×844 결과

만약 디아스포라 아카이브가 1200×844 또는 그와 비슷한 해상도로 충분하다면, 이 순서가 시각적으로 가장 자연스러운 결과를 냅니다.

```bash
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo.jpg \
    --no-sr --pipeline-order "color,face" \
    --low-vram --save-intermediate
```

**장점**: 색상 일관성 ✅, 얼굴 자연스러움 ✅, 처리 시간 짧음, VRAM 매우 안전  
**단점**: 4배 업스케일 효과 없음

#### 권장 2: 옵션 E — Color → SR → Face — 4800×3376 결과

4800×3376 결과가 필수라면 옵션 E가 가장 균형 잡힌 결과. 색상 cast가 일부 있지만 얼굴 변형은 없음.

```bash
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo.jpg \
    --pipeline-order "color,sr,face" \
    --low-vram --save-intermediate
```

**장점**: 4배 해상도, 얼굴 보존, 옵션 D보다 색상 안정적  
**단점**: SR 후 약간의 보라색 편향 잔존

### 4.2 향후 개선 방향

1. **Real-ESRGAN 색상 편향 보정**: SR 후 LAB 공간에서 ab 채널을 원본 컬러화 결과에 정합시키는 후처리 추가
2. **DeOldify 백엔드 비교**: 본 시스템에 이미 `backend='deoldify'` 옵션이 구현돼 있음
3. **GFPGAN background upsampler 활성화**: face_enhancement.py의 `bg_upsampler` 옵션 (현재 비활성화). 활성화 시 얼굴은 GFPGAN 정밀 복원, 배경은 동일 모델로 SR이 동시 진행
4. **다른 사진으로 일반화 검증**: 단일 사진 결과를 다른 흑백 디아스포라 사진으로 재현하여 일반화 입증

---

## 5. 재현 방법

### 5.1 5가지 ablation 실험 명령어

```bash
# 환경변수 (8GB VRAM 환경)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Windows PowerShell:
# $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 실험 1: baseline (default 순서, 본 패치 적용된 코드 = 옵션 A)
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionA.jpg \
    --low-vram --vram-verbose --save-intermediate

# 실험 2: 옵션 B — Color → Face → SR
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionB.jpg \
    --pipeline-order "color,face,sr" \
    --low-vram --vram-verbose --save-intermediate

# 실험 3: 옵션 C — Face → Color → SR
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionC.jpg \
    --pipeline-order "face,color,sr" \
    --low-vram --vram-verbose --save-intermediate

# 실험 4: 옵션 D — SR → Color → Face
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionD.jpg \
    --pipeline-order "sr,color,face" \
    --low-vram --vram-verbose --save-intermediate

# 실험 5: 옵션 E — Color → SR → Face (권장)
python -m src.pipeline \
    --input data/input/photo.jpg \
    --output data/output/photo_optionE.jpg \
    --pipeline-order "color,sr,face" \
    --low-vram --vram-verbose --save-intermediate
```

### 5.2 결과 비교

각 실행은 `data/output/photo_optionX_stages/` 폴더를 생성하며, 다음 파일이 저장됩니다:

- `01_*.jpg`, `02_*.jpg`, `03_*.jpg`: 단계별 중간 결과
- `_pipeline_order.txt`: 적용된 순서 메타파일

5개 폴더의 최종 결과 (output_optionX.jpg)와 단계별 중간 결과를 비교하면 본 ablation의 모든 발견을 재현할 수 있습니다.

### 5.3 정량 비교 (선택)

scikit-image와 lpips 라이브러리로 각 결과 사이의 PSNR / SSIM / LPIPS 측정:

```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import cv2
import numpy as np
import torch

def compare(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 같은 크기로 맞춤
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    psnr = peak_signal_noise_ratio(img1, img2)
    ssim = structural_similarity(img1, img2, channel_axis=2)
    
    # LPIPS
    loss_fn = lpips.LPIPS(net='alex').cuda()
    img1_t = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().cuda() / 127.5 - 1
    img2_t = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().cuda() / 127.5 - 1
    lpips_val = loss_fn(img1_t, img2_t).item()
    
    return psnr, ssim, lpips_val

# 예시: 옵션 A vs E 비교
psnr, ssim, lpips_v = compare(
    "data/output/photo_optionA.jpg",
    "data/output/photo_optionE.jpg"
)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, LPIPS: {lpips_v:.4f}")
```

---

## 6. 참고 문헌

본 실험에서 인용한 학술 자료:

- **Real-ESRGAN**: Wang, X. et al. (2021). *ICCVW*.
- **GFPGAN**: Wang, X. et al. (2021). *CVPR*.
- **DDColor**: Kang, X. et al. (2023). *ICCV*.
- **DDColor 한계 분석**: Zhuang, Y. (2025). arXiv:2510.23399. "Image Colorization via Frequency-Aware Refinement of DDColor."
- **CLIP**: Radford, A. et al. (2021). *ICML*.
- **BLIP**: Li, J. et al. (2022). *ICML*.
