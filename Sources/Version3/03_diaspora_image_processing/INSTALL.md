# 설치 가이드

> 이 문서는 새 컴퓨터에서 본 시스템을 처음부터 동작시키기까지의 모든 단계를 안내합니다.  
> 총 작업 시간: 약 2~4시간 (모델 다운로드 시간 포함)

## 목차

- [0. 시스템 요구사항](#0-시스템-요구사항)
- [1. Python 환경 설치](#1-python-환경-설치)
- [2. PyTorch + CUDA 설치](#2-pytorch--cuda-설치)
- [3. AI 라이브러리 설치](#3-ai-라이브러리-설치)
- [4. 모델 가중치 다운로드](#4-모델-가중치-다운로드)
- [5. 환경변수 설정](#5-환경변수-설정)
- [6. 동작 검증](#6-동작-검증)
- [7. 자주 발생하는 문제](#7-자주-발생하는-문제)

---

## 0. 시스템 요구사항

### 하드웨어

| 항목 | 최소 사양 | 권장 사양 |
|---|---|---|
| OS | Windows 10 / Linux (Ubuntu 22.04+) | Windows 11 |
| GPU | NVIDIA RTX 8GB VRAM (예: 4060 Laptop) | NVIDIA RTX 12GB+ VRAM |
| RAM | 16GB | 32GB |
| 저장공간 | 30GB | 50GB+ |
| CUDA Capability | ≥ 7.5 (Turing) | ≥ 8.6 (Ampere) |

> ⚠️ **AMD/Intel GPU는 본 가이드 범위 밖입니다.** CPU 모드로도 동작하지만 SR이 사진 1장당 10분 이상 걸려 실용적이지 않습니다.

### GPU 드라이버 확인

PowerShell (Windows) 또는 터미널 (Linux)에서:

```bash
nvidia-smi
```

다음을 확인:
- `Driver Version`: 535 이상 (CUDA 12.1 호환)
- `CUDA Version`: 12.1 이상
- `Memory-Usage`: GPU의 총 메모리

> 드라이버가 오래됐다면 [NVIDIA 공식 사이트](https://www.nvidia.com/drivers)에서 최신 Studio Driver 또는 Game Ready Driver를 설치하세요. **CUDA Toolkit 자체는 별도 설치할 필요가 없습니다** (PyTorch가 자체 CUDA 런타임 포함).

---

## 1. Python 환경 설치

### 1.1 Miniconda 설치

[Miniconda 다운로드 페이지](https://docs.conda.io/en/latest/miniconda.html)에서 **Miniconda3 64-bit** 설치.

설치 시:
- ✅ "Add Miniconda3 to my PATH environment variable" (경고가 떠도 OK)

설치 후 PowerShell 새 창에서 확인:

```bash
conda --version
# 예: conda 23.5.2
```

### 1.2 가상환경 생성

```bash
conda create -n diaspora python=3.11 -y
conda activate diaspora
```

> 💡 **Python 3.11을 정확히 지정하는 이유**: dlib-bin · face_recognition_models 등 핵심 패키지가 3.11 wheel을 제공합니다. 3.12나 3.13은 wheel이 없어 컴파일 에러가 발생할 수 있습니다.

---

## 2. PyTorch + CUDA 설치

### 2.1 PyTorch 2.x (CUDA 12.1 빌드)

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

설치는 약 3~5분 (PyTorch 약 2.5GB).

### 2.2 GPU 인식 검증

**반드시 다음 명령으로 GPU가 인식되는지 확인하세요:**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')"
```

기대 출력:
```
CUDA available: True
Device: NVIDIA GeForce RTX 4070 Laptop GPU
VRAM: 8.0 GB
```

> ⚠️ `CUDA available: False`가 나오면:
> - GPU 드라이버 미설치 또는 너무 오래됨 → 드라이버 업데이트
> - Python이 CPU 전용 PyTorch를 받았을 가능성 → `pip uninstall torch torchvision torchaudio` 후 위 명령 재실행

---

## 3. AI 라이브러리 설치

### 3.1 일반 라이브러리 (한 번에)

```bash
pip install -r requirements.txt
```

> 단, requirements.txt에 명시된 PyTorch 줄은 위의 2.1에서 이미 정확한 버전으로 설치했으므로 다시 설치되지 않습니다 (pip이 이미 만족된 의존성을 건너뜀).

### 3.2 ⚠️ face_recognition — 분리 설치 필수 (한국어 Windows의 함정)

한국어 Windows 환경에서는 단순 `pip install face_recognition`이 **dlib 컴파일 시 cp949 인코딩 충돌**로 실패합니다. 다음 순서로 설치:

#### 3.2.1 dlib-bin 먼저 (사전 컴파일 wheel)

```bash
pip install dlib-bin
```

Python 3.11 Windows x64용 사전 빌드 wheel(약 2.7MB)이 다운로드됩니다. CMake나 Visual Studio Build Tools가 필요 없습니다.

#### 3.2.2 face_recognition을 의존성 무시 모드로

```bash
pip install face_recognition --no-deps
```

**`--no-deps`가 핵심**입니다. face_recognition은 의존성으로 dlib≥19.7을 명시하는데, pip은 dlib-bin이 dlib을 제공하는 줄 모르고 정식 dlib을 빌드하려 시도합니다.

#### 3.2.3 나머지 의존성 명시 설치

```bash
pip install face_recognition_models click
```

#### 3.2.4 검증

```bash
python -c "import dlib; import face_recognition; print('dlib:', dlib.__version__); print('face_recognition: OK')"
```

기대 출력:
```
dlib: 20.0.1
face_recognition: OK
```

> 💡 설치 후 `face-recognition 1.3.0 requires dlib>=19.7, which is not installed` 경고가 떠도 무시하세요. pip의 의존성 메타데이터 한계일 뿐, 실제 동작은 정상입니다.

---

## 4. 모델 가중치 다운로드

### 4.1 weights/ 폴더 생성

프로젝트 루트에서:

```bash
mkdir weights
cd weights
```

### 4.2 사용자가 직접 다운로드해야 하는 가중치

| 파일 | 크기 | URL |
|---|---|---|
| RealESRGAN_x4plus.pth | 64MB | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth |
| GFPGANv1.4.pth | 348MB | https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth |

PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "RealESRGAN_x4plus.pth"
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "GFPGANv1.4.pth"
```

Linux/macOS:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

### 4.3 (선택) 배경 업샘플러 — 보통 불필요

face_enhancement의 `bg_upsampler` 옵션을 활성화할 때만 필요. 현재 default 설정에서는 사용하지 않으므로 다운로드 생략 가능.

```
RealESRGAN_x2plus.pth (~64MB)
URL: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

### 4.4 자동 다운로드되는 가중치 (별도 작업 불필요)

다음은 첫 실행 시 자동 다운로드됩니다:

| 모델 | 크기 | 저장 위치 |
|---|---|---|
| DDColor | ~870MB | `~/.cache/modelscope/hub/damo/cv_ddcolor_image-colorization/` |
| RetinaFace ResNet50 | ~110MB | `~/.cache/torch/hub/checkpoints/` |
| BLIP-large | ~1.8GB | `~/.cache/huggingface/hub/` |
| CLIP ViT-Large/14 | ~1.7GB | `~/.cache/huggingface/hub/` |

> 💡 자동 다운로드는 첫 실행 시 인터넷 연결 필수입니다. 회사망에서 외부 접속이 차단되면 별도 다운로드 후 캐시 경로에 수동 배치 필요.

---

## 5. 환경변수 설정

### 5.1 PYTORCH_CUDA_ALLOC_CONF (8GB VRAM 권장)

8GB VRAM 환경에서 메모리 단편화로 인한 OOM을 방지합니다.

#### 매 세션마다 (PowerShell)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

#### 영구 설정 (한 번만 실행, 권장)

```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True', 'User')
```

```bash
# Linux/macOS — ~/.bashrc 또는 ~/.zshrc에 추가
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

> ⚠️ Windows에서 "expandable_segments not supported on this platform" 경고가 뜨지만 동작은 됩니다.

> 💡 16GB+ VRAM GPU에서는 이 환경변수가 굳이 필요하지 않지만, 켜둬도 무해합니다.

### 5.2 (선택) HuggingFace Token

BLIP/CLIP 다운로드 시 rate limit을 피하려면:

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxx"
```

토큰 발급: https://huggingface.co/settings/tokens

---

## 6. 동작 검증

### 6.1 import 테스트

```bash
python -c "from src.pipeline import ImageRestorationPipeline, ProcessingOptions; print('All OK')"
```

`All OK`만 출력되면 성공.

### 6.2 첫 실행 (테스트 이미지 1장)

`data/input/test.jpg`에 작은 흑백 사진 (예: 800×600 이내)을 둔 후:

```bash
python -m src.pipeline \
    --input data/input/test.jpg \
    --output data/output/test_result.jpg \
    --low-vram --vram-verbose --save-intermediate
```

첫 실행 시 모델 다운로드 로그가 길게 흐릅니다 (수 분~10분). 이후 실행은 빠릅니다.

#### 기대 결과

- `성공: True` 메시지
- `적용된 단계: ['super_resolution', 'face_enhancement', 'colorization', 'analysis']`
- `data/output/test_result.jpg` 파일 생성
- `data/output/test_result_stages/` 폴더에 단계별 중간 결과

### 6.3 ablation 옵션 검증

```bash
python -m src.pipeline \
    --input data/input/test.jpg \
    --output data/output/test_optionE.jpg \
    --pipeline-order "color,sr,face" \
    --low-vram --save-intermediate
```

초기 로그에 다음이 출력되면 OK:

```
파이프라인 순서: colorization → super_resolution → face_enhancement → analysis
```

`data/output/test_optionE_stages/_pipeline_order.txt`도 확인:

```
colorization → super_resolution → face_enhancement
```

---

## 7. 자주 발생하는 문제

### 7.1 dlib-bin 설치 실패: "No matching distribution found"

**원인**: Python 버전 불일치.

```bash
python --version
# 3.11.x 여야 함
```

3.12 또는 3.13이라면 conda 환경을 다시 만드세요. 그래도 실패 시 [GitHub의 미리 빌드된 wheel](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)에서 직접 다운로드.

### 7.2 import 시 "functional_tensor" 에러

```
ImportError: cannot import name 'rgb_to_grayscale' from
    'torchvision.transforms.functional_tensor'
```

**원인**: torchvision 0.17+에서 `functional_tensor` 모듈 제거.

**해결**: `basicsr/data/degradations.py` 파일을 찾아 import 경로 수정:

```python
# 변경 전
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# 변경 후
from torchvision.transforms.functional import rgb_to_grayscale
```

위치 찾기:

```bash
python -c "import basicsr; print(basicsr.__file__)"
```

### 7.3 첫 실행 시 모델 다운로드 끊김

**원인**: huggingface.co 차단 또는 회사망 정책.

**해결**:
- VPN 또는 다른 네트워크에서 재시도
- HF_TOKEN 발급 후 환경변수 설정 (5.2절 참조)

### 7.4 CUDA out of memory

처리 중 `torch.AcceleratorError: CUDA error: out of memory`.

**해결 (시도 순서)**:

1. `--low-vram` 옵션이 켜져 있는지 확인
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경변수 확인
3. 입력 이미지가 너무 크면 (예: 6000×4000) 미리 1500px 정도로 줄여서 입력
4. 다른 GPU 사용 프로그램(브라우저, 게임, OBS 등) 종료
5. `--no-sr` 옵션으로 SR 단계만 끄고 처리

### 7.5 한글 경로 입력 파일을 못 찾음

```
FileNotFoundError: [Errno 2] No such file or directory: 'data\\input\\제사사진.jpg'
```

본 시스템의 `_imread_unicode` 함수가 한글 경로를 처리하지만, 파일이 실제로 존재하는지 확인:

```powershell
Test-Path "data\input\제사사진.jpg"
```

`True`가 나오는데도 실패하면 PowerShell의 인코딩 문제일 수 있으므로 영문 파일명으로 변경 권장.

### 7.6 face_recognition 미인식

실행 로그에 `Warning: face_recognition not installed` 발생.

**해결**:

```bash
conda activate diaspora
pip list | findstr -i "face_recognition dlib"
```

다음과 같이 두 줄이 모두 나와야 함:

```
dlib-bin             20.0.1
face-recognition     1.3.0
```

빠진 패키지는 [3.2절](#32-️-face_recognition--분리-설치-필수-한국어-windows의-함정) 참고하여 재설치.

---

## 8. 빠른 시작 (Quick Start) — 명령어 모음

위 모든 단계를 순서대로 PowerShell에서 한 번에:

```powershell
# 1. conda 환경
conda create -n diaspora python=3.11 -y
conda activate diaspora

# 2. PyTorch + CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 3. AI 라이브러리
pip install -r requirements.txt
pip install dlib-bin
pip install face_recognition --no-deps
pip install face_recognition_models click

# 4. 환경변수 영구 설정
[System.Environment]::SetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True', 'User')

# 5. 가중치 다운로드
mkdir weights -ErrorAction SilentlyContinue
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "weights\RealESRGAN_x4plus.pth"
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "weights\GFPGANv1.4.pth"

# 6. 검증
python -c "import torch; import dlib; import face_recognition; from src.pipeline import ImageRestorationPipeline; print('All OK')"

# 7. 첫 처리 테스트
python -m src.pipeline --input data/input/test.jpg --output data/output/test.jpg --low-vram --vram-verbose --save-intermediate
```

이 단계까지 모두 성공하면 모든 ablation 실험을 재현할 수 있습니다.
