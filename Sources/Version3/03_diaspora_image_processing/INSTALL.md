# 설치 가이드 (INSTALL.md)

> 본 가이드는 **아무것도 설치되지 않은 노트북**에 본 시스템을 처음부터 구축하는 방법을 단계별로 설명합니다. 학회 발표·시연용 노트북 셋업을 가정합니다.

---

## 목차

1. [사전 점검: 하드웨어 확인](#1-사전-점검-하드웨어-확인)
2. [NVIDIA 드라이버 및 CUDA 설치](#2-nvidia-드라이버-및-cuda-설치)
3. [Python 환경 (Miniconda)](#3-python-환경-miniconda)
4. [Git 설치](#4-git-설치)
5. [본 저장소 클론](#5-본-저장소-클론)
6. [conda 가상환경 생성](#6-conda-가상환경-생성)
7. [PyTorch 설치 (CUDA 12.1)](#7-pytorch-설치-cuda-121)
8. [Python 패키지 설치 (순서대로)](#8-python-패키지-설치-순서대로)
9. [모델 가중치 다운로드](#9-모델-가중치-다운로드)
10. [설치 검증](#10-설치-검증)
11. [발표용 시연 준비](#11-발표용-시연-준비)
12. [트러블슈팅](#12-트러블슈팅)

---

## 1. 사전 점검: 하드웨어 확인

### 필수 사양

- **GPU**: NVIDIA GPU (CUDA 11.8 이상 호환). 최소 8GB VRAM 권장 (RTX 3060/3070 이상)
- **RAM**: 최소 16GB (32GB 권장)
- **저장 공간**: 최소 30GB 여유 (모델 가중치 약 2GB, conda 환경 약 10GB, 캐시 약 5GB)
- **OS**: Windows 10/11, 64bit

### GPU 확인 방법 (Windows PowerShell)

```powershell
# GPU 모델 확인
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM

# 또는
dxdiag
```

GPU가 NVIDIA가 아니거나 매우 오래된 경우, CPU 모드로도 동작은 가능하나 처리 시간이 매우 길어집니다 (이미지당 5~10분).

---

## 2. NVIDIA 드라이버 및 CUDA 설치

### 2.1 NVIDIA 드라이버

이미 최신 GeForce Experience 또는 NVIDIA 드라이버가 설치되어 있다면 스킵 가능합니다.

1. https://www.nvidia.com/Download/index.aspx 접속
2. GPU 모델, OS 선택 후 다운로드
3. 설치 (재부팅 필요)

설치 확인 (PowerShell):
```powershell
nvidia-smi
```

다음과 같이 GPU 정보가 표시되면 정상:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.xx       Driver Version: 545.xx       CUDA Version: 12.3   |
+-----------------------------------------------------------------------------+
| GPU  Name           ...                                                    |
| 0    NVIDIA GeForce RTX 4080  ...                                          |
+-----------------------------------------------------------------------------+
```

> **참고**: `nvidia-smi`에 표시되는 CUDA Version은 **드라이버가 지원하는 최대 CUDA 버전**입니다. 실제 PyTorch가 사용할 CUDA 12.1보다 높아도 무방합니다.

### 2.2 CUDA Toolkit

**CUDA Toolkit을 별도로 설치할 필요는 없습니다**. PyTorch가 자체 CUDA 런타임을 포함하므로, NVIDIA 드라이버만 있으면 충분합니다.

---

## 3. Python 환경 (Miniconda)

Anaconda보다 가벼운 Miniconda를 권장합니다.

### 3.1 Miniconda 다운로드

1. https://docs.conda.io/projects/miniconda/en/latest/ 접속
2. **Windows 64-bit** 인스톨러 다운로드 (Python 3.11 권장 버전)
3. 실행

### 3.2 설치 옵션

설치 시 다음 옵션 권장:
- ☑ **Add Miniconda to my PATH environment variable** (체크 권장 — 어디서나 conda 명령 사용 가능)
- ☑ Register Miniconda as the system default Python

### 3.3 설치 확인

새 PowerShell 창을 열고:

```powershell
conda --version
python --version
```

각각 버전 정보가 출력되면 정상입니다.

> **PATH 추가를 안 한 경우**: 시작 메뉴에서 "Anaconda Prompt (miniconda3)" 또는 "Anaconda PowerShell Prompt"를 사용하세요.

---

## 4. Git 설치

본 저장소를 클론하기 위해 필요합니다.

1. https://git-scm.com/download/win 접속
2. 64-bit Windows installer 다운로드
3. 설치 (모든 기본 옵션 그대로)

설치 확인:
```powershell
git --version
```

---

## 5. 본 저장소 클론

원하는 디렉토리로 이동 후 클론:

```powershell
# 예: 다운로드 폴더로 이동
cd C:\Users\<YourName>\Downloads

# 저장소 클론 (실제 URL로 교체)
git clone https://github.com/<your_username>/diaspora_thesis_code.git

# 이미지 처리 모듈 디렉토리로 이동
cd diaspora_thesis_code\03_diaspora_image_processing
```

---

## 6. conda 가상환경 생성

```powershell
# 가상환경 생성 (Python 3.11)
conda create -n diaspora python=3.11 -y

# 환경 활성화
conda activate diaspora
```

활성화되면 프롬프트 앞에 `(diaspora)` 가 표시됩니다.

```
(diaspora) PS C:\...\03_diaspora_image_processing>
```

> **이후 모든 명령은 이 환경이 활성화된 상태에서 실행해야 합니다.**

---

## 7. PyTorch 설치 (CUDA 12.1)

본 시스템은 PyTorch 2.1.2 + CUDA 12.1로 검증되었습니다.

```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

설치 시간: 약 5~10분 (PyTorch 약 2.5GB 다운로드).

### 검증

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

다음과 같이 출력되면 정상:
```
PyTorch: 2.1.2+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4080
```

`CUDA available: False`로 나오면 NVIDIA 드라이버를 확인하고 PyTorch를 재설치해야 합니다.

---

## 8. Python 패키지 설치 (순서대로)

**순서가 중요합니다.** 의존성 충돌을 피하기 위해 다음 순서를 따르세요.

### 8.1 1단계: 핵심 이미지 처리 라이브러리

```powershell
pip install opencv-python==4.13.0.86
pip install Pillow
pip install numpy
pip install pyyaml
```

### 8.2 2단계: 초해상도 (Real-ESRGAN)

```powershell
pip install basicsr
pip install realesrgan
```

> **만약 basicsr 설치 시 에러가 나면**: 8.1 단계의 numpy/Pillow가 먼저 설치되어 있는지 확인하세요.

### 8.3 3단계: 얼굴 복원 (GFPGAN)

```powershell
pip install gfpgan
pip install facexlib
```

### 8.4 4단계: 얼굴 검출 fallback (face_recognition / dlib)

> **주의**: face_recognition은 dlib을 의존하며, dlib은 C++ 컴파일이 필요합니다. Windows에서는 사전 빌드된 wheel을 사용하는 것이 안전합니다.

```powershell
pip install dlib
pip install face_recognition
```

만약 dlib 설치 실패 시 사전 빌드 wheel 사용:
```powershell
# Python 3.11용 사전 빌드 wheel
pip install https://github.com/sachadee/Dlib/raw/main/dlib-19.22.99-cp311-cp311-win_amd64.whl
pip install face_recognition
```

### 8.5 5단계: 컬러화 (DDColor + modelscope)

DDColor는 modelscope 경유로 사용합니다. modelscope의 의존성이 자동 설치되지 않는 경우가 있어, 명시적으로 함께 설치합니다.

```powershell
pip install modelscope
pip install datasets
pip install simplejson
pip install addict
pip install sortedcontainers
pip install oss2
pip install timm
```

### 8.6 6단계: 이미지 분석 (BLIP, CLIP)

```powershell
pip install transformers
pip install accelerate
```

### 8.7 7단계 (선택): DeOldify (보조 백엔드)

DeOldify는 의존성 충돌이 자주 발생하므로 **선택사항**입니다. 설치하지 않아도 DDColor가 주 모델로 동작합니다.

DeOldify를 사용하려면 별도 conda 환경 권장:
```powershell
# 별도 환경
conda create -n diaspora_deoldify python=3.9 -y
conda activate diaspora_deoldify
pip install deoldify
# ...
```

본 발표용 시연에는 **DeOldify 설치 불필요**.

---

## 9. 모델 가중치 다운로드

프로젝트 루트(`03_diaspora_image_processing/`)에 `weights/` 디렉토리를 만들고 가중치 파일을 다운로드합니다.

```powershell
# weights 디렉토리 생성
mkdir weights
cd weights
```

### 9.1 Real-ESRGAN 가중치

```powershell
# 4배 SR (메인)
curl.exe -L -o RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 2배 SR (GFPGAN의 배경 업샘플러용)
curl.exe -L -o RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

> **`curl` 명령이 PowerShell에서 안 되는 경우**: PowerShell의 `curl`은 `Invoke-WebRequest`의 별칭이라 `-L` 옵션을 인식하지 못합니다. 반드시 `curl.exe`로 호출하세요. 또는 브라우저로 위 URL을 직접 열어 다운로드 후 weights/ 폴더에 넣어도 됩니다.

### 9.2 GFPGAN 가중치

```powershell
curl.exe -L -o GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

### 9.3 DDColor 가중치

DDColor는 **별도 다운로드가 필요 없습니다**. 최초 실행 시 modelscope이 자동으로 약 870MB의 가중치를 `~/.cache/modelscope/hub/`에 다운로드합니다.

다만 발표 직전에 처음 실행하면 다운로드 시간이 걸리므로, **사전에 한 번 실행**하여 캐시를 만들어두는 것을 권장합니다 (10단계 검증 시 자동 수행됨).

### 9.4 디렉토리 구조 확인

```powershell
cd ..
ls weights
```

다음과 같이 표시되면 정상:
```
RealESRGAN_x4plus.pth     (~64MB)
RealESRGAN_x2plus.pth     (~64MB)
GFPGANv1.4.pth            (~333MB)
```

---

## 10. 설치 검증

### 10.1 import 테스트

```powershell
python -c "from src.pipeline import ImageRestorationPipeline; print('OK')"
```

`OK`가 출력되면 모든 모듈이 정상 로드된 것입니다. (deprecation 경고는 무시해도 됨)

### 10.2 단일 이미지 처리 테스트

`data/input/` 폴더에 테스트 이미지를 1장 두고:

```powershell
python main.py --input data/input/<your_test_image>.jpg --output data/output/test_restored.jpg --verbose
```

처음 실행 시 DDColor 가중치(약 870MB)와 BLIP/CLIP 모델이 자동 다운로드됩니다 (10~30분 소요). 이후 실행은 캐시 사용.

다음과 같이 출력되면 성공:
```
✓ 처리 성공
   원본 크기: ...
   최종 크기: ...
   적용된 단계: super_resolution, face_enhancement, colorization, analysis
   처리 시간: ~30~60초
```

### 10.3 일괄 처리 테스트

```powershell
python batch_process.py --limit 1
```

`data/output/batch_<timestamp>/` 디렉토리에 결과가 생성되면 정상.

---

## 11. 발표용 시연 준비

학회·논문발표 노트북에 본 시스템을 시연용으로 준비할 때 추가 권장 사항입니다.

### 11.1 사전 캐시 워밍업

발표 직전에 모델이 처음 다운로드되는 상황을 피하기 위해, **사전에 1장이라도 처리**해두면 모든 모델이 캐시에 저장됩니다.

```powershell
# 시연 전 한 번 실행 (캐시 생성)
python batch_process.py --limit 1
```

이후 발표장에서는 캐시된 모델을 바로 로드하여 빠르게 동작합니다.

### 11.2 시연용 입력 이미지 준비

`data/input/` 폴더에 다음과 같이 시연용 이미지를 미리 정리:

```
data/input/
├── 01_demo_couple.jpg          # 단순 인물 (효과 명확)
├── 02_demo_group.jpg           # 단체사진 (얼굴 검출 데모)
└── 03_demo_protest.jpg         # 역사적 장면 (스토리텔링)
```

각 이미지에 대해 사전 1회 처리 후, 결과 이미지도 발표 자료로 백업해두면 발표 중 GPU 문제가 생겨도 백업으로 시연 가능.

### 11.3 발표 중 빠른 시연 명령어

```powershell
# 활성화 (매번 필요)
conda activate diaspora

# 단일 시연
python main.py -i data/input/01_demo_couple.jpg -o data/output/demo01.jpg

# 비교 이미지 함께
python batch_process.py --input-dir data/input
```

### 11.4 인터넷 연결 권장

- DDColor 가중치 자동 다운로드 (한 번만)
- BLIP/CLIP 모델 자동 다운로드 (한 번만)

발표장에서 인터넷이 불안정할 경우를 대비해 **반드시 사전 설치 + 사전 캐시 워밍업**을 마쳐두세요.

### 11.5 전원 설정

GPU 작업은 전력 소모가 큽니다. 발표 중에는 노트북을 **전원에 연결**하고, Windows 전원 옵션을 "고성능"으로 설정하세요.

---

## 12. 트러블슈팅

### 12.1 `CUDA available: False`

**원인**: NVIDIA 드라이버 미설치 또는 PyTorch가 CPU 버전으로 설치됨.

**해결**:
1. `nvidia-smi` 명령으로 드라이버 확인
2. PyTorch 재설치: `pip uninstall torch torchvision torchaudio -y`
3. 7단계의 명령으로 다시 설치

### 12.2 `ModuleNotFoundError: No module named 'xxx'`

**원인**: 의존성 누락. 특히 modelscope 관련 패키지가 자동 설치되지 않는 경우 발생.

**해결**: 누락된 모듈을 `pip install <module>`로 설치. 흔한 케이스:
```powershell
pip install datasets simplejson addict sortedcontainers oss2 timm
```

### 12.3 `AttributeError: 'NoneType' object has no attribute 'startswith'`

**원인**: 모델 가중치 파일이 `weights/` 폴더에 없음.

**해결**: 9단계에서 누락한 가중치 파일을 다운로드.

### 12.4 GFPGAN 또는 RetinaFace 동작 시 `Can't call numpy() on Tensor that requires grad`

**원인**: facexlib와 PyTorch 2.x의 호환성 문제.

**해결**: 본 시스템의 face_enhancement.py는 이를 자동 감지하여 HOG fallback으로 전환합니다. **무시해도 결과는 정상** (콘솔에 "HOG fallback이 N개 검출" 메시지 확인).

### 12.5 GPU 메모리 부족 (CUDA out of memory)

**원인**: 입력 이미지가 너무 큼 또는 다른 GPU 프로그램이 동작 중.

**해결**:
1. 다른 GPU 프로그램(브라우저, 게임 등) 종료
2. `super_resolution.py`의 `tile_size`를 1024 → 512로 낮춤
3. 매우 큰 이미지(>5MP)는 시스템이 자동으로 SR 스킵 (24MP 안전 임계값)

### 12.6 한글 파일명에서 `cv2.imread`가 None 반환

**원인**: OpenCV가 Windows 한글 경로를 처리하지 못함.

**해결**: 본 시스템의 모든 모듈은 한글 경로 안전 I/O를 구현하고 있습니다. 만약 새 모듈을 추가하실 경우, 다음 패턴 사용:
```python
data = np.fromfile(str(path), dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
```

### 12.7 modelscope 다운로드가 매우 느림

**원인**: modelscope 서버는 중국에 위치하여 한국에서 다운로드 속도가 일정하지 않음.

**해결**: 인내심을 가지고 기다리거나, VPN 사용. 한 번 다운로드 후에는 캐시 사용.

---

## 13. 빠른 환경 재구축 체크리스트

새 노트북에 본 시스템을 옮길 때 사용할 체크리스트입니다.

- [ ] NVIDIA 드라이버 설치 (`nvidia-smi` 동작 확인)
- [ ] Miniconda 설치 (`conda --version` 동작 확인)
- [ ] Git 설치 (`git --version` 동작 확인)
- [ ] 저장소 클론 완료
- [ ] `conda create -n diaspora python=3.11 -y && conda activate diaspora`
- [ ] PyTorch 2.1.2 (cu121) 설치 → `torch.cuda.is_available()` True 확인
- [ ] 8단계의 패키지 그룹 1~6 모두 설치
- [ ] `weights/` 폴더에 가중치 3개 (RealESRGAN_x4plus, RealESRGAN_x2plus, GFPGANv1.4) 위치
- [ ] `python -c "from src.pipeline import ImageRestorationPipeline; print('OK')"` 동작
- [ ] 테스트 이미지 1장으로 `main.py` 실행 성공
- [ ] DDColor 가중치 자동 다운로드 완료 (캐시 워밍업)
- [ ] 시연용 이미지를 `data/input/` 에 배치
- [ ] `batch_process.py --limit 1` 으로 일괄 처리 동작 확인

위 항목이 모두 체크되면 발표 시연 준비 완료입니다.

---

## 부록: 환경 export 및 백업

발표 후 환경을 백업하거나 다른 노트북에 동일하게 복제할 때 사용:

### 환경 export

```powershell
# 가상환경 활성화 상태에서
conda env export > environment.yml
pip freeze > requirements_full.txt
```

### 환경 복제

```powershell
# 백업한 yaml로 동일 환경 생성
conda env create -f environment.yml
conda activate diaspora
```

이렇게 하면 다른 노트북에서도 동일한 의존성을 보장할 수 있어, **재현성** 측면에서 학위논문 부록에 포함할 가치가 있습니다.

---

문제 발생 시 트러블슈팅 섹션을 먼저 확인하시고, 해결되지 않으면 GitHub Issues에 보고 부탁드립니다.
