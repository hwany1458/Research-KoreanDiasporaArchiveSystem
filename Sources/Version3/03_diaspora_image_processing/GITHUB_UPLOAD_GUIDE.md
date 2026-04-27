# GitHub 업로드 파일 선별 가이드

본 문서는 `03_diaspora_image_processing/` 디렉토리의 어떤 파일을 GitHub에 올리고, 어떤 파일은 제외해야 하는지 정리한 가이드입니다.

---

## ✅ 업로드해야 할 파일

### 1. 핵심 코드

```
03_diaspora_image_processing/
├── main.py                           ← 단일 처리 진입점
├── batch_process.py                  ← 일괄 처리 스크립트
├── src/
│   ├── __init__.py                   ← (있다면) 패키지 마커
│   ├── pipeline.py                   ← 메인 파이프라인
│   └── modules/
│       ├── __init__.py               ← (있다면) 패키지 마커
│       ├── super_resolution.py
│       ├── face_enhancement.py
│       ├── colorization.py
│       └── image_analysis.py
```

### 2. 문서

```
├── README.md                         ← 본 저장소 메인 문서
├── INSTALL.md                        ← 상세 설치 가이드
├── LICENSE                           ← (선택) 라이선스 파일
└── .gitignore                        ← Git 제외 목록
```

### 3. 설정 파일 (있다면)

```
├── config.yaml                       ← 처리 설정 예시
├── requirements.txt                  ← Python 의존성 목록
└── environment.yml                   ← (선택) conda 환경 정의
```

### 4. 테스트 데이터 (선택, 작은 샘플 1~2장만)

```
└── data/
    ├── input/
    │   └── sample_test.jpg           ← 1~2장만, 저작권 확인 필수
    └── README.md                     ← 데이터셋 출처 명시
```

> **권장**: 학위논문 검증에 사용한 KADA/USC 사진은 저작권 이슈가 있을 수 있어 GitHub에 올리지 않는 것이 안전. 대신 `data/README.md`에 출처 URL을 명시하여 사용자가 직접 다운로드하도록 안내.

---

## ❌ 업로드하면 안 되는 파일

### 1. 모델 가중치 (대용량 + 라이선스)

```
weights/                              ← 전체 제외
├── RealESRGAN_x4plus.pth             (64MB)
├── RealESRGAN_x2plus.pth             (64MB)
├── GFPGANv1.4.pth                    (333MB)
└── ColorizeArtistic_gen.pth          (255MB, DeOldify용)
```

**이유**:
- GitHub 단일 파일 제한 100MB (GFPGAN 가중치는 이를 초과)
- 모델 가중치는 각 프로젝트의 공식 릴리스에서 다운로드하는 것이 표준
- README/INSTALL에 다운로드 URL을 명시했으므로 재현성 보장

### 2. 처리 결과물 (재생성 가능)

```
data/output/                          ← 전체 제외
├── batch_20260427_*/
└── *.jpg
```

**이유**: 코드와 입력만 있으면 재생성 가능. 또한 결과 이미지는 KADA/USC 원본의 derived work이므로 저작권 이슈.

### 3. 캐시 / 임시 파일

```
__pycache__/                          ← Python 바이트코드
*.pyc
*.pyo
.ipynb_checkpoints/                   ← Jupyter 캐시
.cache/                               ← 일반 캐시
cache/                                ← 임시 처리 파일
gfpgan/                               ← GFPGAN이 자동 생성하는 결과 폴더 (있다면)
```

### 4. IDE / OS 메타데이터

```
.vscode/
.idea/
*.swp
.DS_Store                             ← macOS
Thumbs.db                             ← Windows
desktop.ini                           ← Windows
```

### 5. 가상환경

```
venv/
env/
.venv/
diaspora/                             ← conda 환경 폴더 자체 (있을 경우)
```

### 6. 개인 정보 / 민감 데이터

```
.env                                  ← 환경변수 (API 키 등)
secrets.yaml
*.pem
```

### 7. 대용량 디버깅 로그

```
*.log
debug_output/
```

---

## 권장 .gitignore 파일

프로젝트 루트에 다음 내용으로 `.gitignore` 파일을 만드세요.

```gitignore
# ==============================================
# Python
# ==============================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/

# ==============================================
# 가상환경
# ==============================================
venv/
env/
.venv/
ENV/

# ==============================================
# IDE / 에디터
# ==============================================
.vscode/
.idea/
*.swp
*.swo
*~

# ==============================================
# OS
# ==============================================
.DS_Store
Thumbs.db
desktop.ini

# ==============================================
# Jupyter
# ==============================================
.ipynb_checkpoints/
*.ipynb

# ==============================================
# 모델 가중치 (대용량, 외부 다운로드)
# ==============================================
weights/
*.pth
*.pt
*.ckpt
*.bin
*.safetensors

# ==============================================
# 처리 결과 (재생성 가능)
# ==============================================
data/output/
data/intermediate/

# ==============================================
# 캐시
# ==============================================
.cache/
cache/
.pytest_cache/
gfpgan/

# ==============================================
# 로그
# ==============================================
*.log
logs/

# ==============================================
# 환경변수
# ==============================================
.env
.env.local
secrets.yaml

# ==============================================
# 입력 데이터 (저작권 보호)
# ==============================================
# data/input/ 의 모든 이미지는 제외하되, README는 포함
data/input/*
!data/input/README.md
!data/input/.gitkeep

# ==============================================
# 일괄 처리 출력
# ==============================================
batch_*.json
batch_*.md
```

---

## requirements.txt 권장 내용

`pip freeze > requirements.txt`로 생성하면 너무 많은 의존성이 잡히므로, 핵심만 정리한 `requirements.txt`를 권장합니다.

```text
# ==============================================
# AI 기반 한인 디아스포라 기록유산 디지털화 시스템
# 이미지 처리 모듈 - Python 의존성
# ==============================================

# PyTorch (CUDA 12.1 빌드는 별도 명령으로 설치)
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 핵심 이미지 처리
opencv-python==4.13.0.86
Pillow>=10.0.0
numpy>=1.24.0
pyyaml>=6.0

# 초해상도 (Real-ESRGAN)
basicsr>=1.4.2
realesrgan>=0.3.0

# 얼굴 복원 (GFPGAN)
gfpgan>=1.3.8
facexlib>=0.3.0

# 얼굴 검출 fallback (HOG)
dlib>=19.24.0
face-recognition>=1.3.0

# 컬러화 (DDColor via modelscope)
modelscope==1.36.2
datasets>=2.0.0
simplejson>=3.18.0
addict>=2.4.0
sortedcontainers>=2.4.0
oss2>=2.17.0
timm>=1.0.0

# 이미지 분석 (BLIP, CLIP)
transformers>=4.30.0
accelerate>=0.20.0
```

---

## data/input/README.md 권장 내용

`data/input/.gitkeep` 파일을 만들어 디렉토리는 유지하되, 실제 이미지는 GitHub에 올리지 않으니, README로 안내합니다.

```markdown
# 입력 이미지 디렉토리

본 디렉토리는 처리할 이미지를 두는 곳입니다. 저작권 보호를 위해 검증에 사용된 KADA/USC 자료는 저장소에 포함되지 않았습니다.

## 검증 데이터셋 출처

본 시스템의 검증에는 다음 두 공식 한인 디아스포라 아카이브의 자료를 사용하였습니다.

- **KADA (Korean American Digital Archive)** — UC Irvine
  - URL: https://www.lib.uci.edu/kada
  
- **USC Korean American Digital Archive** — University of Southern California
  - URL: https://digitallibrary.usc.edu/

## 사용 방법

1. 위 아카이브에서 처리할 이미지를 다운로드하여 본 디렉토리에 저장
2. `python batch_process.py` 명령으로 일괄 처리

지원 형식: jpg, jpeg, png, bmp, tiff, tif, webp
```

---

## 업로드 전 최종 체크리스트

- [ ] `.gitignore` 파일 생성 및 위 내용 적용
- [ ] `weights/` 디렉토리가 git status에 표시되지 않는지 확인
- [ ] `data/output/` 디렉토리가 git status에 표시되지 않는지 확인
- [ ] `__pycache__/` 가 git status에 표시되지 않는지 확인
- [ ] `README.md` 작성 완료 (본 가이드 1번에 포함)
- [ ] `INSTALL.md` 작성 완료 (본 가이드 1번에 포함)
- [ ] `requirements.txt` 작성 완료
- [ ] 코드 안에 개인 경로(`C:\Users\user\...`)가 하드코딩되어 있지 않은지 확인
- [ ] 코드 안에 API 키, 비밀번호 등 민감 정보가 없는지 확인
- [ ] 라이선스 정책 결정 (MIT, Apache 2.0, 또는 학술 전용 등)

---

## 업로드 명령 예시

```powershell
# 1. .gitignore 적용 후 상태 확인
cd C:\Users\<YourName>\downloads\youkyoung\diaspora_thesis_code\03_diaspora_image_processing

# .gitignore 생성 (위 내용 사용)
notepad .gitignore

# 2. git 초기화 (이미 했다면 스킵)
git init
git add .gitignore           # 먼저 .gitignore부터 추가

# 3. 상태 확인 (제외되어야 할 것들이 안 보이는지)
git status

# weights/, data/output/, __pycache__/ 가 안 보이면 정상

# 4. 모든 추적 대상 파일 추가
git add .

# 5. 다시 상태 확인
git status

# 6. 첫 커밋
git commit -m "Initial commit: image processing module"

# 7. GitHub 원격 저장소 연결 (이미 GitHub에서 저장소 생성한 후)
git remote add origin https://github.com/<your_username>/<repo_name>.git
git branch -M main
git push -u origin main
```

---

## 업로드 후 확인

GitHub 저장소 페이지에서 다음을 확인:

1. **README.md가 메인 페이지에 잘 렌더링** 되는지
2. **`weights/` 폴더가 보이지 않는지** (제외 확인)
3. **이미지 파일이 보이지 않는지** (저작권 보호)
4. **저장소 크기가 적절한지** (수십 MB 이내가 정상)

저장소 크기가 1GB를 넘으면 `weights/` 같은 대용량이 잘못 포함된 것이니, 다음 명령으로 정리:

```powershell
# 잘못 추가된 파일 제거 (예: weights 디렉토리)
git rm -r --cached weights/
git commit -m "Remove weights from tracking"
git push
```

---

이상으로 GitHub 업로드 가이드를 마칩니다. README.md, INSTALL.md, requirements.txt를 함께 사용하시면, 다른 사람이 본 저장소를 받아 동일하게 환경을 재구축할 수 있습니다.
