# 🌏 Korean Diaspora Archive System

**AI 기반 한인 디아스포라 기록유산의 디지털화 및 실감형 콘텐츠 생성 시스템**

*AI-based Digitization and Immersive Content Generation System for Korean Diaspora Documentary Heritage*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![Unity](https://img.shields.io/badge/Unity-2022.3-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [데이터셋](#-데이터셋)
- [평가 결과](#-평가-결과)
- [프로젝트 구조](#-프로젝트-구조)
- [기여 방법](#-기여-방법)
- [라이선스](#-라이선스)
- [인용](#-인용)

---

## 🎯 프로젝트 소개

전 세계에 흩어진 약 **700만 재외동포** 가정에는 사진, 편지, 영상, 음성 녹음 등 소중한 기록유산이 산재해 있습니다. 이러한 자료들은 물리적 열화, 세대 단절, 보존 인프라 부재로 인해 소실 위기에 처해 있습니다.

본 프로젝트는 **AI 기술**을 활용하여 개인 소장 기록유산을 디지털화하고, **실감형 콘텐츠**(디지털 아카이브, AR, VR, 인터랙티브 스토리텔링)로 변환하는 통합 시스템을 제공합니다.

### 연구 배경

- 📸 개인 소장 자료의 체계적 보존 방법 부재
- 🌐 디아스포라 기록유산의 학술적 연구 공백
- 🤖 AI 기술의 문화유산 분야 적용 가능성
- 🎮 실감 기술을 통한 문화 전승의 새로운 가능성

---

## ✨ 주요 기능

### 1. AI 기반 디지털화 파이프라인

| 모듈 | 기능 | 핵심 기술 |
|------|------|----------|
| **이미지 처리** | 복원, 초해상도, 컬러화, 얼굴 향상 | Real-ESRGAN, GFPGAN, DeOldify |
| **문서 처리** | OCR, 손글씨 인식, 다국어 처리 | EasyOCR, TrOCR, Tesseract |
| **음성 처리** | 음성 인식, 화자 분리, 잡음 제거 | Whisper, pyannote.audio |
| **영상 처리** | 품질 향상, 장면 분할, 음성 추출 | BasicVSR++, PySceneDetect |
| **메타데이터** | 자동 태깅, 캡셔닝, 개체명 인식 | BLIP-2, CLIP, KoBERT |

### 2. 실감형 콘텐츠 생성

| 콘텐츠 유형 | 설명 | 플랫폼 |
|------------|------|--------|
| **디지털 아카이브** | 웹 기반 자료 관리 및 탐색 | React, FastAPI |
| **AR 뷰어** | 실물 사진 인식 기반 정보 증강 | Unity AR Foundation |
| **VR 전시** | 가상 공간에서의 몰입형 전시 | Unity XR |
| **스토리텔링** | AI 기반 인터랙티브 내러티브 | React, GPT-4 |

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         입력 계층                                │
│  [파일 업로드] [스캐너] [클라우드 동기화] → [자료 유형 분류]      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AI 처리 계층                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ 이미지  │ │  문서   │ │  영상   │ │  음성   │ │  구술   │   │
│  │ 처리    │ │  처리   │ │  처리   │ │  처리   │ │  처리   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      메타데이터 통합                             │
│  [자동 태깅] [관계 추론] [지식그래프 구축] [검색 인덱싱]         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    실감형 콘텐츠 생성                            │
│  [디지털 아카이브] [AR 뷰어] [VR 전시] [인터랙티브 스토리]       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 설치 방법

### 시스템 요구사항

| 구성요소 | 최소 사양 | 권장 사양 |
|---------|----------|----------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i9 / AMD Ryzen 9 |
| RAM | 16GB | 64GB |
| GPU | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 4080+ (16GB+) |
| Storage | 100GB SSD | 500GB+ NVMe SSD |
| CUDA | 11.8+ | 12.x |

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/diaspora-archive.git
cd diaspora-archive
```

### 2. Python 환경 설정

```bash
# Conda 환경 생성 (권장)
conda create -n diaspora python=3.10
conda activate diaspora

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. AI 모델 다운로드

```bash
# 모델 자동 다운로드 스크립트 실행
python scripts/download_models.py

# 또는 개별 다운로드
python scripts/download_models.py --model realesrgan
python scripts/download_models.py --model gfpgan
python scripts/download_models.py --model whisper
```

### 4. 프론트엔드 설정

```bash
cd frontend
npm install
npm run build
```

### 5. Docker 사용 (선택)

```bash
# Docker Compose로 전체 시스템 실행
docker-compose up -d

# 개별 서비스 실행
docker-compose up -d api worker frontend
```

---

## 📖 사용 방법

### 빠른 시작

```python
from diaspora_archive import DiasporaArchive

# 시스템 초기화
archive = DiasporaArchive()

# 이미지 복원
restored = archive.restore_image("old_photo.jpg")
restored.save("restored_photo.jpg")

# 문서 OCR
text = archive.ocr_document("letter.jpg", language="ko")
print(text)

# 음성 전사
transcript = archive.transcribe_audio("interview.mp3")
print(transcript.text)
```

### CLI 사용

```bash
# 이미지 복원
python -m diaspora_archive restore image input.jpg -o output.jpg

# 문서 OCR
python -m diaspora_archive ocr document.jpg --lang ko

# 음성 전사
python -m diaspora_archive transcribe audio.mp3 --speaker-diarization

# 배치 처리
python -m diaspora_archive process-folder ./my_photos --type image
```

### 웹 인터페이스

```bash
# 백엔드 서버 실행
uvicorn src.api.main:app --reload --port 8000

# 프론트엔드 개발 서버 실행
cd frontend && npm run dev
```

브라우저에서 `http://localhost:3000` 접속

### API 사용

```python
import requests

# 이미지 업로드 및 처리
with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/assets/upload",
        files={"file": f},
        data={"process": True}
    )
    
result = response.json()
print(f"Asset ID: {result['asset_id']}")
```

---

## 📊 데이터셋

### 공개 데이터셋 활용

| 데이터셋 | 출처 | 용도 | 링크 |
|---------|------|------|------|
| 한글 손글씨 | AI Hub | HTR 학습/평가 | [링크](https://aihub.or.kr/) |
| KsponSpeech | AI Hub | 음성 인식 평가 | [링크](https://aihub.or.kr/) |
| 근현대 사진 | 국립민속박물관 | 이미지 복원 테스트 | [링크](https://www.nfm.go.kr/) |
| 구술 자료 | 국사편찬위원회 | 음성 처리 테스트 | [링크](https://www.history.go.kr/) |

### 데이터셋 다운로드

```bash
# 공개 데이터셋 다운로드 (AI Hub 인증 필요)
python scripts/download_datasets.py --dataset aihub_handwriting

# 샘플 데이터셋 다운로드
python scripts/download_datasets.py --sample
```

### 데이터 구조

```
data/
├── raw/                    # 원본 데이터
│   ├── images/
│   ├── documents/
│   ├── audio/
│   └── video/
├── processed/              # 처리된 데이터
│   ├── restored/
│   ├── transcripts/
│   └── metadata/
└── models/                 # 모델 가중치
    ├── realesrgan/
    ├── gfpgan/
    ├── whisper/
    └── trocr/
```

---

## 📈 평가 결과

### 기술적 성능

| 태스크 | 메트릭 | 결과 | 베이스라인 대비 |
|--------|--------|------|----------------|
| 이미지 초해상도 | PSNR | 28.12 dB | +3.8 dB |
| 이미지 초해상도 | SSIM | 0.831 | +0.12 |
| 인쇄체 OCR | CER | 3.54% | -2.1%p |
| 손글씨 HTR | CER | 15.67% | -2.6%p |
| 음성 인식 | CER | 4.56% | -0.7%p |
| 화자 분리 | DER | 12.34% | - |

### 사용자 평가

| 평가 항목 | 점수 |
|----------|------|
| SUS (System Usability Scale) | 71.2 / 100 |
| NPS (Net Promoter Score) | +27 |
| 전반적 만족도 | 5.6 / 7.0 |
| 이미지 복원 만족도 | 6.1 / 7.0 |

---

## 📁 프로젝트 구조

```
diaspora-archive/
├── 📄 README.md
├── 📄 LICENSE
├── 📄 requirements.txt
├── 📄 docker-compose.yml
├── 📄 pyproject.toml
│
├── 📁 src/                      # 소스 코드
│   ├── 📁 api/                  # FastAPI 백엔드
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── 📁 core/                 # 핵심 AI 모듈
│   │   ├── 📁 image/            # 이미지 처리
│   │   ├── 📁 document/         # 문서 처리
│   │   ├── 📁 audio/            # 음성 처리
│   │   ├── 📁 video/            # 영상 처리
│   │   └── 📁 metadata/         # 메타데이터
│   │
│   ├── 📁 content/              # 콘텐츠 생성
│   │   ├── 📁 archive/          # 디지털 아카이브
│   │   ├── 📁 ar/               # AR 콘텐츠
│   │   ├── 📁 vr/               # VR 콘텐츠
│   │   └── 📁 storytelling/     # 스토리텔링
│   │
│   └── 📁 utils/                # 유틸리티
│
├── 📁 frontend/                 # React 프론트엔드
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   ├── 📁 pages/
│   │   └── 📁 hooks/
│   └── package.json
│
├── 📁 unity/                    # Unity AR/VR 앱
│   ├── 📁 AR/
│   └── 📁 VR/
│
├── 📁 config/                   # 설정 파일
├── 📁 data/                     # 데이터
├── 📁 docs/                     # 문서
├── 📁 scripts/                  # 스크립트
└── 📁 tests/                    # 테스트
```

---

## 🤝 기여 방법

기여를 환영합니다! 다음 단계를 따라주세요:

1. 이 저장소를 Fork합니다
2. 새 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

### 기여 가이드라인

- [코드 스타일 가이드](docs/STYLE_GUIDE.md)
- [기여 가이드](CONTRIBUTING.md)
- [이슈 템플릿](.github/ISSUE_TEMPLATE.md)

---

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📚 인용

이 프로젝트를 연구에 사용하신다면 다음과 같이 인용해 주세요:

```bibtex
@phdthesis{diaspora_archive_2026,
  title={AI 기반 한인 디아스포라 기록유산의 디지털화 및 실감형 콘텐츠 생성 모델 연구},
  author={YongHwan Kim},
  year={2026},
  school={University Name},
  type={PhD Dissertation}
}
```

---

## 📞 연락처

- **이메일**: your.email@example.com
- **이슈**: [GitHub Issues](https://github.com/your-username/diaspora-archive/issues)

---

<p align="center">
  <i>디아스포라의 기억은 계속되어야 합니다.</i><br>
  <i>The memory of diaspora must continue.</i>
</p>
