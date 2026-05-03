# Model Weights

이 폴더에는 본 시스템이 사용하는 사전 훈련된 모델 가중치 파일을 저장합니다.

> ⚠️ **가중치 파일은 GitHub repo에 포함되어 있지 않습니다.**  
> 파일 크기 (~400MB)와 저작권을 고려하여 사용자가 직접 다운로드해야 합니다.

## 필수 가중치

다음 두 파일을 이 폴더에 다운로드하세요:

### 1. RealESRGAN_x4plus.pth (~64MB)

초해상도 모델 가중치.

**다운로드 URL**: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

```bash
# Linux/macOS
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "RealESRGAN_x4plus.pth"
```

### 2. GFPGANv1.4.pth (~348MB)

얼굴 향상 모델 가중치.

**다운로드 URL**: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

```bash
# Linux/macOS
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "GFPGANv1.4.pth"
```

## 선택 가중치

### RealESRGAN_x2plus.pth (~64MB) — 배경 업샘플러용

`face_enhancement.py`의 `bg_upsampler='realesrgan'` 옵션을 활성화할 때만 필요. 현재 default 설정에서는 사용되지 않습니다.

**다운로드 URL**: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

## 자동 다운로드되는 가중치 (별도 작업 불필요)

다음 모델들은 **첫 실행 시** 라이브러리에 의해 자동으로 다운로드되어 시스템 캐시에 저장됩니다. 본 폴더에 저장할 필요 없습니다.

| 모델 | 크기 | 저장 위치 |
|---|---|---|
| DDColor | ~870MB | `~/.cache/modelscope/hub/damo/cv_ddcolor_image-colorization/` |
| RetinaFace ResNet50 | ~110MB | `gfpgan/weights/` (또는 `~/.cache/torch/hub/`) |
| Parsing Parsenet | ~85MB | `gfpgan/weights/` |
| BLIP-large | ~1.8GB | `~/.cache/huggingface/hub/` |
| CLIP ViT-Large/14 | ~1.7GB | `~/.cache/huggingface/hub/` |
| face_recognition_models | ~100MB | site-packages 내부 |

## 폴더 구조 확인

다운로드 완료 후 폴더 구조:

```
weights/
├── README.md                    (이 파일)
├── RealESRGAN_x4plus.pth        (필수)
├── GFPGANv1.4.pth               (필수)
└── RealESRGAN_x2plus.pth        (선택)
```

## 라이선스

각 모델 가중치는 원 모델의 라이선스를 따릅니다:

- Real-ESRGAN: [BSD 3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
- GFPGAN: [Apache License 2.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
- DDColor: [Apache License 2.0](https://github.com/piddnad/DDColor/blob/master/LICENSE)

상업적 사용 전 각 라이선스를 반드시 확인하세요.
