#!/usr/bin/env python3
"""
download_models.py
사전학습 모델 다운로드 스크립트

사용법:
    python download_models.py --all
    python download_models.py --realesrgan --gfpgan
"""

import os
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """파일 다운로드 with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"  ✓ 이미 존재함: {output_path}")
        return True
    
    print(f"  다운로드 중: {desc}")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        print(f"  ✓ 완료: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def download_realesrgan():
    """Real-ESRGAN 모델 다운로드"""
    print("\n[Real-ESRGAN 모델]")
    
    models = {
        'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    }
    
    for name, url in models.items():
        download_file(url, f"models/realesrgan/{name}", name)


def download_gfpgan():
    """GFPGAN 모델 다운로드"""
    print("\n[GFPGAN 모델]")
    
    models = {
        'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
        'GFPGANv1.3.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    }
    
    for name, url in models.items():
        download_file(url, f"models/gfpgan/{name}", name)
    
    # 얼굴 감지 모델
    print("\n[Face Detection 모델]")
    face_models = {
        'detection_Resnet50_Final.pth': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing_parsenet.pth': 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
    }
    
    for name, url in face_models.items():
        download_file(url, f"models/facexlib/{name}", name)


def download_deoldify():
    """DeOldify 모델 다운로드"""
    print("\n[DeOldify 모델]")
    
    models = {
        'ColorizeArtistic_gen.pth': 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth',
        'ColorizeStable_gen.pth': 'https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=1',
    }
    
    for name, url in models.items():
        download_file(url, f"models/deoldify/{name}", name)


def download_blip():
    """BLIP 모델 안내 (HuggingFace에서 자동 다운로드)"""
    print("\n[BLIP 모델]")
    print("  ℹ️ BLIP 모델은 transformers 라이브러리를 통해 자동 다운로드됩니다.")
    print("  모델명: Salesforce/blip-image-captioning-large")


def download_clip():
    """CLIP 모델 안내 (HuggingFace에서 자동 다운로드)"""
    print("\n[CLIP 모델]")
    print("  ℹ️ CLIP 모델은 transformers 라이브러리를 통해 자동 다운로드됩니다.")
    print("  모델명: openai/clip-vit-large-patch14")


def main():
    parser = argparse.ArgumentParser(description="사전학습 모델 다운로드")
    parser.add_argument("--all", action="store_true", help="모든 모델 다운로드")
    parser.add_argument("--realesrgan", action="store_true", help="Real-ESRGAN 모델")
    parser.add_argument("--gfpgan", action="store_true", help="GFPGAN 모델")
    parser.add_argument("--deoldify", action="store_true", help="DeOldify 모델")
    parser.add_argument("--info", action="store_true", help="모델 정보만 표시")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI 이미지 복원 - 사전학습 모델 다운로드")
    print("=" * 60)
    
    if args.info:
        download_blip()
        download_clip()
        return
    
    if args.all or args.realesrgan:
        download_realesrgan()
    
    if args.all or args.gfpgan:
        download_gfpgan()
    
    if args.all or args.deoldify:
        download_deoldify()
    
    if args.all:
        download_blip()
        download_clip()
    
    if not any([args.all, args.realesrgan, args.gfpgan, args.deoldify]):
        print("\n사용법:")
        print("  python download_models.py --all        # 모든 모델")
        print("  python download_models.py --realesrgan # Real-ESRGAN만")
        print("  python download_models.py --gfpgan     # GFPGAN만")
        print("  python download_models.py --info       # 모델 정보")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
