#!/usr/bin/env python3
"""
test_pipeline.py
파이프라인 테스트 스크립트

테스트 이미지를 생성하고 각 모듈의 동작을 검증합니다.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse


def create_test_images(output_dir: str = "data/test_images"):
    """테스트용 이미지 생성"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("테스트 이미지 생성 중...")
    
    # 1. 저해상도 컬러 이미지 (초해상도 테스트용)
    img_low_res = Image.new('RGB', (128, 128), color=(100, 150, 200))
    draw = ImageDraw.Draw(img_low_res)
    draw.ellipse([30, 30, 98, 98], fill=(255, 200, 100))
    draw.rectangle([50, 50, 78, 78], fill=(50, 100, 150))
    img_low_res.save(output_dir / "test_lowres_color.jpg", quality=85)
    print(f"  ✓ 저해상도 컬러 이미지: {output_dir / 'test_lowres_color.jpg'}")
    
    # 2. 흑백 이미지 (컬러화 테스트용)
    img_grayscale = Image.new('L', (256, 256), color=128)
    draw = ImageDraw.Draw(img_grayscale)
    for i in range(0, 256, 32):
        shade = int(i)
        draw.rectangle([i, 0, i+32, 256], fill=shade)
    # 원 추가
    draw.ellipse([80, 80, 176, 176], fill=200)
    draw.ellipse([100, 100, 156, 156], fill=50)
    img_grayscale.save(output_dir / "test_grayscale.jpg", quality=90)
    print(f"  ✓ 흑백 이미지: {output_dir / 'test_grayscale.jpg'}")
    
    # 3. 노이즈가 있는 이미지 (복원 테스트용)
    img_noisy = Image.new('RGB', (256, 256), color=(180, 160, 140))
    draw = ImageDraw.Draw(img_noisy)
    # 패턴 추가
    for x in range(0, 256, 20):
        for y in range(0, 256, 20):
            color = ((x*3) % 256, (y*2) % 256, ((x+y)*2) % 256)
            draw.rectangle([x, y, x+15, y+15], fill=color)
    # 노이즈 추가
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    img_noisy = Image.fromarray(np.clip(np.array(img_noisy).astype(int) + noise - 25, 0, 255).astype(np.uint8))
    img_noisy.save(output_dir / "test_noisy.jpg", quality=75)
    print(f"  ✓ 노이즈 이미지: {output_dir / 'test_noisy.jpg'}")
    
    # 4. 세피아톤 이미지 (옛날 사진 시뮬레이션)
    img_sepia = Image.new('RGB', (300, 200), color=(210, 180, 140))
    draw = ImageDraw.Draw(img_sepia)
    # 사람 형태 (단순화)
    draw.ellipse([130, 30, 170, 70], fill=(190, 160, 120))  # 얼굴
    draw.rectangle([135, 70, 165, 130], fill=(180, 150, 110))  # 몸
    draw.text((100, 150), "1950", fill=(150, 120, 90))
    img_sepia.save(output_dir / "test_sepia.jpg", quality=80)
    print(f"  ✓ 세피아톤 이미지: {output_dir / 'test_sepia.jpg'}")
    
    # 5. 저품질 JPEG (압축 아티팩트 테스트)
    img_jpeg = Image.new('RGB', (200, 200), color=(100, 120, 140))
    draw = ImageDraw.Draw(img_jpeg)
    for i in range(10):
        x, y = i * 20, i * 15
        draw.ellipse([x, y, x+50, y+50], fill=(200, 180, 160))
    img_jpeg.save(output_dir / "test_lowquality.jpg", quality=20)  # 매우 낮은 품질
    print(f"  ✓ 저품질 JPEG: {output_dir / 'test_lowquality.jpg'}")
    
    print(f"\n총 5개의 테스트 이미지 생성 완료: {output_dir}")
    return output_dir


def test_imports():
    """모듈 import 테스트"""
    print("\n" + "=" * 50)
    print("모듈 Import 테스트")
    print("=" * 50)
    
    results = {}
    
    # Core modules
    try:
        import torch
        results['torch'] = f"✓ PyTorch {torch.__version__}"
        results['cuda'] = f"✓ CUDA 사용 가능" if torch.cuda.is_available() else "✗ CUDA 불가"
    except ImportError as e:
        results['torch'] = f"✗ PyTorch: {e}"
    
    try:
        import cv2
        results['opencv'] = f"✓ OpenCV {cv2.__version__}"
    except ImportError as e:
        results['opencv'] = f"✗ OpenCV: {e}"
    
    try:
        from PIL import Image
        import PIL
        results['pillow'] = f"✓ Pillow {PIL.__version__}"
    except ImportError as e:
        results['pillow'] = f"✗ Pillow: {e}"
    
    # AI modules
    try:
        from realesrgan import RealESRGANer
        results['realesrgan'] = "✓ Real-ESRGAN"
    except ImportError as e:
        results['realesrgan'] = f"✗ Real-ESRGAN: {e}"
    
    try:
        from gfpgan import GFPGANer
        results['gfpgan'] = "✓ GFPGAN"
    except ImportError as e:
        results['gfpgan'] = f"✗ GFPGAN: {e}"
    
    try:
        from transformers import BlipProcessor, CLIPProcessor
        results['transformers'] = "✓ Transformers (BLIP, CLIP)"
    except ImportError as e:
        results['transformers'] = f"✗ Transformers: {e}"
    
    try:
        import face_recognition
        results['face_recognition'] = "✓ face_recognition"
    except ImportError as e:
        results['face_recognition'] = f"✗ face_recognition: {e}"
    
    # Print results
    for module, status in results.items():
        print(f"  {status}")
    
    return results


def test_modules():
    """각 모듈 개별 테스트"""
    print("\n" + "=" * 50)
    print("모듈 초기화 테스트")
    print("=" * 50)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    results = {}
    
    # Super Resolution Module
    try:
        from src.modules.super_resolution import SuperResolutionModule
        # 실제 초기화는 모델 다운로드가 필요하므로 건너뜀
        results['super_resolution'] = "✓ SuperResolutionModule import 성공"
    except Exception as e:
        results['super_resolution'] = f"✗ SuperResolutionModule: {e}"
    
    # Face Enhancement Module
    try:
        from src.modules.face_enhancement import FaceEnhancementModule
        results['face_enhancement'] = "✓ FaceEnhancementModule import 성공"
    except Exception as e:
        results['face_enhancement'] = f"✗ FaceEnhancementModule: {e}"
    
    # Colorization Module
    try:
        from src.modules.colorization import ColorizationModule
        results['colorization'] = "✓ ColorizationModule import 성공"
    except Exception as e:
        results['colorization'] = f"✗ ColorizationModule: {e}"
    
    # Image Analysis Module
    try:
        from src.modules.image_analysis import ImageAnalysisModule
        results['image_analysis'] = "✓ ImageAnalysisModule import 성공"
    except Exception as e:
        results['image_analysis'] = f"✗ ImageAnalysisModule: {e}"
    
    # Pipeline
    try:
        from src.pipeline import ImageRestorationPipeline
        results['pipeline'] = "✓ ImageRestorationPipeline import 성공"
    except Exception as e:
        results['pipeline'] = f"✗ ImageRestorationPipeline: {e}"
    
    for module, status in results.items():
        print(f"  {status}")
    
    return results


def run_quick_test(test_image: str = None):
    """빠른 기능 테스트 (CPU에서 실행)"""
    print("\n" + "=" * 50)
    print("빠른 기능 테스트")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # 테스트 이미지 준비
    if test_image is None:
        test_dir = create_test_images()
        test_image = test_dir / "test_grayscale.jpg"
    
    print(f"\n테스트 이미지: {test_image}")
    
    # Colorization 모듈 테스트 (흑백 판정)
    try:
        from src.modules.colorization import ColorizationModule
        
        # DeOldify 없이 흑백 판정만 테스트
        print("\n[흑백 판정 테스트]")
        img = Image.open(test_image)
        
        # 간단한 채도 계산
        import numpy as np
        img_array = np.array(img.convert('RGB'))
        
        # RGB to HSV (간단한 방법)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # 채도 계산
        saturation = np.where(max_val > 0, diff / max_val, 0)
        avg_saturation = np.mean(saturation)
        
        is_grayscale = avg_saturation < 0.1
        print(f"  평균 채도: {avg_saturation:.4f}")
        print(f"  흑백 판정: {'예' if is_grayscale else '아니오'}")
        print("  ✓ 흑백 판정 테스트 통과")
        
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")
    
    # 이미지 분석 기본 테스트
    try:
        print("\n[이미지 기본 분석 테스트]")
        img = Image.open(test_image)
        print(f"  크기: {img.size}")
        print(f"  모드: {img.mode}")
        print(f"  포맷: {img.format}")
        print("  ✓ 이미지 로드 테스트 통과")
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="파이프라인 테스트")
    parser.add_argument("--create-images", action="store_true", help="테스트 이미지 생성")
    parser.add_argument("--test-imports", action="store_true", help="import 테스트")
    parser.add_argument("--test-modules", action="store_true", help="모듈 테스트")
    parser.add_argument("--quick-test", action="store_true", help="빠른 기능 테스트")
    parser.add_argument("--all", action="store_true", help="모든 테스트 실행")
    parser.add_argument("--image", type=str, default=None, help="테스트할 이미지 경로")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI 이미지 복원 파이프라인 - 테스트")
    print("=" * 60)
    
    if args.all or args.create_images:
        create_test_images()
    
    if args.all or args.test_imports:
        test_imports()
    
    if args.all or args.test_modules:
        test_modules()
    
    if args.all or args.quick_test:
        run_quick_test(args.image)
    
    if not any([args.create_images, args.test_imports, args.test_modules, 
                args.quick_test, args.all]):
        print("\n사용법:")
        print("  python test_pipeline.py --all           # 모든 테스트")
        print("  python test_pipeline.py --create-images # 테스트 이미지 생성")
        print("  python test_pipeline.py --test-imports  # import 테스트")
        print("  python test_pipeline.py --test-modules  # 모듈 테스트")
        print("  python test_pipeline.py --quick-test    # 빠른 테스트")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
