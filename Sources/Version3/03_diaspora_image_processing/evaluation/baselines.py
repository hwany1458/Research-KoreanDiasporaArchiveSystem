"""
baselines.py
Baseline 방법론 모듈

학위논문에서 우리 시스템과 공정한 비교(fair baseline comparison)를 위한
baseline 처리 방법들을 제공합니다.

Baseline 단계:
    Level 1: Bicubic interpolation (가장 단순한 SR baseline)
    Level 2: Real-ESRGAN 단독 (최신 SR 단일 모델)
    Level 3: 우리 통합 시스템 (SR + 얼굴 복원 + 컬러화)

이 점진적 비교는 메모리에 명시된 "공정한 baseline 비교" mandatory 이슈에 
직접 대응합니다.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import cv2
import numpy as np
from PIL import Image


def _imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """한글 경로 안전 이미지 로더."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_unicode(path: Union[str, Path], img: np.ndarray,
                     quality: int = 95) -> bool:
    """한글 경로 안전 이미지 저장."""
    path = str(path)
    ext = os.path.splitext(path)[1] or '.jpg'
    encode_params = []
    if ext.lower() in ('.jpg', '.jpeg'):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    try:
        success, encoded = cv2.imencode(ext, img, encode_params)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────
# Level 1: Bicubic Interpolation
# ──────────────────────────────────────────────────────────────────
class BicubicBaseline:
    """
    Bicubic interpolation 기반 단순 업스케일링.
    
    SR 분야의 가장 단순한 baseline. 학습된 모델 대비
    얼마나 개선되는지를 보여주는 기준선.
    """
    
    def __init__(self, scale: int = 4):
        self.scale = scale
    
    def upscale(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Bicubic으로 4배 업스케일."""
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image)
            if img is None:
                raise FileNotFoundError(f"이미지 로드 실패: {image}")
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        new_h, new_w = h * self.scale, w * self.scale
        
        upscaled = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _imwrite_unicode(output_path, upscaled, quality=95)
        
        return {
            'method': 'bicubic',
            'result': upscaled,
            'original_size': (w, h),
            'output_size': (new_w, new_h),
            'scale': self.scale
        }
    
    def upscale_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """디렉토리 내 모든 이미지에 Bicubic 적용."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])
        
        succeeded = 0
        for img_path in images:
            try:
                output_path = output_dir / f"{img_path.stem}_bicubic.jpg"
                self.upscale(img_path, output_path)
                succeeded += 1
            except Exception as e:
                print(f"  ⚠ {img_path.name} 실패: {e}")
        
        return {
            'method': 'bicubic',
            'total': len(images),
            'succeeded': succeeded,
            'output_dir': str(output_dir)
        }


# ──────────────────────────────────────────────────────────────────
# Level 2: Real-ESRGAN 단독 (얼굴 복원·컬러화 없이 SR만)
# ──────────────────────────────────────────────────────────────────
class RealESRGANBaseline:
    """
    Real-ESRGAN만 단독으로 적용 (우리 시스템에서 SR 단계만).
    
    이 baseline과 우리 통합 시스템을 비교하면,
    얼굴 복원·컬러화·분석 단계가 추가로 가져오는 효과를 정량화 가능.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        model_path: Optional[str] = None
    ):
        # 프로젝트의 super_resolution 모듈 재사용
        # (PYTHONPATH가 프로젝트 루트를 포함해야 함)
        try:
            from src.modules.super_resolution import SuperResolutionModule
        except ImportError:
            # evaluation/ 디렉토리에서 실행되는 경우 상위 경로 추가
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root))
            from src.modules.super_resolution import SuperResolutionModule
        
        self.sr_module = SuperResolutionModule(
            device=device,
            model_path=model_path,
            always_apply=True,
            tile_size=1024
        )
    
    def upscale(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Real-ESRGAN으로 4배 업스케일."""
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image)
            if img is None:
                raise FileNotFoundError(f"이미지 로드 실패: {image}")
        else:
            img = image
        
        result = self.sr_module.enhance(img)
        enhanced_pil = result['enhanced']
        
        # PIL → BGR numpy
        enhanced_rgb = np.array(enhanced_pil)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _imwrite_unicode(output_path, enhanced_bgr, quality=95)
        
        return {
            'method': 'real_esrgan',
            'result': enhanced_bgr,
            'original_size': result['original_size'],
            'output_size': result['enhanced_size'],
            'scale': result['scale']
        }
    
    def upscale_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """디렉토리 내 모든 이미지에 Real-ESRGAN 적용."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])
        
        succeeded = 0
        for img_path in images:
            try:
                output_path = output_dir / f"{img_path.stem}_realesrgan.jpg"
                self.upscale(img_path, output_path)
                succeeded += 1
            except Exception as e:
                print(f"  ⚠ {img_path.name} 실패: {e}")
        
        return {
            'method': 'real_esrgan',
            'total': len(images),
            'succeeded': succeeded,
            'output_dir': str(output_dir)
        }


# ──────────────────────────────────────────────────────────────────
# Level 3: 우리 통합 시스템 (참조용 thin wrapper)
# ──────────────────────────────────────────────────────────────────
class FullPipelineBaseline:
    """
    우리 통합 시스템을 baseline 평가에 맞춰 호출하는 wrapper.
    
    실제로는 src.pipeline.ImageRestorationPipeline.process()를 호출하지만,
    평가 컨텍스트에서 "method='full_pipeline'" 식의 일관된 인터페이스 제공.
    """
    
    def __init__(self, device: str = 'cuda'):
        try:
            from src.pipeline import ImageRestorationPipeline
        except ImportError:
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root))
            from src.pipeline import ImageRestorationPipeline
        
        self.pipeline = ImageRestorationPipeline(
            device=device,
            lazy_load=True
        )
    
    def process(
        self,
        image: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """우리 통합 시스템으로 처리."""
        result = self.pipeline.process(
            input_path=image,
            output_path=output_path
        )
        return {
            'method': 'full_pipeline',
            'success': result.success,
            'original_size': result.original_size,
            'output_size': result.final_size,
            'stages_applied': result.stages_applied,
            'total_time': result.total_time
        }
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """디렉토리 일괄 처리."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])
        
        results = self.pipeline.process_batch(
            input_paths=images,
            output_dir=output_dir
        )
        
        succeeded = sum(1 for r in results if r.success)
        return {
            'method': 'full_pipeline',
            'total': len(images),
            'succeeded': succeeded,
            'output_dir': str(output_dir),
            'results': results
        }


# ──────────────────────────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline 방법 적용")
    parser.add_argument('--method', required=True,
                        choices=['bicubic', 'real_esrgan', 'full_pipeline'])
    parser.add_argument('--input', '-i', required=True, help='입력 디렉토리')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    if args.method == 'bicubic':
        baseline = BicubicBaseline(scale=4)
        result = baseline.upscale_directory(args.input, args.output)
    elif args.method == 'real_esrgan':
        baseline = RealESRGANBaseline(device=args.device)
        result = baseline.upscale_directory(args.input, args.output)
    elif args.method == 'full_pipeline':
        baseline = FullPipelineBaseline(device=args.device)
        result = baseline.process_directory(args.input, args.output)
    
    print(f"처리 완료: 성공 {result['succeeded']}/{result['total']}")
