"""
super_resolution.py
초해상도 복원 모듈 - Real-ESRGAN 기반

Real-ESRGAN을 활용하여 저해상도 이미지를 4배 업스케일링합니다.

Reference:
    Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Blind 
    Super-Resolution with Pure Synthetic Data. ICCVW.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path

# Real-ESRGAN imports
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    REALESRGAN_AVAILABLE = False
    print(f"[WARNING] Real-ESRGAN import 실패: {e}")
except Exception as e:
    REALESRGAN_AVAILABLE = False
    print(f"[WARNING] Real-ESRGAN 로드 중 예외 발생: {e}")


def _find_model_path(model_name: str) -> Optional[str]:
    """
    모델 파일(.pth)을 여러 경로에서 자동으로 탐색합니다.
    
    탐색 순서:
      1. 현재 작업 디렉토리 기준 models/realesrgan/
      2. 이 파일 위치 기준 ../../models/realesrgan/
      3. 환경변수 REALESRGAN_MODEL_DIR
    """
    filename = f"{model_name}.pth"
    
    candidates = [
        # 1. 프로젝트 루트 기준
        os.path.join("models", "realesrgan", filename),
        # 2. 이 소스파일 위치 기준 (src/modules/ → ../../models/)
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "realesrgan", filename),
        # 3. 환경변수
        os.path.join(os.environ.get("REALESRGAN_MODEL_DIR", ""), filename),
    ]
    
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    return None


class SuperResolutionModule:
    """
    초해상도 복원 모듈 - Real-ESRGAN 기반
    
    RRDB 아키텍처를 사용하여 실제 환경의 다양한 열화에 강건한 성능을 제공합니다.
    """
    
    SUPPORTED_MODELS = {
        'RealESRGAN_x4plus': {'scale': 4, 'num_block': 23},
        'RealESRGAN_x2plus': {'scale': 2, 'num_block': 23},
    }
    
    def __init__(
        self,
        device: str = 'cuda',
        model_name: str = 'RealESRGAN_x4plus',
        model_path: Optional[str] = None,
        tile_size: int = 512,
        half_precision: bool = True
    ):
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")
        
        # 모델 경로 자동 탐색 (model_path가 명시적으로 주어지지 않은 경우)
        if model_path is None:
            model_path = _find_model_path(model_name)
            if model_path:
                print(f"  [SR] 모델 경로 자동 탐색 성공: {model_path}")
            else:
                raise FileNotFoundError(
                    f"모델 파일을 찾을 수 없습니다: {model_name}.pth\n"
                    f"다음 경로를 확인하세요: models/realesrgan/{model_name}.pth\n"
                    f"모델 다운로드: python download_models.py --realesrgan"
                )
        
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tile_size = tile_size
        self.half_precision = half_precision and self.device == 'cuda'
        
        model_config = self.SUPPORTED_MODELS.get(model_name, self.SUPPORTED_MODELS['RealESRGAN_x4plus'])
        self.scale = model_config['scale']
        
        # 모델 초기화
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=model_config['num_block'], num_grow_ch=32, scale=self.scale)
        
        self.upsampler = RealESRGANer(
            scale=self.scale, model_path=model_path, model=model,
            tile=tile_size, tile_pad=10, pre_pad=0,
            half=self.half_precision, device=self.device
        )
        
        print(f"✓ SuperResolutionModule 초기화 완료 (모델: {model_name}, 스케일: {self.scale}x, 장치: {self.device})")
    
    def should_process(self, image: Union[str, Path, Image.Image, np.ndarray],
                       min_resolution: int = 512, max_resolution: int = 2048) -> Tuple[bool, str]:
        """이미지가 초해상도 처리가 필요한지 판단"""
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            width, height = img.size
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        min_dim, max_dim = min(width, height), max(width, height)
        
        if max_dim > max_resolution:
            return False, f"해상도가 충분히 높음 ({width}x{height})"
        if min_dim < min_resolution:
            return True, f"저해상도 이미지 ({width}x{height})"
        return False, f"해상도가 적절함 ({width}x{height})"
    
    def enhance(self, image: Union[str, Path, Image.Image, np.ndarray],
                outscale: Optional[float] = None) -> Dict[str, Any]:
        """이미지 초해상도 복원 수행"""
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        original_size = (img.shape[1], img.shape[0])
        outscale = outscale or self.scale
        
        # Real-ESRGAN 적용
        output, _ = self.upsampler.enhance(img, outscale=outscale)
        
        # PIL Image로 변환
        enhanced_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return {
            'enhanced': enhanced_pil,
            'original': original_pil,
            'scale': outscale,
            'original_size': original_size,
            'enhanced_size': enhanced_pil.size
        }
