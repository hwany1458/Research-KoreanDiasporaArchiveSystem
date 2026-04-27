"""
super_resolution.py
초해상도 복원 모듈 - Real-ESRGAN 기반

Real-ESRGAN을 활용하여 저해상도 이미지를 4배 업스케일링합니다.
디아스포라 아카이브 표준화된 고해상도 보존을 위해 기본 정책은
'모든 입력 이미지에 SR 적용'입니다.

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
except ImportError:
    REALESRGAN_AVAILABLE = False


def _resolve_default_weight_path(model_name: str) -> str:
    """
    프로젝트 루트의 weights/ 폴더에서 모델 가중치 파일 경로를 계산.
    
    이 파일 위치: <project_root>/src/modules/super_resolution.py
    가중치 파일 : <project_root>/weights/<model_name>.pth
    """
    here = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(project_root, 'weights', f'{model_name}.pth')


def _imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    한글 경로 대응 이미지 로더.
    OpenCV의 cv2.imread()는 Windows에서 한글이 포함된 경로를 처리하지 못하므로
    numpy.fromfile() + cv2.imdecode() 조합으로 대체한다.
    """
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


class SuperResolutionModule:
    """
    초해상도 복원 모듈 - Real-ESRGAN 기반
    
    RRDB 아키텍처를 사용하여 실제 환경의 다양한 열화에 강건한 성능을 제공합니다.
    
    정책:
        디아스포라 아카이브의 표준화된 고해상도 보존을 위해
        기본적으로 모든 입력 이미지에 SR을 적용합니다(always_apply=True).
        ablation study 등을 위해서는 always_apply=False로 설정 후
        min_resolution / max_resolution 임계값으로 조정할 수 있습니다.
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
        tile_size: int = 1024,
        half_precision: bool = True,
        always_apply: bool = True,
        max_megapixels: float = 24.0
    ):
        """
        Args:
            device: 연산 장치 ('cuda', 'cpu')
            model_name: 모델 이름 ('RealESRGAN_x4plus', 'RealESRGAN_x2plus')
            model_path: 가중치 경로 (None이면 weights/ 자동 탐색)
            tile_size: 타일 크기 (메모리 부족 시 분할 처리)
                       1024 권장 (RTX 4080 16GB 기준 안전, 분할 최소화)
                       0 = 분할 없이 한 번에 처리 (가장 빠르지만 OOM 위험)
            half_precision: FP16 사용 여부 (GPU 가속)
            always_apply: True면 모든 이미지에 SR 적용 (기본값, 디아스포라 정책)
            max_megapixels: 안전 임계값 - SR 결과가 이보다 크면 스킵
                           (4배 SR 결과 메모리 보호용; 6000x4000 = 24MP 기준)
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")
        
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tile_size = tile_size
        self.half_precision = half_precision and self.device == 'cuda'
        self.always_apply = always_apply
        self.max_megapixels = max_megapixels
        
        model_config = self.SUPPORTED_MODELS.get(
            model_name, self.SUPPORTED_MODELS['RealESRGAN_x4plus']
        )
        self.scale = model_config['scale']
        
        # ──────────────────────────────────────────────────────────
        # 가중치 파일 경로 해석
        # ──────────────────────────────────────────────────────────
        if model_path is None:
            model_path = _resolve_default_weight_path(model_name)
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"가중치 파일을 찾을 수 없습니다: {model_path}\n"
                f"  → 다음 위치에 '{model_name}.pth' 파일을 다운로드하세요:\n"
                f"    {os.path.dirname(model_path)}\n"
                f"  → 다운로드 URL: "
                f"https://github.com/xinntao/Real-ESRGAN/releases"
            )
        
        self.model_path = model_path
        
        # 모델 초기화
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=model_config['num_block'],
            num_grow_ch=32, scale=self.scale
        )
        
        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=self.half_precision,
            device=self.device
        )
        
        tile_desc = "분할 없음" if tile_size == 0 else f"tile={tile_size}"
        print(f"✓ SuperResolutionModule 초기화 완료 "
              f"(모델: {model_name}, 스케일: {self.scale}x, 장치: {self.device})")
        print(f"  가중치 경로: {model_path}")
        print(f"  적용 정책: "
              f"{'모든 이미지에 SR 적용' if always_apply else '임계값 기반 선택적 적용'} "
              f"({tile_desc}, 메모리 안전 한도: {max_megapixels}MP)")
    
    def should_process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        min_resolution: int = 512,
        max_resolution: int = 2048
    ) -> Tuple[bool, str]:
        """
        이미지가 초해상도 처리가 필요한지 판단
        
        정책:
            always_apply=True (기본): 메모리 안전 임계값 이하 모든 이미지 처리
            always_apply=False: 기존 임계값 기반 로직 (legacy / ablation 용)
        
        Args:
            image: 입력 이미지
            min_resolution: (legacy) 최소 해상도 임계값
            max_resolution: (legacy) 최대 해상도 임계값
        """
        # 이미지 크기 추출
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            width, height = img.size
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        # ──────────────────────────────────────────────────────────
        # 메모리 안전 검사 (always_apply=True 여도 너무 큰 이미지는 보호)
        # ──────────────────────────────────────────────────────────
        input_mp = (width * height) / 1_000_000
        output_mp = input_mp * (self.scale ** 2)
        
        if output_mp > self.max_megapixels:
            return False, (
                f"메모리 안전 임계값 초과 "
                f"(입력 {width}x{height} = {input_mp:.1f}MP, "
                f"{self.scale}배 후 {output_mp:.1f}MP > {self.max_megapixels}MP)"
            )
        
        # ──────────────────────────────────────────────────────────
        # 정책 분기
        # ──────────────────────────────────────────────────────────
        if self.always_apply:
            # 디아스포라 정책: 모든 이미지에 SR 적용
            return True, (
                f"표준 SR 적용 ({width}x{height} → "
                f"{width * self.scale}x{height * self.scale})"
            )
        
        # Legacy 임계값 기반 로직 (ablation 용)
        min_dim, max_dim = min(width, height), max(width, height)
        if max_dim > max_resolution:
            return False, f"해상도가 충분히 높음 ({width}x{height})"
        if min_dim < min_resolution:
            return True, f"저해상도 이미지 ({width}x{height})"
        return False, f"해상도가 적절함 ({width}x{height})"
    
    def enhance(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        outscale: Optional[float] = None
    ) -> Dict[str, Any]:
        """이미지 초해상도 복원 수행"""
        # 이미지 로드 (한글 경로 대응)
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image)
            if img is None:
                raise FileNotFoundError(
                    f"이미지를 읽을 수 없습니다: {image}\n"
                    f"  → 파일 존재 여부 및 형식을 확인하세요."
                )
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # 알파 채널이 있는 경우 RGB로 변환 (Real-ESRGAN 호환성)
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
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
