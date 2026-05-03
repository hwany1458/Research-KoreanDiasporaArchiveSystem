"""
colorization.py
흑백 컬러화 모듈 - 다중 백엔드 지원

백엔드 우선순위:
    1) DDColor   - 주 모델 (ICCV 2023, 최신, 권장)
    2) DeOldify  - 보조 모델 (CVPR 2019, baseline 비교용)
    3) Sepia     - 최후 fallback (백엔드 모두 불가 시)

References:
    Kang, X. et al. (2023). DDColor: Towards Photo-Realistic Image
    Colorization via Dual Decoders. ICCV.
    Antic, J. (2019). DeOldify: A Deep Learning based project for
    colorizing and restoring old images.

가중치:
    DDColor   - modelscope: damo/cv_ddcolor_image-colorization (자동 다운로드)
    DeOldify  - https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import warnings


# ──────────────────────────────────────────────────────────────────
# 백엔드 가용성 검사 (지연 임포트)
# ──────────────────────────────────────────────────────────────────
DDCOLOR_AVAILABLE = False
DEOLDIFY_AVAILABLE = False


def _check_ddcolor() -> bool:
    """DDColor (modelscope 경유) 사용 가능 여부 확인"""
    global DDCOLOR_AVAILABLE
    try:
        import modelscope  # noqa: F401
        from modelscope.pipelines import pipeline  # noqa: F401
        from modelscope.utils.constant import Tasks  # noqa: F401
        DDCOLOR_AVAILABLE = True
        return True
    except ImportError:
        DDCOLOR_AVAILABLE = False
        return False


def _check_deoldify() -> bool:
    """DeOldify 설치 확인"""
    global DEOLDIFY_AVAILABLE
    try:
        from deoldify import device  # noqa: F401
        from deoldify.device_id import DeviceId  # noqa: F401
        from deoldify.visualize import get_image_colorizer  # noqa: F401
        DEOLDIFY_AVAILABLE = True
        return True
    except ImportError:
        DEOLDIFY_AVAILABLE = False
        return False


# ──────────────────────────────────────────────────────────────────
# 한글 경로 대응 I/O 헬퍼
# ──────────────────────────────────────────────────────────────────
def _imread_unicode(path: Union[str, Path],
                    flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Windows 한글 경로 대응 이미지 로더."""
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    """Windows 한글 경로 대응 이미지 저장."""
    path = str(path)
    ext = os.path.splitext(path)[1] or '.jpg'
    try:
        success, encoded = cv2.imencode(ext, img)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


class ColorizationModule:
    """
    흑백 컬러화 모듈 (다중 백엔드 지원)
    
    DDColor (주 모델) → DeOldify (보조) → Sepia (최후 fallback)
    순서로 사용 가능한 백엔드를 자동 선택합니다.
    
    Attributes:
        device (str): 연산 장치
        model_type (str): DeOldify용 모델 타입 ('artistic' or 'stable')
        render_factor (int): DeOldify 렌더링 품질 (7-45)
        backend (str): 실제 사용 중인 백엔드 ('ddcolor', 'deoldify', 'sepia')
        
    Example:
        >>> colorizer = ColorizationModule(device='cuda')
        >>> result = colorizer.colorize("bw_photo.jpg")
        >>> print(result['backend'])  # 'ddcolor'
        >>> result['colorized'].save("color_photo.jpg")
    """
    
    # DeOldify 모델 정보 (보조 백엔드용)
    MODELS = {
        'artistic': {
            'url': 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth',
            'filename': 'ColorizeArtistic_gen.pth',
            'description': '생동감 있는 컬러, 세부 묘사 우수'
        },
        'stable': {
            'url': 'https://data.deepai.org/deoldify/ColorizeStable_gen.pth',
            'filename': 'ColorizeStable_gen.pth',
            'description': '안정적인 컬러, 아티팩트 감소'
        }
    }
    
    # 백엔드 우선순위
    BACKEND_PREFERENCE = ('ddcolor', 'deoldify', 'sepia')
    
    def __init__(
        self,
        device: str = 'cuda',
        model_type: str = 'artistic',
        model_path: Optional[str] = None,
        render_factor: int = 35,
        watermark: bool = False,
        backend: Optional[str] = None
    ):
        """
        컬러화 모듈 초기화
        
        Args:
            device: 연산 장치 ('cuda', 'cpu')
            model_type: DeOldify 모델 타입 ('artistic', 'stable')
            model_path: DeOldify 가중치 경로 (None이면 자동 탐색)
            render_factor: DeOldify 렌더링 품질 (7-45)
            watermark: DeOldify 워터마크 표시 여부
            backend: 백엔드 강제 지정 ('ddcolor'/'deoldify'/'sepia'/None=자동)
        """
        self.device = self._setup_device(device)
        self.model_type = model_type
        self.render_factor = render_factor
        self.watermark = watermark
        self.model_path = model_path
        
        # 모델 타입 검증
        if model_type not in self.MODELS:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"지원 타입: {list(self.MODELS.keys())}"
            )
        
        # 백엔드 가용성 확인
        self._ddcolor_ok = _check_ddcolor()
        self._deoldify_ok = _check_deoldify()
        
        # 백엔드 선택
        self.backend = self._select_backend(requested=backend)
        
        # 백엔드별 객체 (지연 초기화)
        self.ddcolor_pipeline = None
        self.deoldify_colorizer = None
        
        # 선택된 백엔드 즉시 로드
        if self.backend == 'ddcolor':
            self.ddcolor_pipeline = self._load_ddcolor()
            if self.ddcolor_pipeline is None:
                # DDColor 로드 실패 시 다음 백엔드로
                print("⚠ DDColor 로드 실패 → DeOldify 시도")
                self.backend = 'deoldify' if self._deoldify_ok else 'sepia'
        
        if self.backend == 'deoldify':
            self.deoldify_colorizer = self._load_deoldify_model()
            if self.deoldify_colorizer is None:
                print("⚠ DeOldify 로드 실패 → Sepia fallback")
                self.backend = 'sepia'
        
        # 호환성 속성 (기존 코드와의 호환성 유지)
        self.use_deoldify = (self.backend == 'deoldify'
                             and self.deoldify_colorizer is not None)
        self.colorizer = (self.deoldify_colorizer
                          if self.backend == 'deoldify' else None)
        
        # 초기화 로그
        print(f"✓ ColorizationModule 초기화 완료")
        print(f"  - 백엔드: {self.backend.upper()}")
        print(f"  - 장치: {self.device}")
        if self.backend == 'ddcolor':
            print(f"  - 주 모델: DDColor (ICCV 2023)")
        elif self.backend == 'deoldify':
            print(f"  - 모델 타입: {model_type}")
            print(f"  - 렌더 팩터: {render_factor}")
        elif self.backend == 'sepia':
            print(f"  ⚠ 주의: Sepia fallback 사용 중 (품질 낮음)")
            print(f"     → DDColor 활성화: pip install modelscope datasets simplejson timm addict sortedcontainers oss2")
    
    # ──────────────────────────────────────────────────────────
    # 초기화 도우미
    # ──────────────────────────────────────────────────────────
    def _setup_device(self, device: str) -> str:
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device
    
    def _select_backend(self, requested: Optional[str]) -> str:
        """백엔드 선택 (사용자 요청 + 가용성 고려)"""
        if requested is not None:
            requested = requested.lower()
            if requested == 'ddcolor' and self._ddcolor_ok:
                return 'ddcolor'
            if requested == 'deoldify' and self._deoldify_ok:
                return 'deoldify'
            if requested == 'sepia':
                return 'sepia'
            print(f"Warning: 요청한 백엔드 '{requested}' 사용 불가 → 자동 선택")
        
        # 자동 선택: 우선순위 순으로
        if self._ddcolor_ok:
            return 'ddcolor'
        if self._deoldify_ok:
            return 'deoldify'
        print("Warning: DDColor/DeOldify 모두 사용 불가 → Sepia fallback 사용")
        return 'sepia'
    
    def _load_ddcolor(self):
        """DDColor 파이프라인 로드 (modelscope 경유)"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            print("DDColor 모델 로딩 중... (최초 실행 시 가중치 자동 다운로드)")
            ddcolor = pipeline(
                Tasks.image_colorization,
                model='damo/cv_ddcolor_image-colorization',
                device=self.device
            )
            return ddcolor
        except Exception as e:
            print(f"Warning: DDColor 로드 실패: {e}")
            return None
    
    def _load_deoldify_model(self):
        """DeOldify 모델 로드"""
        try:
            from deoldify import device as deoldify_device
            from deoldify.device_id import DeviceId
            from deoldify.visualize import get_image_colorizer
            
            if self.device == 'cuda':
                deoldify_device.set(device=DeviceId.GPU0)
            else:
                deoldify_device.set(device=DeviceId.CPU)
            
            if self.model_type == 'artistic':
                colorizer = get_image_colorizer(artistic=True)
            else:
                colorizer = get_image_colorizer(artistic=False)
            return colorizer
        except Exception as e:
            print(f"Warning: DeOldify 모델 로드 실패: {e}")
            return None
    
    # ──────────────────────────────────────────────────────────
    # 흑백 판정
    # ──────────────────────────────────────────────────────────
    def is_grayscale(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1,
        std_threshold: float = 10.0
    ) -> Tuple[bool, Dict[str, float]]:
        """이미지가 흑백인지 판단 (HSV 채도 + RGB 채널 편차)"""
        img = self._load_image(image)
        
        if len(img.shape) == 2:
            return True, {
                'reason': '그레이스케일 이미지',
                'saturation': 0.0, 'rgb_std': 0.0
            }
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        rgb_std = np.std([
            img[:, :, 0].mean(),
            img[:, :, 1].mean(),
            img[:, :, 2].mean()
        ])
        
        analysis = {
            'saturation': float(saturation),
            'rgb_std': float(rgb_std),
            'saturation_threshold': saturation_threshold,
            'std_threshold': std_threshold
        }
        
        is_gray = saturation < saturation_threshold or rgb_std < std_threshold
        if is_gray:
            analysis['reason'] = f'채도={saturation:.3f}, RGB편차={rgb_std:.1f}'
        else:
            analysis['reason'] = f'컬러 이미지 (채도={saturation:.3f})'
        return is_gray, analysis
    
    def should_process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1
    ) -> Tuple[bool, str]:
        """컬러화 처리가 필요한지 판단"""
        is_gray, analysis = self.is_grayscale(image, saturation_threshold)
        if is_gray:
            return True, f"흑백 이미지 - {analysis['reason']}"
        return False, f"이미 컬러 이미지 - {analysis['reason']}"
    
    # ──────────────────────────────────────────────────────────
    # 이미지 로드
    # ──────────────────────────────────────────────────────────
    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """이미지를 numpy array (BGR)로 로드 (한글 경로 대응)"""
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"지원하지 않는 이미지 형식: {type(image)}")
        return img
    
    # ──────────────────────────────────────────────────────────
    # 컬러화 메인 인터페이스
    # ──────────────────────────────────────────────────────────
    def colorize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        render_factor: Optional[int] = None,
        force: bool = False,
        return_original: bool = False
    ) -> Dict[str, Any]:
        """
        이미지 컬러화 수행
        
        Args:
            image: 입력 이미지
            render_factor: DeOldify 전용 렌더링 품질 (다른 백엔드는 무시)
            force: 컬러 이미지도 강제 처리
            return_original: 결과에 원본 포함 여부
            
        Returns:
            {
                'colorized': PIL.Image,         # 컬러화 결과
                'was_grayscale': bool,           # 입력이 흑백이었는지
                'render_factor': int,            # 사용된 렌더 팩터
                'backend': str,                  # 사용된 백엔드명
                'skipped': bool,                 # 처리 건너뜀 여부
                'reason': str (선택),            # 건너뛴 사유
                'original': PIL.Image (선택)     # 원본 이미지
            }
        """
        if render_factor is None:
            render_factor = self.render_factor
        
        img_bgr = self._load_image(image)
        original_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        is_gray, analysis = self.is_grayscale(image)
        
        # 이미 컬러인 경우 (force가 아니면 건너뜀)
        if not is_gray and not force:
            result = {
                'colorized': original_pil,
                'was_grayscale': False,
                'render_factor': render_factor,
                'backend': self.backend,
                'skipped': True,
                'reason': analysis['reason']
            }
            if return_original:
                result['original'] = original_pil
            return result
        
        # 백엔드별 컬러화 분기
        if self.backend == 'ddcolor' and self.ddcolor_pipeline is not None:
            colorized_pil = self._colorize_with_ddcolor(img_bgr)
        elif self.backend == 'deoldify' and self.deoldify_colorizer is not None:
            colorized_pil = self._colorize_with_deoldify(image, render_factor)
        else:
            colorized_pil = self._colorize_fallback(img_bgr)
        
        result = {
            'colorized': colorized_pil,
            'was_grayscale': is_gray,
            'render_factor': render_factor,
            'backend': self.backend,
            'skipped': False
        }
        if return_original:
            result['original'] = original_pil
        return result
    
    # ──────────────────────────────────────────────────────────
    # 백엔드별 구현
    # ──────────────────────────────────────────────────────────
    def _colorize_with_ddcolor(self, img_bgr: np.ndarray) -> Image.Image:
        """DDColor를 사용한 컬러화"""
        from modelscope.outputs import OutputKeys
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = self.ddcolor_pipeline(img_bgr)
        
        output_bgr = output[OutputKeys.OUTPUT_IMG]
        return Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
    
    def _colorize_with_deoldify(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        render_factor: int
    ) -> Image.Image:
        """DeOldify를 사용한 컬러화 (파일 경로 기반)"""
        temp_dir = Path("./cache/temp_colorize")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, (str, Path)):
            source_path = Path(image)
        else:
            temp_path = temp_dir / "temp_input.jpg"
            if isinstance(image, Image.Image):
                image.save(temp_path)
            elif isinstance(image, np.ndarray):
                _imwrite_unicode(temp_path, image)
            source_path = temp_path
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_image = self.deoldify_colorizer.get_transformed_image(
                str(source_path),
                render_factor=render_factor,
                watermarked=self.watermark
            )
        return result_image
    
    def _colorize_fallback(self, img_bgr: np.ndarray) -> Image.Image:
        """
        최후 fallback: 세피아 톤 (DDColor/DeOldify 모두 사용 불가 시)
        품질이 매우 낮으므로 학위논문 평가에는 부적합. 시연용으로만 사용.
        """
        print("Warning: Sepia fallback 사용 중 (학위논문 평가에는 부적합)")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sepia_img = img_bgr.copy().astype(np.float32)
        sepia_img[:, :, 0] = gray * 0.393 + gray * 0.769 * 0.1  # B
        sepia_img[:, :, 1] = gray * 0.349 + gray * 0.686 * 0.3  # G
        sepia_img[:, :, 2] = gray * 0.272 + gray * 0.534 * 0.5  # R
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB))
    
    # ──────────────────────────────────────────────────────────
    # 색상 후처리 / 결과 저장
    # ──────────────────────────────────────────────────────────
    def adjust_colors(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_factor: float = 1.0,
        brightness_factor: float = 1.0
    ) -> Image.Image:
        """컬러화된 이미지의 채도/밝기 조정"""
        from PIL import ImageEnhance
        
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        if saturation_factor != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation_factor)
        if brightness_factor != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        return img
    
    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path],
        quality: int = 95,
        save_comparison: bool = False
    ) -> None:
        """결과 저장 (한글 경로 대응)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        colorized = result['colorized']
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            colorized.save(str(output_path), quality=quality)
        else:
            colorized.save(str(output_path))
        
        # 전후 비교 이미지
        if save_comparison and 'original' in result:
            original = result['original']
            if original.size != colorized.size:
                original = original.resize(
                    colorized.size, Image.Resampling.LANCZOS
                )
            comparison = Image.new(
                'RGB', (colorized.width * 2, colorized.height)
            )
            comparison.paste(original, (0, 0))
            comparison.paste(colorized, (colorized.width, 0))
            comparison_path = (
                output_path.parent /
                f"{output_path.stem}_comparison{output_path.suffix}"
            )
            comparison.save(str(comparison_path), quality=quality)


# ============================================
# 보조 클래스 (호환성 유지)
# ============================================
class SimpleColorizer:
    """
    DeOldify 없이 사용할 수 있는 간단한 컬러화 클래스 (placeholder).
    실제 사용 시 ColorizationModule(backend='ddcolor')를 권장합니다.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def colorize(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        return img


# ============================================
# 테스트 코드
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Colorization Module Test")
    parser.add_argument("--input", "-i", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output", "-o", type=str, default="output_color.jpg", help="출력 이미지 경로")
    parser.add_argument("--backend", "-b", type=str, default=None,
                        choices=['ddcolor', 'deoldify', 'sepia', None],
                        help="백엔드 강제 지정 (기본: 자동 선택)")
    parser.add_argument("--model", "-m", type=str, default="artistic",
                        choices=['artistic', 'stable'],
                        help="DeOldify 모델 타입")
    parser.add_argument("--render-factor", "-r", type=int, default=35,
                        help="DeOldify 렌더 팩터 (7-45)")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="장치")
    parser.add_argument("--force", "-f", action="store_true",
                        help="컬러 이미지도 강제 처리")
    
    args = parser.parse_args()
    
    colorizer = ColorizationModule(
        device=args.device,
        model_type=args.model,
        render_factor=args.render_factor,
        backend=args.backend
    )
    
    should_process, reason = colorizer.should_process(args.input)
    print(f"처리 필요: {should_process} - {reason}")
    
    print(f"컬러화 중: {args.input}")
    result = colorizer.colorize(args.input, force=args.force)
    
    print(f"흑백이었는지: {result['was_grayscale']}")
    print(f"사용 백엔드: {result['backend']}")
    if result.get('skipped'):
        print(f"건너뜀: {result.get('reason')}")
    
    colorizer.save_result(result, args.output, save_comparison=True)
    print(f"결과 저장 완료: {args.output}")
