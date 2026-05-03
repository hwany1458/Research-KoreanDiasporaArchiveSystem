"""
colorization.py
흑백 컬러화 모듈 - DeOldify 기반

DeOldify를 활용하여 흑백 사진을 컬러로 변환합니다.
NoGAN 학습 방식을 통해 안정적인 컬러 추론을 수행합니다.

Reference:
    Antic, J. (2019). DeOldify: A Deep Learning based project for 
    colorizing and restoring old images (and video!)
    
GitHub: https://github.com/jantic/DeOldify
모델 다운로드: https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import warnings

# DeOldify 관련 imports - 동적 로딩
DEOLDIFY_AVAILABLE = False

def _check_deoldify():
    """DeOldify 설치 확인 및 로드"""
    global DEOLDIFY_AVAILABLE
    try:
        from deoldify import device
        from deoldify.device_id import DeviceId
        from deoldify.visualize import get_image_colorizer
        DEOLDIFY_AVAILABLE = True
        return True
    except ImportError:
        return False


class ColorizationModule:
    """
    흑백 컬러화 모듈
    
    DeOldify의 Self-Attention GAN과 NoGAN 학습 방식을 활용하여
    흑백 이미지를 자연스럽게 컬러화합니다.
    
    두 가지 모드 지원:
    - Artistic: 생동감 있는 컬러, 세부 묘사 우수
    - Stable: 안정적인 컬러, 아티팩트 감소
    
    Attributes:
        device (str): 연산 장치
        model_type (str): 모델 타입 ('artistic' or 'stable')
        render_factor (int): 렌더링 품질 (7-45)
        colorizer: DeOldify colorizer 인스턴스
        
    Example:
        >>> colorizer = ColorizationModule(device='cuda')
        >>> result = colorizer.colorize("bw_photo.jpg")
        >>> result['colorized'].save("color_photo.jpg")
    """
    
    # 모델 정보
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
    
    def __init__(
        self,
        device: str = 'cuda',
        model_type: str = 'artistic',
        model_path: Optional[str] = None,
        render_factor: int = 35,
        watermark: bool = False
    ):
        """
        컬러화 모듈 초기화
        
        Args:
            device: 연산 장치 ('cuda', 'cpu')
            model_type: 모델 타입 ('artistic', 'stable')
            model_path: 모델 가중치 경로 (None이면 자동 다운로드)
            render_factor: 렌더링 품질 (7-45, 높을수록 고품질)
            watermark: 워터마크 표시 여부
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
        
        # DeOldify 사용 가능 여부 확인
        self.use_deoldify = _check_deoldify()
        
        if self.use_deoldify:
            self.colorizer = self._load_deoldify_model()
        else:
            print("Warning: DeOldify를 사용할 수 없습니다. 대체 방법을 사용합니다.")
            self.colorizer = None
        
        print(f"✓ ColorizationModule 초기화 완료")
        print(f"  - 모델 타입: {model_type}")
        print(f"  - 렌더 팩터: {render_factor}")
        print(f"  - 장치: {self.device}")
        print(f"  - DeOldify 사용: {self.use_deoldify}")
    
    def _setup_device(self, device: str) -> str:
        """장치 설정 및 검증"""
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device
    
    def _load_deoldify_model(self):
        """DeOldify 모델 로드"""
        try:
            from deoldify import device as deoldify_device
            from deoldify.device_id import DeviceId
            from deoldify.visualize import get_image_colorizer
            
            # 장치 설정
            if self.device == 'cuda':
                deoldify_device.set(device=DeviceId.GPU0)
            else:
                deoldify_device.set(device=DeviceId.CPU)
            
            # Colorizer 생성
            if self.model_type == 'artistic':
                colorizer = get_image_colorizer(artistic=True)
            else:
                colorizer = get_image_colorizer(artistic=False)
            
            return colorizer
            
        except Exception as e:
            print(f"Warning: DeOldify 모델 로드 실패: {e}")
            return None
    
    def is_grayscale(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1,
        std_threshold: float = 10.0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        이미지가 흑백인지 판단
        
        두 가지 기준 사용:
        1. HSV 색공간에서 채도(Saturation) 평균이 임계값 미만
        2. RGB 채널 간 표준편차의 평균이 임계값 미만
        
        Args:
            image: 입력 이미지
            saturation_threshold: 채도 임계값 (0-1)
            std_threshold: RGB 표준편차 임계값
            
        Returns:
            (흑백 여부, 분석 정보)
        """
        # 이미지 로드
        img = self._load_image(image)
        
        # RGB로 변환 (BGR인 경우)
        if len(img.shape) == 2:
            # 이미 그레이스케일
            return True, {'reason': '그레이스케일 이미지', 'saturation': 0.0, 'rgb_std': 0.0}
        
        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        
        # RGB 채널 간 표준편차
        rgb_std = np.std([img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()])
        
        analysis = {
            'saturation': float(saturation),
            'rgb_std': float(rgb_std),
            'saturation_threshold': saturation_threshold,
            'std_threshold': std_threshold
        }
        
        is_grayscale = saturation < saturation_threshold or rgb_std < std_threshold
        
        if is_grayscale:
            analysis['reason'] = f'채도={saturation:.3f}, RGB편차={rgb_std:.1f}'
        else:
            analysis['reason'] = f'컬러 이미지 (채도={saturation:.3f})'
        
        return is_grayscale, analysis
    
    def should_process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1
    ) -> Tuple[bool, str]:
        """
        컬러화 처리가 필요한지 판단
        
        Args:
            image: 입력 이미지
            saturation_threshold: 채도 임계값
            
        Returns:
            (처리 필요 여부, 사유)
        """
        is_gray, analysis = self.is_grayscale(image, saturation_threshold)
        
        if is_gray:
            return True, f"흑백 이미지 - {analysis['reason']}"
        else:
            return False, f"이미 컬러 이미지 - {analysis['reason']}"
    
    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """이미지를 numpy array (BGR)로 로드"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
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
    
    def colorize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        render_factor: Optional[int] = None,
        return_original: bool = True,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        흑백 이미지 컬러화 수행
        
        Args:
            image: 입력 이미지
            render_factor: 렌더링 품질 (None이면 기본값 사용)
            return_original: 원본 이미지도 반환할지
            force: 이미 컬러인 이미지도 강제 처리할지
            
        Returns:
            dict: {
                'colorized': PIL.Image - 컬러화된 이미지,
                'original': PIL.Image - 원본 이미지 (옵션),
                'was_grayscale': bool - 흑백이었는지,
                'render_factor': int - 사용된 렌더 팩터
            }
        """
        render_factor = render_factor or self.render_factor
        
        # 이미지 로드
        img_bgr = self._load_image(image)
        original_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        # 흑백 여부 확인
        is_gray, analysis = self.is_grayscale(image)
        
        if not is_gray and not force:
            # 이미 컬러인 경우 원본 반환
            result = {
                'colorized': original_pil,
                'was_grayscale': False,
                'render_factor': render_factor,
                'skipped': True,
                'reason': analysis['reason']
            }
            if return_original:
                result['original'] = original_pil
            return result
        
        # 컬러화 수행
        if self.use_deoldify and self.colorizer is not None:
            colorized_pil = self._colorize_with_deoldify(image, render_factor)
        else:
            colorized_pil = self._colorize_fallback(img_bgr)
        
        result = {
            'colorized': colorized_pil,
            'was_grayscale': is_gray,
            'render_factor': render_factor,
            'skipped': False
        }
        
        if return_original:
            result['original'] = original_pil
        
        return result
    
    def _colorize_with_deoldify(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        render_factor: int
    ) -> Image.Image:
        """DeOldify를 사용한 컬러화"""
        
        # 임시 파일로 저장 (DeOldify는 파일 경로 필요)
        temp_dir = Path("./cache/temp_colorize")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, (str, Path)):
            source_path = Path(image)
        else:
            # 이미지를 임시 파일로 저장
            temp_path = temp_dir / "temp_input.jpg"
            if isinstance(image, Image.Image):
                image.save(temp_path)
            elif isinstance(image, np.ndarray):
                cv2.imwrite(str(temp_path), image)
            source_path = temp_path
        
        # DeOldify 컬러화
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_image = self.colorizer.get_transformed_image(
                str(source_path),
                render_factor=render_factor,
                watermarked=self.watermark
            )
        
        return result_image
    
    def _colorize_fallback(self, img_bgr: np.ndarray) -> Image.Image:
        """
        DeOldify 없이 대체 컬러화 (간단한 방법)
        
        참고: 이 방법은 품질이 낮으므로 DeOldify 설치를 권장합니다.
        LAB 색공간을 활용한 기본적인 컬러 힌트 적용
        """
        print("Warning: 대체 컬러화 방법 사용 (품질이 낮을 수 있음)")
        
        # 그레이스케일로 변환
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # LAB 색공간에서 L 채널만 사용하고 a, b는 중립값으로
        # (실제로는 약간의 세피아 톤 적용)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 세피아 톤 적용 (간단한 예시)
        # 실제 딥러닝 컬러화가 아님
        sepia_img = img_bgr.copy().astype(np.float32)
        sepia_img[:, :, 0] = gray * 0.393 + gray * 0.769 * 0.1  # B
        sepia_img[:, :, 1] = gray * 0.349 + gray * 0.686 * 0.3  # G
        sepia_img[:, :, 2] = gray * 0.272 + gray * 0.534 * 0.5  # R
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        # PIL Image로 변환
        result = Image.fromarray(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB))
        
        return result
    
    def adjust_colors(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_factor: float = 1.0,
        brightness_factor: float = 1.0
    ) -> Image.Image:
        """
        컬러화된 이미지의 색상 조정
        
        Args:
            image: 입력 이미지
            saturation_factor: 채도 조정 (1.0 = 원본, >1.0 = 증가)
            brightness_factor: 밝기 조정 (1.0 = 원본, >1.0 = 증가)
            
        Returns:
            조정된 이미지
        """
        from PIL import ImageEnhance
        
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        # 채도 조정
        if saturation_factor != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_factor)
        
        # 밝기 조정
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        return img
    
    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path],
        quality: int = 95,
        save_comparison: bool = False
    ) -> None:
        """
        결과 저장
        
        Args:
            result: colorize() 반환 결과
            output_path: 출력 파일 경로
            quality: JPEG 품질
            save_comparison: 전후 비교 이미지도 저장할지
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 컬러화된 이미지 저장
        colorized = result['colorized']
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            colorized.save(output_path, quality=quality)
        else:
            colorized.save(output_path)
        
        # 비교 이미지 저장
        if save_comparison and 'original' in result:
            original = result['original']
            
            # 크기 맞추기
            if original.size != colorized.size:
                original = original.resize(colorized.size, Image.Resampling.LANCZOS)
            
            # 나란히 배치
            comparison = Image.new('RGB', (colorized.width * 2, colorized.height))
            comparison.paste(original, (0, 0))
            comparison.paste(colorized, (colorized.width, 0))
            
            comparison_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
            comparison.save(comparison_path, quality=quality)


class SimpleColorizer:
    """
    DeOldify 없이 사용할 수 있는 간단한 컬러화 클래스
    
    Transformers 라이브러리의 사전학습 모델을 사용합니다.
    품질은 DeOldify보다 낮을 수 있지만 설치가 간단합니다.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """모델 지연 로딩"""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForImageToImage, AutoProcessor
            
            # 대안 모델 (설치 용이)
            # 참고: 실제로는 컬러화 전용 모델을 사용해야 함
            print("Warning: 간단한 대체 컬러화 모델 로드 중...")
            
        except ImportError:
            print("Warning: transformers 라이브러리가 없습니다.")
    
    def colorize(self, image_path: str) -> Image.Image:
        """간단한 컬러화 (placeholder)"""
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
    parser.add_argument("--model", "-m", type=str, default="artistic", 
                        choices=['artistic', 'stable'], help="모델 타입")
    parser.add_argument("--render-factor", "-r", type=int, default=35, help="렌더 팩터 (7-45)")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="장치")
    parser.add_argument("--force", "-f", action="store_true", help="컬러 이미지도 강제 처리")
    
    args = parser.parse_args()
    
    # 모듈 초기화
    colorizer = ColorizationModule(
        device=args.device,
        model_type=args.model,
        render_factor=args.render_factor
    )
    
    # 흑백 여부 확인
    should_process, reason = colorizer.should_process(args.input)
    print(f"처리 필요: {should_process} - {reason}")
    
    # 컬러화 수행
    print(f"컬러화 중: {args.input}")
    result = colorizer.colorize(args.input, force=args.force)
    
    print(f"흑백이었는지: {result['was_grayscale']}")
    if result.get('skipped'):
        print(f"건너뜀: {result.get('reason')}")
    
    # 결과 저장
    colorizer.save_result(result, args.output, save_comparison=True)
    print(f"결과 저장 완료: {args.output}")
