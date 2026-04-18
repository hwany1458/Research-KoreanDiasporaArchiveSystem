"""
colorization.py
흑백 컬러화 모듈 - DDColor 기반 (HuggingFace 직접 추론)

DDColor(Kang et al., 2023)를 HuggingFace에서 직접 로드하여
흑백 사진을 자연스러운 컬러로 변환합니다.
basicsr 버전 충돌 없이 독립적으로 동작합니다.

우선순위:
  1. DeOldify (설치된 경우)
  2. DDColor via HuggingFace (딥러닝, 고품질)
  3. LAB 색공간 폴백 (딥러닝 없을 때)

Reference:
    Kang, X., et al. (2023). DDColor: Towards Photo-Realistic and
    Semantic-Aware Image Colorization via Dual Decoders. ICCV.
    HuggingFace: piddnad/ddcolor_modelscope
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import warnings

# ── DeOldify 확인 ──────────────────────────────────────────
DEOLDIFY_AVAILABLE = False

def _check_deoldify():
    global DEOLDIFY_AVAILABLE
    try:
        from deoldify import device
        from deoldify.device_id import DeviceId
        from deoldify.visualize import get_image_colorizer
        DEOLDIFY_AVAILABLE = True
        return True
    except ImportError:
        return False

# ── DDColor HuggingFace 모델 확인 ──────────────────────────
DDCOLOR_AVAILABLE = False

def _check_ddcolor():
    global DDCOLOR_AVAILABLE
    try:
        from huggingface_hub import hf_hub_download
        DDCOLOR_AVAILABLE = True
        return True
    except ImportError:
        return False


def _load_ddcolor_model(device: str):
    """
    DDColor 모델을 HuggingFace from_pretrained 방식으로 로드합니다.
    공식 방법: PyTorchModelHubMixin 활용
    """
    try:
        import sys
        # ddcolor_src 경로 추가
        ddcolor_src = os.path.join(os.getcwd(), 'ddcolor_src')
        if os.path.exists(ddcolor_src) and ddcolor_src not in sys.path:
            sys.path.insert(0, ddcolor_src)

        from ddcolor.model import DDColor
        from huggingface_hub import PyTorchModelHubMixin

        # HuggingFace 공식 방법: DDColorHF 클래스 동적 생성
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        print("  [Color] DDColor 모델 다운로드 중... (첫 실행 시 약 900MB)")
        model = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope")
        model.eval()
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        print("  [Color] DDColor 모델 로드 완료 ✓")
        return model

    except Exception as e:
        print(f"  [Color] DDColor 모델 로드 실패: {e}")
        return None


def _colorize_with_ddcolor(model, img_pil: Image.Image, device: str) -> Image.Image:
    """
    DDColor 공식 ColorizationPipeline.process() 방식으로 컬러화.
    입력: PIL RGB, 출력: PIL RGB
    """
    import torch.nn.functional as F

    # PIL → BGR numpy (공식 코드는 BGR uint8 입력)
    img_rgb_np = np.array(img_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)

    input_size = 512
    height, width = img_bgr.shape[:2]

    # 0~1 float 변환
    img = (img_bgr / 255.0).astype(np.float32)

    # 원본 L 채널 보존
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

    # 512x512로 리사이즈 후 L채널만 추출, ab=0으로 만들어 회색 RGB 생성
    img_resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate(
        (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
    )
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    # 텐서 변환: [1, 3, 512, 512]
    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
        .to(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    )

    # 모델 추론
    with torch.no_grad():
        output_ab = model(tensor_gray_rgb).cpu()  # (1, 2, 512, 512)

    # ab 채널을 원본 해상도로 리사이즈
    output_ab_resized = (
        F.interpolate(output_ab, size=(height, width))[0]
        .float()
        .numpy()
        .transpose(1, 2, 0)
    )  # (h, w, 2)

    # 원본 L + 예측 ab → BGR → RGB
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_img = (output_bgr * 255.0).round().astype(np.uint8)
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(output_rgb)


class ColorizationModule:
    """
    흑백 컬러화 모듈

    우선순위:
      1. DeOldify (설치된 경우, 최고 품질)
      2. DDColor HuggingFace (딥러닝, 고품질, 기본값)
      3. LAB 색공간 폴백 (딥러닝 없을 때)
    """

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
        self.device = self._setup_device(device)
        self.model_type = model_type
        self.render_factor = render_factor
        self.watermark = watermark
        self.model_path = model_path
        self._ddcolor_model = None

        if model_type not in self.MODELS:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        # 컬러화 엔진 결정
        self.use_deoldify = _check_deoldify()
        self.use_ddcolor = False

        if self.use_deoldify:
            self.colorizer = self._load_deoldify_model()
            self._engine = 'deoldify'
            print("  - 컬러화 엔진: DeOldify")
        else:
            self.colorizer = None
            if _check_ddcolor():
                self._ddcolor_model = _load_ddcolor_model(self.device)
                if self._ddcolor_model is not None:
                    self.use_ddcolor = True
                    self._engine = 'ddcolor'
                    print("  - 컬러화 엔진: DDColor (HuggingFace 딥러닝)")
                else:
                    self._engine = 'fallback'
                    print("  - 컬러화 엔진: LAB 폴백 (DDColor 로드 실패)")
            else:
                self._engine = 'fallback'
                print("  - 컬러화 엔진: LAB 폴백")

        print(f"✓ ColorizationModule 초기화 완료")
        print(f"  - 모델 타입: {model_type}")
        print(f"  - 렌더 팩터: {render_factor}")
        print(f"  - 장치: {self.device}")
        print(f"  - 엔진: {self._engine}")

    def _setup_device(self, device: str) -> str:
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device

    def _load_deoldify_model(self):
        try:
            from deoldify import device as deoldify_device
            from deoldify.device_id import DeviceId
            from deoldify.visualize import get_image_colorizer
            if self.device == 'cuda':
                deoldify_device.set(device=DeviceId.GPU0)
            else:
                deoldify_device.set(device=DeviceId.CPU)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                colorizer = get_image_colorizer(
                    artistic=(self.model_type == 'artistic'),
                    watermarked=self.watermark
                )
            return colorizer
        except Exception as e:
            print(f"Warning: DeOldify 모델 로드 실패: {e}")
            return None

    def is_grayscale(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1,
        std_threshold: float = 10.0
    ) -> Tuple[bool, Dict]:
        if isinstance(image, (str, Path)):
            img_np = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert('RGB'))
        else:
            img_np = image

        if len(img_np.shape) == 2 or (len(img_np.shape) == 3 and img_np.shape[2] == 1):
            return True, {'reason': '그레이스케일 이미지', 'saturation': 0.0, 'rgb_std': 0.0}

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        rgb_std = np.std(img_np.astype(float).reshape(-1, 3), axis=0).mean()

        analysis = {
            'saturation': float(saturation),
            'rgb_std': float(rgb_std),
            'saturation_threshold': saturation_threshold,
        }
        is_gray = saturation < saturation_threshold or rgb_std < std_threshold
        analysis['reason'] = (
            f'채도={saturation:.3f}, RGB편차={rgb_std:.1f}'
            if is_gray else f'컬러 이미지 (채도={saturation:.3f})'
        )
        return is_gray, analysis

    def should_process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_threshold: float = 0.1
    ) -> Tuple[bool, str]:
        is_gray, analysis = self.is_grayscale(image, saturation_threshold)
        if is_gray:
            return True, f"흑백 이미지 감지 ({analysis['reason']})"
        return False, f"컬러 이미지 ({analysis['reason']})"

    def colorize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        render_factor: Optional[int] = None,
        outscale: float = 1.0
    ) -> Dict[str, Any]:
        render_factor = render_factor or self.render_factor

        if isinstance(image, (str, Path)):
            img_pil = Image.open(str(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            img_pil = image.convert('RGB')
        else:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        original_pil = img_pil.copy()

        try:
            if self.use_deoldify and self.colorizer:
                colorized_pil = self._colorize_deoldify(img_pil, render_factor)
                method = 'deoldify'
            elif self.use_ddcolor and self._ddcolor_model is not None:
                colorized_pil = _colorize_with_ddcolor(
                    self._ddcolor_model, img_pil, self.device
                )
                method = 'ddcolor'
            else:
                colorized_pil = self._colorize_fallback(img_bgr)
                method = 'fallback'
        except Exception as e:
            print(f"  Warning: 컬러화 실패 ({e}), 폴백 사용")
            colorized_pil = self._colorize_fallback(img_bgr)
            method = 'fallback'

        return {
            'colorized': colorized_pil,
            'original': original_pil,
            'method': method,
            'render_factor': render_factor,
        }

    def _colorize_deoldify(self, img_pil: Image.Image, render_factor: int) -> Image.Image:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        try:
            img_pil.save(tmp_path)
            result = self.colorizer.get_transformed_image(
                tmp_path, render_factor=render_factor, watermarked=self.watermark
            )
            return result
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _colorize_fallback(self, img_bgr: np.ndarray) -> Image.Image:
        """LAB 색공간 기반 폴백 컬러화"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray_norm = gray.astype(np.float32) / 255.0

        lab = np.zeros((h, w, 3), dtype=np.float32)
        lab[:, :, 0] = gray.astype(np.float32)
        lab[:, :, 1] = (gray_norm * 5.0 - 2.0) + 128
        lab[:, :, 2] = (gray_norm * 8.0 - 3.0) + 128
        lab = np.clip(lab, 0, 255).astype(np.uint8)

        result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result_bgr = cv2.addWeighted(result_bgr, 0.6, img_bgr, 0.4, 0)
        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

    def adjust_colors(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        saturation_factor: float = 1.0,
        brightness_factor: float = 1.0
    ) -> Image.Image:
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
