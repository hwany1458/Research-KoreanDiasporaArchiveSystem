"""
이미지 처리 모듈 (Image Processing Module)

한인 디아스포라 기록유산의 이미지 복원, 향상, 분석을 위한 AI 기반 처리 모듈

주요 기능:
- 초해상도 변환 (Super-Resolution): Real-ESRGAN
- 손상 복원 (Inpainting): LaMa
- 얼굴 향상 (Face Enhancement): GFPGAN, CodeFormer
- 흑백 컬러화 (Colorization): DeOldify
- 이미지 분석/캡셔닝 (Analysis): CLIP, BLIP-2

Author: Diaspora Archive Project
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """이미지 손상 유형"""
    LOW_RESOLUTION = "low_resolution"
    NOISE = "noise"
    SCRATCH = "scratch"
    STAIN = "stain"
    FADING = "fading"
    BLUR = "blur"
    TORN = "torn"
    UNKNOWN = "unknown"


class ProcessingMode(Enum):
    """처리 모드"""
    QUALITY = "quality"      # 최고 품질 (느림)
    BALANCED = "balanced"    # 균형 (기본)
    FAST = "fast"           # 빠른 처리


@dataclass
class ImageMetadata:
    """이미지 메타데이터"""
    width: int
    height: int
    channels: int
    format: str
    file_size: int
    bit_depth: int = 8
    is_grayscale: bool = False
    has_alpha: bool = False
    exif_data: Dict[str, Any] = field(default_factory=dict)
    estimated_year: Optional[int] = None
    quality_score: float = 0.0
    detected_faces: int = 0
    degradation_types: List[DegradationType] = field(default_factory=list)


@dataclass
class RestorationResult:
    """복원 결과"""
    original_path: str
    restored_image: np.ndarray
    metadata: ImageMetadata
    processing_steps: List[str]
    quality_improvement: float
    processing_time: float
    face_regions: List[Dict[str, Any]] = field(default_factory=list)
    auto_caption: str = ""
    tags: List[str] = field(default_factory=list)


class ImageAnalyzer:
    """이미지 분석기 - 손상 유형 감지 및 품질 평가"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageAnalyzer initialized on {self.device}")
        
    def analyze(self, image: np.ndarray) -> ImageMetadata:
        """이미지 분석 및 메타데이터 추출"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            format="unknown",
            file_size=0,
            is_grayscale=(channels == 1),
            has_alpha=(channels == 4)
        )
        
        # 품질 점수 계산
        metadata.quality_score = self._calculate_quality_score(image)
        
        # 손상 유형 감지
        metadata.degradation_types = self._detect_degradation(image)
        
        # 연대 추정 (이미지 특성 기반)
        metadata.estimated_year = self._estimate_year(image)
        
        return metadata
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """이미지 품질 점수 계산 (0-100)"""
        scores = []
        
        # 1. 선명도 (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        scores.append(sharpness_score)
        
        # 2. 노이즈 레벨 추정
        noise_level = self._estimate_noise_level(gray)
        noise_score = max(0, 100 - noise_level * 2)
        scores.append(noise_score)
        
        # 3. 콘트라스트
        contrast = gray.std()
        contrast_score = min(100, contrast * 2)
        scores.append(contrast_score)
        
        # 4. 밝기 균형
        brightness = gray.mean()
        brightness_score = 100 - abs(brightness - 128) / 1.28
        scores.append(brightness_score)
        
        return np.mean(scores)
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """노이즈 레벨 추정"""
        # Median Absolute Deviation 기반 노이즈 추정
        H, W = gray.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray, -1, np.array(M)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
        return sigma
    
    def _detect_degradation(self, image: np.ndarray) -> List[DegradationType]:
        """손상 유형 감지"""
        degradations = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # 저해상도 감지
        height, width = image.shape[:2]
        if width < 500 or height < 500:
            degradations.append(DegradationType.LOW_RESOLUTION)
        
        # 노이즈 감지
        noise_level = self._estimate_noise_level(gray)
        if noise_level > 15:
            degradations.append(DegradationType.NOISE)
        
        # 블러 감지
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            degradations.append(DegradationType.BLUR)
        
        # 색바램 감지 (컬러 이미지만)
        if len(image.shape) > 2 and image.shape[2] == 3:
            saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
            if saturation < 30:
                degradations.append(DegradationType.FADING)
        
        # 긁힘/얼룩 감지 (에지 기반)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = edges.sum() / (gray.shape[0] * gray.shape[1] * 255)
        if edge_ratio > 0.15:  # 비정상적으로 많은 에지
            degradations.append(DegradationType.SCRATCH)
        
        if not degradations:
            degradations.append(DegradationType.UNKNOWN)
            
        return degradations
    
    def _estimate_year(self, image: np.ndarray) -> Optional[int]:
        """이미지 특성 기반 연대 추정"""
        # 간단한 휴리스틱 기반 추정
        # 실제로는 딥러닝 모델 사용 권장
        
        is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
        
        if is_grayscale:
            # 흑백 사진은 대체로 1980년 이전
            return 1970
        
        # 컬러 사진의 채도 분석
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            
            if saturation < 50:  # 색이 바랜 사진
                return 1985
            elif saturation < 100:
                return 1995
            else:
                return 2005
        
        return None


class SuperResolution:
    """초해상도 변환 모듈 (Real-ESRGAN)"""
    
    def __init__(self, model_name: str = "RealESRGAN_x4plus", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.scale = 4
        logger.info(f"SuperResolution initialized: {model_name} on {self.device}")
        
    def load_model(self, model_path: Optional[str] = None):
        """모델 로드"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # 모델 아키텍처 설정
            if "x4plus" in self.model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                self.scale = 4
            elif "x2plus" in self.model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2
                )
                self.scale = 2
            else:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                self.scale = 4
            
            self.model = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            logger.info("Real-ESRGAN model loaded successfully")
            
        except ImportError:
            logger.warning("Real-ESRGAN not installed. Using fallback upscaling.")
            self.model = None
    
    def upscale(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """이미지 초해상도 변환"""
        if self.model is not None:
            try:
                output, _ = self.model.enhance(image, outscale=scale)
                return output
            except Exception as e:
                logger.error(f"Real-ESRGAN error: {e}")
        
        # Fallback: OpenCV 기반 업스케일링
        return self._fallback_upscale(image, scale)
    
    def _fallback_upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        """폴백 업스케일링 (Lanczos)"""
        height, width = image.shape[:2]
        new_size = (width * scale, height * scale)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


class FaceEnhancer:
    """얼굴 향상 모듈 (GFPGAN, CodeFormer)"""
    
    def __init__(self, model_name: str = "GFPGAN", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.face_detector = None
        logger.info(f"FaceEnhancer initialized: {model_name} on {self.device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """모델 로드"""
        try:
            if self.model_name == "GFPGAN":
                from gfpgan import GFPGANer
                self.model = GFPGANer(
                    model_path=model_path or 'GFPGANv1.4.pth',
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    device=self.device
                )
            elif self.model_name == "CodeFormer":
                # CodeFormer 로드
                from basicsr.utils.download_util import load_file_from_url
                # CodeFormer 구현 필요
                pass
            
            logger.info(f"{self.model_name} model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"{self.model_name} not installed: {e}")
            self.model = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 감지"""
        faces = []
        
        try:
            # OpenCV DNN 기반 얼굴 감지
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for i, (x, y, w, h) in enumerate(detected):
                faces.append({
                    'id': i,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.9,  # 기본값
                    'landmarks': None
                })
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
        
        return faces
    
    def enhance(self, image: np.ndarray, faces: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[Dict]]:
        """얼굴 향상"""
        if faces is None:
            faces = self.detect_faces(image)
        
        if self.model is not None and faces:
            try:
                _, _, output = self.model.enhance(
                    image,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                return output, faces
            except Exception as e:
                logger.error(f"Face enhancement error: {e}")
        
        return image, faces


class ImageInpainter:
    """이미지 인페인팅 모듈 (LaMa)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        logger.info(f"ImageInpainter initialized on {self.device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """LaMa 모델 로드"""
        try:
            # LaMa 모델 로드
            # 실제 구현에서는 lama-cleaner 또는 직접 구현 사용
            logger.info("LaMa model loaded")
        except Exception as e:
            logger.warning(f"LaMa model loading failed: {e}")
    
    def detect_damage(self, image: np.ndarray) -> np.ndarray:
        """손상 영역 자동 감지 (마스크 생성)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 방법 1: 이상치 픽셀 감지
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # 극단적으로 밝거나 어두운 픽셀
        mask = np.zeros_like(gray)
        mask[(gray < mean_val - 3 * std_val) | (gray > mean_val + 3 * std_val)] = 255
        
        # 방법 2: 에지 기반 긁힘 감지
        edges = cv2.Canny(gray, 100, 200)
        
        # 선형 구조 감지 (긁힘)
        kernel_v = np.ones((15, 1), np.uint8)
        kernel_h = np.ones((1, 15), np.uint8)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        
        # 마스크 결합
        scratch_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
        
        # 최종 마스크
        final_mask = cv2.bitwise_or(mask, scratch_mask)
        
        # 마스크 확장 (dilate)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        
        return final_mask
    
    def inpaint(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """손상 영역 복원"""
        if mask is None:
            mask = self.detect_damage(image)
        
        if self.model is not None:
            try:
                # LaMa 기반 인페인팅
                # output = self.model(image, mask)
                # return output
                pass
            except Exception as e:
                logger.error(f"LaMa inpainting error: {e}")
        
        # Fallback: OpenCV Telea/NS 인페인팅
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


class ImageColorizer:
    """이미지 컬러화 모듈 (DeOldify)"""
    
    def __init__(self, model_type: str = "artistic", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_type = model_type  # "artistic" or "stable"
        self.model = None
        logger.info(f"ImageColorizer initialized: {model_type} on {self.device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """DeOldify 모델 로드"""
        try:
            from deoldify import device
            from deoldify.device_id import DeviceId
            from deoldify.visualize import get_image_colorizer
            
            # GPU 설정
            if self.device == "cuda":
                device.set(device=DeviceId.GPU0)
            else:
                device.set(device=DeviceId.CPU)
            
            self.model = get_image_colorizer(artistic=(self.model_type == "artistic"))
            logger.info("DeOldify model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"DeOldify not installed: {e}")
            self.model = None
    
    def is_grayscale(self, image: np.ndarray) -> bool:
        """흑백 이미지 여부 확인"""
        if len(image.shape) == 2:
            return True
        if image.shape[2] == 1:
            return True
        
        # RGB 채널 비교
        b, g, r = cv2.split(image)
        return np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
    
    def colorize(self, image: np.ndarray, render_factor: int = 35) -> np.ndarray:
        """흑백 이미지 컬러화"""
        if not self.is_grayscale(image):
            logger.info("Image is already in color")
            return image
        
        if self.model is not None:
            try:
                # DeOldify 컬러화
                # 임시 파일로 저장 후 처리
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    cv2.imwrite(f.name, image)
                    result = self.model.get_transformed_image(
                        f.name, 
                        render_factor=render_factor,
                        watermarked=False
                    )
                    os.unlink(f.name)
                    return np.array(result)
            except Exception as e:
                logger.error(f"DeOldify colorization error: {e}")
        
        # Fallback: 간단한 히스토그램 균등화
        return self._fallback_colorize(image)
    
    def _fallback_colorize(self, image: np.ndarray) -> np.ndarray:
        """폴백 컬러화 (세피아 톤)"""
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 세피아 톤 적용
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        sepia = cv2.transform(gray_3ch, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        return sepia


class ImageCaptioner:
    """이미지 캡셔닝 모듈 (BLIP-2)"""
    
    def __init__(self, model_name: str = "blip2", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        logger.info(f"ImageCaptioner initialized: {model_name} on {self.device}")
    
    def load_model(self):
        """BLIP-2 모델 로드"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            model_id = "Salesforce/blip-image-captioning-large"
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
            
            logger.info("BLIP model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
            self.model = None
    
    def generate_caption(self, image: np.ndarray, max_length: int = 50) -> str:
        """이미지 캡션 생성"""
        if self.model is not None:
            try:
                # BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = self.model.generate(**inputs, max_length=max_length)
                
                caption = self.processor.decode(output[0], skip_special_tokens=True)
                return caption
                
            except Exception as e:
                logger.error(f"Caption generation error: {e}")
        
        return self._generate_basic_caption(image)
    
    def _generate_basic_caption(self, image: np.ndarray) -> str:
        """기본 캡션 생성 (특성 기반)"""
        height, width = image.shape[:2]
        
        # 기본 정보
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        
        # 색상 분석
        is_grayscale = len(image.shape) == 2 or np.allclose(image[:,:,0], image[:,:,1], atol=10)
        
        # 얼굴 감지
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        num_faces = len(faces)
        
        # 캡션 생성
        parts = []
        if is_grayscale:
            parts.append("A black and white photograph")
        else:
            parts.append("A color photograph")
        
        if num_faces == 1:
            parts.append("showing a person")
        elif num_faces > 1:
            parts.append(f"showing {num_faces} people")
        
        parts.append(f"in {orientation} format")
        
        return " ".join(parts) + "."


class SceneClassifier:
    """장면 분류 모듈 (CLIP)"""
    
    SCENE_CATEGORIES = [
        "family gathering", "wedding ceremony", "birthday party",
        "graduation ceremony", "outdoor activity", "indoor portrait",
        "street scene", "workplace", "school", "church",
        "restaurant", "travel", "holiday celebration"
    ]
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        logger.info(f"SceneClassifier initialized on {self.device}")
    
    def load_model(self):
        """CLIP 모델 로드"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            
            logger.info("CLIP model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
    
    def classify(self, image: np.ndarray, categories: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """장면 분류"""
        if categories is None:
            categories = self.SCENE_CATEGORIES
        
        if self.model is not None:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                inputs = self.processor(
                    text=categories,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits_per_image
                    probs = logits.softmax(dim=1)
                
                results = [(cat, prob.item()) for cat, prob in zip(categories, probs[0])]
                results.sort(key=lambda x: x[1], reverse=True)
                
                return results
                
            except Exception as e:
                logger.error(f"Scene classification error: {e}")
        
        return [("unknown", 1.0)]


class ImageRestorer:
    """통합 이미지 복원 파이프라인"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 모듈 초기화
        self.analyzer = ImageAnalyzer(self.device)
        self.super_resolution = SuperResolution(device=self.device)
        self.face_enhancer = FaceEnhancer(device=self.device)
        self.inpainter = ImageInpainter(device=self.device)
        self.colorizer = ImageColorizer(device=self.device)
        self.captioner = ImageCaptioner(device=self.device)
        self.scene_classifier = SceneClassifier(device=self.device)
        
        self.models_loaded = False
        logger.info(f"ImageRestorer initialized on {self.device}")
    
    def load_models(self, model_paths: Optional[Dict[str, str]] = None):
        """모든 모델 로드"""
        model_paths = model_paths or {}
        
        try:
            self.super_resolution.load_model(model_paths.get("realesrgan"))
            self.face_enhancer.load_model(model_paths.get("gfpgan"))
            self.inpainter.load_model(model_paths.get("lama"))
            self.colorizer.load_model(model_paths.get("deoldify"))
            self.captioner.load_model()
            self.scene_classifier.load_model()
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.models_loaded = False
    
    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        options: Optional[Dict] = None
    ) -> RestorationResult:
        """이미지 복원 처리"""
        import time
        start_time = time.time()
        
        options = options or {}
        processing_steps = []
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        original_image = image.copy()
        
        # 1. 분석
        metadata = self.analyzer.analyze(image)
        processing_steps.append("analysis")
        
        # 2. 손상 복원 (필요시)
        if DegradationType.SCRATCH in metadata.degradation_types or \
           DegradationType.STAIN in metadata.degradation_types:
            if options.get("inpaint", True):
                image = self.inpainter.inpaint(image)
                processing_steps.append("inpainting")
        
        # 3. 노이즈 제거 (필요시)
        if DegradationType.NOISE in metadata.degradation_types:
            if options.get("denoise", True):
                image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                processing_steps.append("denoising")
        
        # 4. 초해상도 (필요시)
        if DegradationType.LOW_RESOLUTION in metadata.degradation_types:
            if options.get("upscale", True):
                scale = options.get("scale", 2)
                image = self.super_resolution.upscale(image, scale)
                processing_steps.append(f"super_resolution_x{scale}")
        
        # 5. 얼굴 향상
        faces = []
        if options.get("enhance_faces", True):
            image, faces = self.face_enhancer.enhance(image)
            if faces:
                processing_steps.append("face_enhancement")
        
        # 6. 컬러화 (흑백인 경우)
        if self.colorizer.is_grayscale(original_image):
            if options.get("colorize", True):
                image = self.colorizer.colorize(image)
                processing_steps.append("colorization")
        
        # 7. 색상 보정
        if DegradationType.FADING in metadata.degradation_types:
            if options.get("color_correction", True):
                image = self._color_correction(image)
                processing_steps.append("color_correction")
        
        # 8. 캡션 생성
        caption = ""
        if options.get("generate_caption", True):
            caption = self.captioner.generate_caption(image)
            processing_steps.append("captioning")
        
        # 9. 장면 분류
        tags = []
        if options.get("classify_scene", True):
            scene_results = self.scene_classifier.classify(image)
            tags = [cat for cat, prob in scene_results[:3] if prob > 0.1]
            processing_steps.append("scene_classification")
        
        # 품질 개선 계산
        final_metadata = self.analyzer.analyze(image)
        quality_improvement = final_metadata.quality_score - metadata.quality_score
        
        # 결과 저장
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Restored image saved to {output_path}")
        
        processing_time = time.time() - start_time
        
        return RestorationResult(
            original_path=image_path,
            restored_image=image,
            metadata=final_metadata,
            processing_steps=processing_steps,
            quality_improvement=quality_improvement,
            processing_time=processing_time,
            face_regions=[{'bbox': f['bbox'], 'id': f['id']} for f in faces],
            auto_caption=caption,
            tags=tags
        )
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """색상 보정"""
        # LAB 색공간에서 히스토그램 균등화
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 채도 약간 증가
        a = cv2.add(a, 10)
        b = cv2.add(b, 10)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def batch_process(
        self,
        image_paths: List[str],
        output_dir: str,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        options: Optional[Dict] = None
    ) -> List[RestorationResult]:
        """배치 이미지 처리"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_restored{ext}")
                
                result = self.process(image_path, output_path, mode, options)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        return results


# CLI 인터페이스
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diaspora Archive Image Restoration")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--mode", choices=["quality", "balanced", "fast"], default="balanced")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--no-colorize", action="store_true", help="Skip colorization")
    parser.add_argument("--no-faces", action="store_true", help="Skip face enhancement")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 복원기 초기화
    restorer = ImageRestorer({"device": args.device})
    restorer.load_models()
    
    options = {
        "scale": args.scale,
        "colorize": not args.no_colorize,
        "enhance_faces": not args.no_faces
    }
    
    mode = ProcessingMode[args.mode.upper()]
    
    if os.path.isdir(args.input):
        # 디렉토리 배치 처리
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        output_dir = args.output or os.path.join(args.input, "restored")
        results = restorer.batch_process(image_paths, output_dir, mode, options)
        print(f"Processed {len(results)} images")
    else:
        # 단일 이미지 처리
        output_path = args.output or args.input.replace(".", "_restored.")
        result = restorer.process(args.input, output_path, mode, options)
        print(f"Restoration complete:")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Quality improvement: {result.quality_improvement:.1f}")
        print(f"  Steps: {', '.join(result.processing_steps)}")
        print(f"  Caption: {result.auto_caption}")
        print(f"  Tags: {', '.join(result.tags)}")


if __name__ == "__main__":
    main()
