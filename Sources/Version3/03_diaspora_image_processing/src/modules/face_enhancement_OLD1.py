"""
face_enhancement.py
얼굴 향상 모듈 - GFPGAN 기반

GFPGAN을 활용하여 저화질 얼굴 이미지를 복원합니다.

Reference:
    Wang, X., et al. (2021). Towards Real-World Blind Face Restoration 
    with Generative Facial Prior. CVPR.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any, List
from pathlib import Path

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# 공용 헬퍼
# ──────────────────────────────────────────────────────────────────
def _project_root() -> str:
    """이 파일 기준 프로젝트 루트(03_diaspora_image_processing) 경로."""
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _resolve_weight_path(filename: str) -> str:
    """프로젝트 루트의 weights/ 폴더에 있는 가중치 파일 경로."""
    return os.path.join(_project_root(), 'weights', filename)


def _imread_unicode(path: Union[str, Path],
                    flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    한글 경로 대응 이미지 로더.
    Windows에서 cv2.imread()가 한글 경로를 처리하지 못하는 문제를 우회한다.
    """
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


# 가중치 파일명 매핑
GFPGAN_WEIGHTS = {
    '1.4': 'GFPGANv1.4.pth',
    '1.3': 'GFPGANv1.3.pth',
    '1.2': 'GFPGANCleanv1-NoCE-C2.pth',
}
BG_UPSAMPLER_WEIGHT = 'RealESRGAN_x2plus.pth'


class FaceEnhancementModule:
    """
    얼굴 향상 모듈 - GFPGAN 기반
    
    StyleGAN2의 사전학습된 얼굴 생성 능력을 활용하여 blind face restoration을 수행합니다.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        model_version: str = '1.4',
        model_path: Optional[str] = None,
        upscale: int = 2,
        bg_upsampler: Optional[str] = 'realesrgan'
    ):
        if not GFPGAN_AVAILABLE:
            raise ImportError("GFPGAN not installed. Run: pip install gfpgan")
        
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model_version = model_version
        self.upscale = upscale
        
        # ──────────────────────────────────────────────────────────
        # GFPGAN 본체 가중치 경로 해석
        # ──────────────────────────────────────────────────────────
        if model_path is None:
            weight_filename = GFPGAN_WEIGHTS.get(
                model_version, GFPGAN_WEIGHTS['1.4']
            )
            model_path = _resolve_weight_path(weight_filename)
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GFPGAN 가중치 파일을 찾을 수 없습니다: {model_path}\n"
                f"  → 다음 위치에 가중치 파일을 다운로드하세요:\n"
                f"    {os.path.dirname(model_path)}\n"
                f"  → 다운로드 URL: "
                f"https://github.com/TencentARC/GFPGAN/releases"
            )
        
        self.model_path = model_path
        
        # ──────────────────────────────────────────────────────────
        # 배경 업샘플러 설정 (선택적, 실패해도 본체는 동작)
        # ──────────────────────────────────────────────────────────
        bg_up = None
        if bg_upsampler == 'realesrgan' and REALESRGAN_AVAILABLE:
            bg_up = self._init_bg_upsampler()
        
        # ──────────────────────────────────────────────────────────
        # GFPGAN 복원기 초기화
        # ──────────────────────────────────────────────────────────
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_up,
            device=self.device
        )
        
        print(f"✓ FaceEnhancementModule 초기화 완료 "
              f"(버전: {model_version}, 장치: {self.device})")
        print(f"  GFPGAN 가중치: {model_path}")
        if bg_up is not None:
            print(f"  배경 업샘플러: 활성화 (RealESRGAN_x2plus)")
        else:
            print(f"  배경 업샘플러: 비활성화")
    
    def _init_bg_upsampler(self) -> Optional[RealESRGANer]:
        """
        배경 업샘플러(RealESRGAN x2) 초기화.
        가중치 파일이 없으면 경고만 출력하고 None 반환 → 본체는 정상 동작.
        """
        bg_weight_path = _resolve_weight_path(BG_UPSAMPLER_WEIGHT)
        
        if not os.path.isfile(bg_weight_path):
            print(f"⚠ 배경 업샘플러 가중치를 찾을 수 없음: {bg_weight_path}")
            print(f"  → 배경 업샘플링 없이 진행합니다.")
            print(f"  → 활성화하려면 다음 파일을 다운로드하세요: "
                  f"https://github.com/xinntao/Real-ESRGAN/releases/"
                  f"download/v0.2.1/{BG_UPSAMPLER_WEIGHT}")
            return None
        
        try:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2
            )
            return RealESRGANer(
                scale=2,
                model_path=bg_weight_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=self.device == 'cuda',
                device=self.device
            )
        except Exception as e:
            print(f"⚠ 배경 업샘플러 초기화 실패: {e}")
            print(f"  → 배경 업샘플링 없이 진행합니다.")
            return None
    
    def should_process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        min_face_size: int = 64
    ) -> Tuple[bool, str, int]:
        """얼굴 향상 처리가 필요한지 판단"""
        try:
            faces = self._detect_faces(image)
            if not faces:
                return False, "얼굴이 감지되지 않음", 0
            
            small_faces = sum(
                1 for f in faces if min(f[2] - f[0], f[3] - f[1]) < min_face_size
            )
            if small_faces > 0:
                return True, f"저화질 얼굴 {small_faces}개 감지", len(faces)
            return False, f"얼굴 품질 양호 ({len(faces)}개)", len(faces)
        except Exception:
            return True, "얼굴 감지 확인 불가", 0
    
    def _detect_faces(self, image):
        """간단한 얼굴 감지 (facexlib 사용)"""
        try:
            from facexlib.detection import init_detection_model
            detector = init_detection_model(
                'retinaface_resnet50', device=self.device
            )
            
            if isinstance(image, (str, Path)):
                img = _imread_unicode(image)
                if img is None:
                    return []
            elif isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            bboxes = detector.detect_faces(img, 0.97)
            return bboxes
        except Exception:
            return []
    
    def enhance(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        only_center_face: bool = False,
        paste_back: bool = True
    ) -> Dict[str, Any]:
        """얼굴 향상 수행"""
        # 이미지 로드 (한글 경로 대응)
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"이미지를 읽을 수 없습니다: {image}\n"
                    f"  → 파일 존재 여부 및 형식을 확인하세요."
                )
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # 알파 채널 / 그레이스케일 정규화
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        original_img = img.copy()
        
        # GFPGAN 적용
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=only_center_face,
            paste_back=paste_back
        )
        
        # 결과 변환
        if restored_img is not None:
            enhanced_pil = Image.fromarray(
                cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            )
        else:
            enhanced_pil = Image.fromarray(
                cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            )
        
        restored_faces_pil = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in restored_faces
        ]
        
        return {
            'enhanced': enhanced_pil,
            'original': Image.fromarray(
                cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ),
            'restored_faces': restored_faces_pil,
            'num_faces': len(restored_faces)
        }
