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
except ImportError as e:
    GFPGAN_AVAILABLE = False
    print(f"[WARNING] GFPGAN import 실패: {e}")

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    REALESRGAN_AVAILABLE = False
    print(f"[WARNING] Real-ESRGAN import 실패 (face_enhancement): {e}")


def _find_model_path(model_dir: str, filename: str) -> Optional[str]:
    """
    모델 파일을 여러 경로에서 자동으로 탐색합니다.
    """
    candidates = [
        # 1. 프로젝트 루트 기준
        os.path.join("models", model_dir, filename),
        # 2. 이 소스파일 위치 기준 (src/modules/ → ../../models/)
        os.path.join(os.path.dirname(__file__), "..", "..", "models", model_dir, filename),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None


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

        # GFPGAN 모델 경로 자동 탐색
        if model_path is None:
            gfpgan_filename = f"GFPGANv{model_version}.pth"
            model_path = _find_model_path("gfpgan", gfpgan_filename)
            if model_path:
                print(f"  [Face] GFPGAN 모델 경로 자동 탐색 성공: {model_path}")
            else:
                raise FileNotFoundError(
                    f"GFPGAN 모델 파일을 찾을 수 없습니다: {gfpgan_filename}\n"
                    f"다음 경로를 확인하세요: models/gfpgan/{gfpgan_filename}\n"
                    f"모델 다운로드: python download_models.py --gfpgan"
                )

        # 배경 업샘플러 설정 (Real-ESRGAN x2, 모델 경로 자동 탐색)
        bg_up = None
        if bg_upsampler == 'realesrgan' and REALESRGAN_AVAILABLE:
            bg_model_path = _find_model_path("realesrgan", "RealESRGAN_x2plus.pth")
            if bg_model_path:
                print(f"  [Face] 배경 업샘플러 모델 경로: {bg_model_path}")
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=2)
                bg_up = RealESRGANer(
                    scale=2, model_path=bg_model_path, model=model,
                    tile=400, half=self.device == 'cuda', device=self.device
                )
            else:
                print("  [Face] 배경 업샘플러 모델 없음 - 배경 업샘플링 비활성화")

        self.restorer = GFPGANer(
            model_path=model_path, upscale=upscale, arch='clean',
            channel_multiplier=2, bg_upsampler=bg_up, device=self.device
        )
        
        print(f"✓ FaceEnhancementModule 초기화 완료 (버전: {model_version}, 장치: {self.device})")
    
    def should_process(self, image: Union[str, Path, Image.Image, np.ndarray],
                       min_face_size: int = 64) -> Tuple[bool, str, int]:
        """얼굴 향상 처리가 필요한지 판단"""
        try:
            faces = self._detect_faces(image)
            if not faces:
                return False, "얼굴이 감지되지 않음", 0
            
            small_faces = sum(1 for f in faces if min(f[2]-f[0], f[3]-f[1]) < min_face_size)
            if small_faces > 0:
                return True, f"저화질 얼굴 {small_faces}개 감지", len(faces)
            return False, f"얼굴 품질 양호 ({len(faces)}개)", len(faces)
        except:
            return True, "얼굴 감지 확인 불가", 0
    
    def _detect_faces(self, image):
        """간단한 얼굴 감지 (facexlib 사용)"""
        try:
            from facexlib.detection import init_detection_model
            detector = init_detection_model('retinaface_resnet50', device=self.device)
            
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
            elif isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            bboxes = detector.detect_faces(img, 0.97)
            return bboxes
        except:
            return []
    
    def enhance(self, image: Union[str, Path, Image.Image, np.ndarray],
                only_center_face: bool = False, paste_back: bool = True,
                max_size: int = 2048) -> Dict[str, Any]:
        """
        얼굴 향상 수행.

        고해상도 이미지(> max_size)는 GFPGAN 처리를 위해
        임시로 리사이즈 후 처리하고 원본 해상도로 복원합니다.
        GFPGAN은 내부적으로 RetinaFace로 얼굴을 감지하므로
        이미지가 너무 크면 감지 실패 → 리사이즈가 핵심 해결책입니다.
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()

        original_img = img.copy()
        orig_h, orig_w = img.shape[:2]

        # 고해상도 이미지 처리: max_size 이하로 리사이즈 후 GFPGAN 적용
        scale_factor = 1.0
        if max(orig_h, orig_w) > max_size:
            scale_factor = max_size / max(orig_h, orig_w)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            img_for_gfpgan = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  [Face] 고해상도 감지 대응: {orig_w}x{orig_h} → {new_w}x{new_h}")
        else:
            img_for_gfpgan = img

        # GFPGAN 적용
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            img_for_gfpgan, has_aligned=False,
            only_center_face=only_center_face, paste_back=paste_back
        )

        # 처리 결과를 원본 해상도로 복원
        if restored_img is not None and scale_factor < 1.0:
            restored_img = cv2.resize(
                restored_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
            )

        # 결과 변환
        if restored_img is not None:
            enhanced_pil = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        else:
            enhanced_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

        restored_faces_pil = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in restored_faces
        ]

        num_faces = len(restored_faces)
        print(f"  [Face] 감지된 얼굴: {num_faces}개")

        return {
            'enhanced': enhanced_pil,
            'original': Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)),
            'restored_faces': restored_faces_pil,
            'num_faces': num_faces
        }
