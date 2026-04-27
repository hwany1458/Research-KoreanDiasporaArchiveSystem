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
        
        # 배경 업샘플러 설정
        bg_up = None
        if bg_upsampler == 'realesrgan' and REALESRGAN_AVAILABLE:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_up = RealESRGANer(scale=2, model_path=None, model=model, tile=400,
                                 half=self.device == 'cuda', device=self.device)
        
        self.restorer = GFPGANer(
            model_path=model_path, upscale=upscale, arch='clean',
            channel_multiplier=2, bg_upsampler=bg_up, device=self.device
        )
        
        print(f"✓ FaceEnhancementModule 초기화 완료 (버전: {model_version})")
    
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
               only_center_face: bool = False, paste_back: bool = True) -> Dict[str, Any]:
        """얼굴 향상 수행"""
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        original_img = img.copy()
        
        # GFPGAN 적용
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            img, has_aligned=False, only_center_face=only_center_face, paste_back=paste_back
        )
        
        # 결과 변환
        if restored_img is not None:
            enhanced_pil = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        else:
            enhanced_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        
        restored_faces_pil = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in restored_faces
        ]
        
        return {
            'enhanced': enhanced_pil,
            'original': Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)),
            'restored_faces': restored_faces_pil,
            'num_faces': len(restored_faces)
        }
