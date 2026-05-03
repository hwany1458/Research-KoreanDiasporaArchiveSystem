"""
face_enhancement.py
얼굴 향상 모듈 - GFPGAN 기반

GFPGAN을 활용하여 저화질 얼굴 이미지를 복원합니다.

검출 정책:
    1차 검출: RetinaFace (facexlib, threshold=0.5 표준값)
    2차 검출: HOG-based (face_recognition, fallback)
    
    1차 검출이 실패하거나 0개를 잡으면 2차 검출기를 시도하여
    아카이브의 흑백·노후 사진에서 underdetection을 방지합니다.

Reference:
    Wang, X., et al. (2021). Towards Real-World Blind Face Restoration 
    with Generative Facial Prior. CVPR.
    Deng, J., et al. (2020). RetinaFace: Single-Shot Multi-Level Face 
    Localisation in the Wild. CVPR.
    Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients 
    for Human Detection. CVPR.
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

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


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
    """Windows 한글 경로 대응 이미지 로더."""
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
    
    검출 정책:
        - RetinaFace (facexlib): threshold=0.5, 학계 표준값
        - HOG fallback (face_recognition): RetinaFace가 0개 검출 시 활용
        - 두 검출기 중 하나라도 얼굴을 찾으면 처리 진행
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        model_version: str = '1.4',
        model_path: Optional[str] = None,
        upscale: int = 2,
        bg_upsampler: Optional[str] = 'realesrgan',
        detection_threshold: float = 0.5,
        enable_fallback_detector: bool = True
    ):
        """
        Args:
            device: 연산 장치 ('cuda', 'cpu')
            model_version: GFPGAN 버전 ('1.4', '1.3', '1.2')
            model_path: 가중치 경로 (None이면 weights/ 자동 탐색)
            upscale: 출력 배율 (1, 2, 4)
            bg_upsampler: 배경 업샘플러 ('realesrgan' 또는 None)
            detection_threshold: RetinaFace confidence 임계값 (학계 표준 0.5)
            enable_fallback_detector: True면 HOG 검출기 fallback 활성화
        """
        if not GFPGAN_AVAILABLE:
            raise ImportError("GFPGAN not installed. Run: pip install gfpgan")
        
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model_version = model_version
        self.upscale = upscale
        self.detection_threshold = detection_threshold
        self.enable_fallback_detector = (
            enable_fallback_detector and FACE_RECOGNITION_AVAILABLE
        )
        
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
        
        # 검출기 캐시 (지연 로드)
        self._retinaface_detector = None
        
        print(f"✓ FaceEnhancementModule 초기화 완료 "
              f"(버전: {model_version}, 장치: {self.device})")
        print(f"  GFPGAN 가중치: {model_path}")
        if bg_up is not None:
            print(f"  배경 업샘플러: 활성화 (RealESRGAN_x2plus)")
        else:
            print(f"  배경 업샘플러: 비활성화")
        print(f"  검출 정책: RetinaFace(threshold={detection_threshold})"
              f"{' + HOG fallback' if self.enable_fallback_detector else ''}")
    
    def _init_bg_upsampler(self) -> Optional[RealESRGANer]:
        """배경 업샘플러(RealESRGAN x2) 초기화."""
        bg_weight_path = _resolve_weight_path(BG_UPSAMPLER_WEIGHT)
        
        if not os.path.isfile(bg_weight_path):
            print(f"⚠ 배경 업샘플러 가중치를 찾을 수 없음: {bg_weight_path}")
            print(f"  → 배경 업샘플링 없이 진행합니다.")
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
            return None
    
    # ──────────────────────────────────────────────────────────
    # 얼굴 검출 (1차 + fallback)
    # ──────────────────────────────────────────────────────────
    def _detect_with_retinaface(self, img_bgr: np.ndarray) -> List:
        """
        RetinaFace 기반 1차 검출 (facexlib).
        
        Returns:
            bbox 리스트. 각 bbox는 [x1, y1, x2, y2, ...] 형태.
            검출 실패 시 빈 리스트 반환.
        """
        try:
            if self._retinaface_detector is None:
                from facexlib.detection import init_detection_model
                self._retinaface_detector = init_detection_model(
                    'retinaface_resnet50', device=self.device
                )
            bboxes = self._retinaface_detector.detect_faces(
                img_bgr, self.detection_threshold
            )
            return list(bboxes) if bboxes is not None else []
        except Exception as e:
            print(f"  ⚠ RetinaFace 검출 실패: {e}")
            return []
    
    def _detect_with_hog(self, img_bgr: np.ndarray) -> List:
        """
        HOG 기반 2차 검출 (face_recognition / dlib).
        
        흑백·노후 사진에서 RetinaFace가 놓치는 얼굴을 보완합니다.
        
        Returns:
            bbox 리스트. [x1, y1, x2, y2] 형태로 RetinaFace와 호환.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []
        try:
            # face_recognition은 RGB를 사용
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # face_recognition 반환: (top, right, bottom, left)
            locs = face_recognition.face_locations(img_rgb, model='hog')
            # RetinaFace 호환 포맷으로 변환: [x1, y1, x2, y2]
            bboxes = [[left, top, right, bottom]
                      for (top, right, bottom, left) in locs]
            return bboxes
        except Exception as e:
            print(f"  ⚠ HOG 검출 실패: {e}")
            return []
    
    def _detect_faces(self, image) -> List:
        """
        통합 얼굴 검출: RetinaFace 1차 → HOG 2차 fallback.
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image, cv2.IMREAD_COLOR)
            if img is None:
                return []
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        # 1차: RetinaFace
        faces = self._detect_with_retinaface(img)
        
        # 2차: 실패 또는 0개 시 HOG fallback
        if not faces and self.enable_fallback_detector:
            faces = self._detect_with_hog(img)
            if faces:
                print(f"  ℹ HOG fallback이 {len(faces)}개의 얼굴을 검출")
        
        return faces
    
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
                1 for f in faces
                if min(f[2] - f[0], f[3] - f[1]) < min_face_size
            )
            if small_faces > 0:
                return True, f"저화질 얼굴 {small_faces}개 감지", len(faces)
            return False, f"얼굴 품질 양호 ({len(faces)}개)", len(faces)
        except Exception:
            return True, "얼굴 감지 확인 불가", 0
    
    # ──────────────────────────────────────────────────────────
    # 얼굴 향상
    # ──────────────────────────────────────────────────────────
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
                    f"이미지를 읽을 수 없습니다: {image}"
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
