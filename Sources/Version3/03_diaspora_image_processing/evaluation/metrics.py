"""
metrics.py
이미지 품질 정량 지표 산출 모듈

지표 분류:
    1. Full-reference (GT 필요):
       - PSNR (Peak Signal-to-Noise Ratio): 픽셀 차이 기반
       - SSIM (Structural Similarity): 구조적 유사도
       - LPIPS (Learned Perceptual Image Patch Similarity): 인간 지각 유사도
    
    2. No-reference (GT 불필요):
       - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
       - NIQE (Natural Image Quality Evaluator)

References:
    - PSNR/SSIM: 표준 신호처리 지표
    - LPIPS: Zhang et al. (2018), CVPR. https://github.com/richzhang/PerceptualSimilarity
    - BRISQUE: Mittal et al. (2012), TIP.
    - NIQE: Mittal et al. (2013), Signal Process. Lett.

Dependencies:
    pip install scikit-image lpips piq
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

# scikit-image (PSNR, SSIM)
try:
    from skimage.metrics import peak_signal_noise_ratio as ski_psnr
    from skimage.metrics import structural_similarity as ski_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# LPIPS
try:
    import torch
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# piq (BRISQUE 등 No-reference)
try:
    import piq
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False


def _imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """한글 경로 안전 이미지 로더 (BGR)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ensure_same_size(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = 'crop'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    두 이미지의 크기를 맞춤.
    
    Args:
        method: 'crop' (작은 쪽에 맞춤) 또는 'resize' (img2를 img1 크기로)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) == (h2, w2):
        return img1, img2
    
    if method == 'crop':
        h = min(h1, h2)
        w = min(w1, w2)
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    elif method == 'resize':
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return img1, img2


# ──────────────────────────────────────────────────────────────────
# Full-reference 지표
# ──────────────────────────────────────────────────────────────────
def compute_psnr(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: int = 255
) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산.
    
    값이 클수록 좋음 (보통 30+ 이상이 양호).
    Args는 BGR uint8 numpy array.
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image이 필요합니다: pip install scikit-image")
    gt, pred = _ensure_same_size(gt, pred)
    return float(ski_psnr(gt, pred, data_range=data_range))


def compute_ssim(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: int = 255
) -> float:
    """
    SSIM (Structural Similarity) 계산.
    
    값이 1에 가까울수록 좋음 (1.0이 완전 동일).
    multichannel=True로 BGR 3채널 처리.
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image이 필요합니다: pip install scikit-image")
    gt, pred = _ensure_same_size(gt, pred)
    
    # scikit-image 0.19+ 에서는 channel_axis 사용
    try:
        return float(ski_ssim(gt, pred, channel_axis=2, data_range=data_range))
    except TypeError:
        # 구버전 호환
        return float(ski_ssim(gt, pred, multichannel=True, data_range=data_range))


class LPIPSCalculator:
    """
    LPIPS 계산기 (모델을 한 번만 로드하기 위한 클래스).
    
    값이 작을수록 좋음 (0에 가까울수록 GT와 지각적으로 유사).
    """
    
    def __init__(
        self,
        net: str = 'alex',
        device: str = 'cuda'
    ):
        """
        Args:
            net: 'alex' (기본, 빠름) 또는 'vgg' (정확하지만 느림)
            device: 'cuda' or 'cpu'
        """
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips 라이브러리가 필요합니다: pip install lpips\n"
                "그리고 PyTorch가 설치되어 있어야 합니다."
            )
        
        self.device = (
            'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )
        self.net_name = net
        # verbose=False로 로딩 메시지 억제
        self.model = lpips_lib.LPIPS(net=net, verbose=False).to(self.device)
        self.model.eval()
    
    def compute(
        self,
        gt: np.ndarray,
        pred: np.ndarray
    ) -> float:
        """
        LPIPS 거리 계산.
        
        Args는 BGR uint8 numpy array.
        Returns: LPIPS 거리 (작을수록 유사).
        """
        gt, pred = _ensure_same_size(gt, pred)
        
        # BGR → RGB → [-1, 1] 정규화 → torch tensor
        gt_t = self._to_tensor(gt)
        pred_t = self._to_tensor(pred)
        
        with torch.no_grad():
            distance = self.model(gt_t, pred_t).item()
        
        return float(distance)
    
    def _to_tensor(self, img_bgr: np.ndarray) -> 'torch.Tensor':
        """BGR uint8 numpy → [-1,1] RGB torch tensor (1, 3, H, W)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # [0, 255] → [-1, 1]
        normalized = rgb.astype(np.float32) / 127.5 - 1.0
        # (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)


# ──────────────────────────────────────────────────────────────────
# No-reference 지표
# ──────────────────────────────────────────────────────────────────
class NoReferenceMetrics:
    """
    No-reference 품질 지표 계산기 (GT 불필요).
    
    실제 디아스포라 사진 등 GT가 없는 경우에 사용.
    """
    
    def __init__(self, device: str = 'cuda'):
        if not PIQ_AVAILABLE:
            raise ImportError("piq 라이브러리가 필요합니다: pip install piq")
        if not LPIPS_AVAILABLE:
            raise ImportError("torch가 필요합니다 (LPIPS 의존성과 동일)")
        
        self.device = (
            'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )
    
    def _to_tensor(self, img_bgr: np.ndarray) -> 'torch.Tensor':
        """BGR uint8 → [0,1] RGB torch tensor (1, 3, H, W)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def compute_brisque(self, img: np.ndarray) -> float:
        """
        BRISQUE (값이 작을수록 좋음, 일반적으로 0~100).
        
        자연 이미지의 통계적 특성에서 벗어나는 정도를 측정.
        """
        tensor = self._to_tensor(img)
        with torch.no_grad():
            score = piq.brisque(tensor, data_range=1.0).item()
        return float(score)
    
    def compute_all(self, img: np.ndarray) -> Dict[str, float]:
        """
        사용 가능한 모든 No-reference 지표 계산.
        
        Returns:
            {'brisque': float, ...}
        """
        results = {}
        try:
            results['brisque'] = self.compute_brisque(img)
        except Exception as e:
            results['brisque_error'] = str(e)
        return results


# ──────────────────────────────────────────────────────────────────
# 통합 지표 계산기
# ──────────────────────────────────────────────────────────────────
class MetricsEvaluator:
    """
    여러 지표를 한 번에 계산하는 통합 평가기.
    
    Full-reference (GT 있음) 모드와 No-reference (GT 없음) 모드 모두 지원.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        enable_lpips: bool = True,
        enable_brisque: bool = True,
        lpips_net: str = 'alex'
    ):
        self.device = device
        
        # PSNR/SSIM은 항상 활성화 (의존성이 가벼움)
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image이 필요합니다: pip install scikit-image")
        
        # LPIPS는 선택적
        self.lpips_calc: Optional[LPIPSCalculator] = None
        if enable_lpips and LPIPS_AVAILABLE:
            try:
                self.lpips_calc = LPIPSCalculator(net=lpips_net, device=device)
                print(f"✓ LPIPS 모델 로드 완료 ({lpips_net}, {self.lpips_calc.device})")
            except Exception as e:
                print(f"⚠ LPIPS 초기화 실패: {e}")
        elif enable_lpips:
            print("⚠ LPIPS 비활성화 (lpips 라이브러리 미설치)")
        
        # BRISQUE는 선택적
        self.nr_metrics: Optional[NoReferenceMetrics] = None
        if enable_brisque and PIQ_AVAILABLE:
            try:
                self.nr_metrics = NoReferenceMetrics(device=device)
                print(f"✓ No-reference 지표 활성화 (piq)")
            except Exception as e:
                print(f"⚠ No-reference 지표 초기화 실패: {e}")
        elif enable_brisque:
            print("⚠ No-reference 지표 비활성화 (piq 라이브러리 미설치)")
    
    def evaluate_pair(
        self,
        gt: Union[str, Path, np.ndarray],
        pred: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        GT와 예측 결과 한 쌍에 대해 모든 지표 계산.
        
        Returns:
            {
                'psnr': float,
                'ssim': float,
                'lpips': float (활성화된 경우),
                'gt_size': tuple,
                'pred_size': tuple,
                'errors': list
            }
        """
        # 이미지 로드
        if isinstance(gt, (str, Path)):
            gt_img = _imread_unicode(gt)
            if gt_img is None:
                return {'error': f"GT 이미지 로드 실패: {gt}"}
        else:
            gt_img = gt
        
        if isinstance(pred, (str, Path)):
            pred_img = _imread_unicode(pred)
            if pred_img is None:
                return {'error': f"예측 이미지 로드 실패: {pred}"}
        else:
            pred_img = pred
        
        result: Dict[str, Any] = {
            'gt_size': (gt_img.shape[1], gt_img.shape[0]),
            'pred_size': (pred_img.shape[1], pred_img.shape[0]),
            'errors': []
        }
        
        # PSNR
        try:
            result['psnr'] = compute_psnr(gt_img, pred_img)
        except Exception as e:
            result['errors'].append(f"PSNR 실패: {e}")
        
        # SSIM
        try:
            result['ssim'] = compute_ssim(gt_img, pred_img)
        except Exception as e:
            result['errors'].append(f"SSIM 실패: {e}")
        
        # LPIPS
        if self.lpips_calc is not None:
            try:
                result['lpips'] = self.lpips_calc.compute(gt_img, pred_img)
            except Exception as e:
                result['errors'].append(f"LPIPS 실패: {e}")
        
        return result
    
    def evaluate_no_reference(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        No-reference 지표만 계산 (GT 없는 실제 사진용).
        """
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image)
            if img is None:
                return {'error': f"이미지 로드 실패: {image}"}
        else:
            img = image
        
        result: Dict[str, Any] = {
            'image_size': (img.shape[1], img.shape[0]),
            'errors': []
        }
        
        if self.nr_metrics is not None:
            nr_result = self.nr_metrics.compute_all(img)
            result.update(nr_result)
        else:
            result['errors'].append("No-reference 지표 미활성화")
        
        return result


# ──────────────────────────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="이미지 품질 지표 계산")
    parser.add_argument('--gt', help='GT 이미지 경로 (full-reference 모드)')
    parser.add_argument('--pred', required=True, help='평가할 이미지 경로')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    evaluator = MetricsEvaluator(device=args.device)
    
    if args.gt:
        # Full-reference
        result = evaluator.evaluate_pair(args.gt, args.pred)
        print("=== Full-Reference 지표 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # No-reference
        result = evaluator.evaluate_no_reference(args.pred)
        print("=== No-Reference 지표 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
