"""
degrade.py
합성 열화 (Synthetic Degradation) 모듈

학위논문 정량 평가를 위해 깨끗한 컬러 이미지(Ground Truth)에 
디아스포라 아카이브 사진과 유사한 열화를 인위적으로 적용합니다.

설계 근거:
    - Real-ESRGAN(Wang et al., 2021) 논문의 second-order degradation model 참고
    - 디아스포라 사진의 실제 열화 양상(저해상도, 흑백, 노이즈, 블러, 압축)을 반영
    - 재현성을 위해 random seed 고정 가능

열화 파이프라인:
    GT 컬러 이미지
      → (1) 다운샘플링 (4배)
      → (2) 가우시안 블러
      → (3) 가우시안 노이즈
      → (4) 흑백 변환
      → (5) JPEG 압축
      → 열화된 이미지

References:
    Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Blind 
    Super-Resolution with Pure Synthetic Data. ICCVW.
"""

import os
import io
import random
from pathlib import Path
from typing import Optional, Union, Dict, Any

import cv2
import numpy as np
from PIL import Image


def _imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """한글 경로 안전 이미지 로더."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_unicode(path: Union[str, Path], img: np.ndarray,
                     quality: int = 95) -> bool:
    """한글 경로 안전 이미지 저장."""
    path = str(path)
    ext = os.path.splitext(path)[1] or '.jpg'
    encode_params = []
    if ext.lower() in ('.jpg', '.jpeg'):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    try:
        success, encoded = cv2.imencode(ext, img, encode_params)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


class SyntheticDegradation:
    """
    합성 열화 적용기.
    
    동일한 시드로 호출 시 동일한 열화 결과를 보장하여 재현성 확보.
    """
    
    DEFAULT_PARAMS = {
        # 다운샘플링 (저해상도화)
        'downscale_factor': 4,
        # 가우시안 블러
        'blur_sigma_range': (0.5, 2.0),
        # 가우시안 노이즈 (픽셀 표준편차)
        'noise_sigma_range': (5, 20),
        # 흑백 변환 가중치 (BT.601 표준)
        'grayscale_weights': (0.299, 0.587, 0.114),
        # JPEG 압축 품질
        'jpeg_quality_range': (50, 80),
    }
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 42
    ):
        """
        Args:
            params: 열화 파라미터 dict (None이면 기본값)
            seed: 재현성을 위한 random seed
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
    
    def degrade(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        이미지에 합성 열화 적용.
        
        Args:
            image: 입력 이미지 (경로 또는 numpy array, BGR)
            output_path: 결과 저장 경로 (None이면 메모리에만)
        
        Returns:
            {
                'degraded': np.ndarray (열화된 이미지, BGR),
                'original_size': tuple,
                'degraded_size': tuple,
                'params_used': dict (실제 사용된 파라미터)
            }
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = _imread_unicode(image)
            if img is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없음: {image}")
        else:
            img = image.copy()
        
        original_size = (img.shape[1], img.shape[0])  # (W, H)
        params_used: Dict[str, Any] = {}
        
        # ─────────────────────────────────────────────
        # (1) 다운샘플링 (저해상도화)
        # ─────────────────────────────────────────────
        scale = self.params['downscale_factor']
        new_w = max(1, original_size[0] // scale)
        new_h = max(1, original_size[1] // scale)
        # 보간 방식을 랜덤하게 선택 (현실적 다양성)
        interp_choices = [
            cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC
        ]
        interp = self._rng.choice(interp_choices)
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        params_used['downscale_factor'] = scale
        params_used['interpolation'] = {
            cv2.INTER_AREA: 'AREA',
            cv2.INTER_LINEAR: 'LINEAR',
            cv2.INTER_CUBIC: 'CUBIC'
        }[interp]
        
        # ─────────────────────────────────────────────
        # (2) 가우시안 블러
        # ─────────────────────────────────────────────
        sigma_min, sigma_max = self.params['blur_sigma_range']
        sigma = self._rng.uniform(sigma_min, sigma_max)
        # 커널 크기는 시그마에 비례, 홀수
        ksize = max(3, int(sigma * 4) | 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        params_used['blur_sigma'] = round(sigma, 2)
        
        # ─────────────────────────────────────────────
        # (3) 가우시안 노이즈
        # ─────────────────────────────────────────────
        noise_min, noise_max = self.params['noise_sigma_range']
        noise_sigma = self._rng.uniform(noise_min, noise_max)
        # numpy의 RNG는 별도 시드 적용
        np_rng = np.random.RandomState(self._rng.randint(0, 2**31 - 1))
        noise = np_rng.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        params_used['noise_sigma'] = round(noise_sigma, 2)
        
        # ─────────────────────────────────────────────
        # (4) 흑백 변환 (BT.601 표준 가중평균)
        # ─────────────────────────────────────────────
        wr, wg, wb = self.params['grayscale_weights']
        # OpenCV는 BGR
        gray = (
            img[:, :, 2].astype(np.float32) * wr +
            img[:, :, 1].astype(np.float32) * wg +
            img[:, :, 0].astype(np.float32) * wb
        )
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        # 흑백을 다시 3채널로 (이후 처리 호환성)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        params_used['grayscale_weights'] = (wr, wg, wb)
        
        # ─────────────────────────────────────────────
        # (5) JPEG 압축
        # ─────────────────────────────────────────────
        q_min, q_max = self.params['jpeg_quality_range']
        quality = self._rng.randint(q_min, q_max)
        success, buf = cv2.imencode(
            '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if success:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        params_used['jpeg_quality'] = quality
        
        degraded_size = (img.shape[1], img.shape[0])
        
        # 저장
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _imwrite_unicode(output_path, img, quality=95)
        
        return {
            'degraded': img,
            'original_size': original_size,
            'degraded_size': degraded_size,
            'params_used': params_used
        }
    
    def degrade_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict[str, Any]:
        """
        디렉토리 내 모든 이미지에 열화 적용.
        
        Returns:
            {
                'total': 처리 시도 수,
                'succeeded': 성공 수,
                'failed': 실패 수,
                'records': [...]
            }
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])
        
        records = []
        succeeded = 0
        failed = 0
        
        for img_path in images:
            output_path = output_dir / f"{img_path.stem}_degraded.jpg"
            try:
                result = self.degrade(img_path, output_path)
                records.append({
                    'input': str(img_path),
                    'output': str(output_path),
                    'success': True,
                    **{k: v for k, v in result.items() if k != 'degraded'}
                })
                succeeded += 1
            except Exception as e:
                records.append({
                    'input': str(img_path),
                    'success': False,
                    'error': str(e)
                })
                failed += 1
        
        return {
            'total': len(images),
            'succeeded': succeeded,
            'failed': failed,
            'records': records
        }


# CLI 테스트
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="합성 열화 적용기")
    parser.add_argument('--input', '-i', required=True, help='입력 이미지 또는 디렉토리')
    parser.add_argument('--output', '-o', required=True, help='출력 경로')
    parser.add_argument('--seed', type=int, default=42, help='재현성 시드')
    args = parser.parse_args()
    
    degrader = SyntheticDegradation(seed=args.seed)
    
    input_path = Path(args.input)
    if input_path.is_dir():
        result = degrader.degrade_directory(input_path, args.output)
        print(f"처리 완료: 성공 {result['succeeded']}/{result['total']}")
    else:
        result = degrader.degrade(input_path, args.output)
        print(f"열화 완료: {args.output}")
        print(f"  원본 크기: {result['original_size']}")
        print(f"  열화 크기: {result['degraded_size']}")
        print(f"  사용 파라미터: {json.dumps(result['params_used'], indent=2)}")
