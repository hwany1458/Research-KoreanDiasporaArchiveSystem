"""
src/modules/ocr.py
인쇄체 OCR 모듈 (EasyOCR 기반)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


def _imread_unicode(path) -> Optional[np.ndarray]:
    """한글 경로 안전 이미지 로더."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class OCRModule:
    """인쇄체 OCR (EasyOCR 기반)."""
    
    def __init__(
        self,
        languages: List[str] = None,
        device: str = 'cuda'
    ):
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR이 필요합니다: pip install easyocr"
            )
        
        languages = languages or ['ko', 'en']
        gpu = (device == 'cuda')
        
        print(f"  EasyOCR 초기화 중 (언어: {languages}, GPU: {gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        self.languages = languages
    
    def extract_text(
        self,
        image_path: Union[str, Path]
    ) -> Dict:
        """이미지에서 텍스트 추출."""
        img = _imread_unicode(image_path)
        if img is None:
            return {'success': False, 'error': '이미지 로드 실패'}
        
        try:
            results = self.reader.readtext(img)
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # 결과 정리
        extracted_items = []
        full_text_parts = []
        
        for bbox, text, confidence in results:
            extracted_items.append({
                'text': text,
                'confidence': float(confidence),
                'bbox': [[float(x), float(y)] for x, y in bbox],
            })
            full_text_parts.append(text)
        
        return {
            'success': True,
            'item_count': len(extracted_items),
            'full_text': '\n'.join(full_text_parts),
            'items': extracted_items,
            'avg_confidence': (
                float(np.mean([item['confidence'] for item in extracted_items]))
                if extracted_items else 0.0
            ),
        }
