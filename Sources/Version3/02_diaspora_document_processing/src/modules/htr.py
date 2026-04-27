"""
src/modules/htr.py
손글씨 텍스트 인식 모듈 (TrOCR 기반)
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class HTRModule:
    """손글씨 인식 (TrOCR 기반)."""
    
    DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = 'cuda'
    ):
        if not TROCR_AVAILABLE:
            raise ImportError(
                "TrOCR이 필요합니다: pip install transformers"
            )
        
        model_name = model_name or self.DEFAULT_MODEL
        self.device = (
            'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )
        
        print(f"  TrOCR 모델 로딩 중 ({model_name})...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_text(
        self,
        image_path: Union[str, Path, Image.Image]
    ) -> Dict:
        """손글씨 이미지에서 텍스트 추출."""
        try:
            if isinstance(image_path, (str, Path)):
                pil_image = Image.open(str(image_path)).convert('RGB')
            else:
                pil_image = image_path
            
            pixel_values = self.processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=256)
            
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return {
                'success': True,
                'text': text,
                'model': self.DEFAULT_MODEL,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }
