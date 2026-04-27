"""
src/modules/ner.py
개체명 인식 모듈 (KoELECTRA 기반)
"""

import re
from typing import Dict, List

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class NERModule:
    """한국어 개체명 인식 (KoELECTRA 기반)."""
    
    DEFAULT_MODEL = "Leo97/KoELECTRA-small-v3-modu-ner"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = 'cuda'
    ):
        self.device = -1
        self.ner_pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            print("  ⚠ transformers 미설치, NER 비활성화")
            return
        
        try:
            model_name = model_name or self.DEFAULT_MODEL
            device_id = 0 if (device == 'cuda' and torch.cuda.is_available()) else -1
            
            print(f"  NER 모델 로딩 중 ({model_name})...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device_id
            )
            self.device = device_id
        except Exception as e:
            print(f"  ⚠ NER 초기화 실패: {e}")
            self.ner_pipeline = None
    
    def extract_entities(self, text: str) -> Dict:
        """텍스트에서 개체명 추출."""
        if not text or not text.strip():
            return {'success': False, 'error': '빈 텍스트'}
        
        # 1. 모델 기반 NER
        entities_by_type: Dict[str, List[str]] = {}
        if self.ner_pipeline is not None:
            try:
                ner_results = self.ner_pipeline(text)
                for ent in ner_results:
                    entity_type = ent.get('entity_group', 'OTHER')
                    entity_text = ent.get('word', '').strip()
                    if entity_text:
                        entities_by_type.setdefault(entity_type, []).append(entity_text)
            except Exception as e:
                pass
        
        # 2. 정규식 fallback (날짜·연도)
        years = re.findall(r'(\b1[89]\d{2}|20\d{2})\b', text)
        if years:
            entities_by_type.setdefault('YEAR', []).extend(years)
        
        dates = re.findall(r'\d{1,2}월\s*\d{1,2}일', text)
        if dates:
            entities_by_type.setdefault('DATE', []).extend(dates)
        
        # 중복 제거
        for k in entities_by_type:
            entities_by_type[k] = list(dict.fromkeys(entities_by_type[k]))
        
        return {
            'success': True,
            'entity_count': sum(len(v) for v in entities_by_type.values()),
            'entities_by_type': entities_by_type,
        }
