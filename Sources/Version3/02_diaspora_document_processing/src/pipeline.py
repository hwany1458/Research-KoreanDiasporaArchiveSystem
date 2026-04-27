"""
src/pipeline.py
문서 처리 파이프라인
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.modules.ocr import OCRModule
from src.modules.htr import HTRModule
from src.modules.ner import NERModule


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class DocumentProcessingPipeline:
    """문서 처리 통합 파이프라인."""
    
    def __init__(
        self,
        device: str = 'cuda',
        languages: List[str] = None,
        mode: str = 'ocr',  # 'ocr', 'htr', 'both'
        enable_ner: bool = True,
        verbose: bool = False
    ):
        self.device = device
        self.mode = mode
        self.verbose = verbose
        self.languages = languages or ['ko', 'en']
        
        # 모듈 lazy 로딩
        self._ocr: Optional[OCRModule] = None
        self._htr: Optional[HTRModule] = None
        self._ner: Optional[NERModule] = None
        self.enable_ner = enable_ner
        
        if verbose:
            print(f"DocumentProcessingPipeline 초기화 (mode={mode})")
    
    @property
    def ocr(self) -> OCRModule:
        if self._ocr is None:
            self._ocr = OCRModule(languages=self.languages, device=self.device)
        return self._ocr
    
    @property
    def htr(self) -> HTRModule:
        if self._htr is None:
            self._htr = HTRModule(device=self.device)
        return self._htr
    
    @property
    def ner(self) -> NERModule:
        if self._ner is None:
            self._ner = NERModule(device=self.device)
        return self._ner
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path
    ) -> Dict:
        """단일 파일 처리."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(input_path),
            'mode': self.mode,
            'success': False,
        }
        
        if not input_path.exists():
            result['error'] = '파일 없음'
            return result
        
        full_text_parts = []
        
        # OCR 처리
        if self.mode in ('ocr', 'both'):
            try:
                ocr_result = self.ocr.extract_text(input_path)
                result['ocr'] = ocr_result
                if ocr_result.get('success'):
                    full_text_parts.append(ocr_result.get('full_text', ''))
                    if self.verbose:
                        print(f"  OCR: {ocr_result['item_count']}개 텍스트 항목 추출")
            except Exception as e:
                result['ocr_error'] = str(e)
        
        # HTR 처리
        if self.mode in ('htr', 'both'):
            try:
                htr_result = self.htr.extract_text(input_path)
                result['htr'] = htr_result
                if htr_result.get('success'):
                    full_text_parts.append(htr_result.get('text', ''))
                    if self.verbose:
                        print(f"  HTR: '{htr_result['text'][:60]}...'")
            except Exception as e:
                result['htr_error'] = str(e)
        
        # NER 처리
        if self.enable_ner and full_text_parts:
            try:
                combined_text = '\n'.join(full_text_parts)
                ner_result = self.ner.extract_entities(combined_text)
                result['ner'] = ner_result
                if self.verbose and ner_result.get('success'):
                    print(f"  NER: {ner_result['entity_count']}개 개체명 추출")
            except Exception as e:
                result['ner_error'] = str(e)
        
        result['combined_text'] = '\n'.join(full_text_parts)
        result['success'] = bool(full_text_parts)
        result['finished_at'] = datetime.now().isoformat()
        
        return result
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> List[Dict]:
        """디렉토리 일괄 처리."""
        files = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        
        results = []
        for i, file_path in enumerate(files, 1):
            if self.verbose:
                print(f"\n[{i}/{len(files)}] {file_path.name}")
            try:
                result = self.process_file(file_path, output_dir)
                results.append(result)
            except Exception as e:
                results.append({
                    'input_path': str(file_path),
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r.get('success'))
        print(f"\n총 {len(files)}장 중 {success_count}장 처리 성공")
        
        return results
