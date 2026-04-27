"""
02_diaspora_document_processing - 문서 처리 모듈
메인 진입점

기능:
    - 인쇄체 OCR (EasyOCR, 한글/영어/한자)
    - 손글씨 HTR (TrOCR)
    - 개체명 인식 (KoBERT/KoELECTRA)
    - 문서 구조 분석

사용법:
    python main.py --input data/input/letter.jpg
    python main.py --input data/input/ --batch
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DocumentProcessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="디아스포라 문서 처리 (OCR/HTR)"
    )
    parser.add_argument('--input', '-i', required=True,
                        help='입력 이미지 또는 디렉토리')
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--mode', choices=['ocr', 'htr', 'both'], default='ocr',
                        help='처리 모드 (인쇄체/손글씨/모두)')
    parser.add_argument('--languages', default='ko,en',
                        help='OCR 언어 (쉼표 구분)')
    parser.add_argument('--no-ner', action='store_true',
                        help='개체명 인식 비활성화')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    languages = [l.strip() for l in args.languages.split(',')]
    
    pipeline = DocumentProcessingPipeline(
        device=args.device,
        languages=languages,
        mode=args.mode,
        enable_ner=not args.no_ner,
        verbose=args.verbose
    )
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch or input_path.is_dir():
        results = pipeline.process_directory(input_path, output_dir)
    else:
        result = pipeline.process_file(input_path, output_dir)
        results = [result]
    
    # JSON 리포트
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"document_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
