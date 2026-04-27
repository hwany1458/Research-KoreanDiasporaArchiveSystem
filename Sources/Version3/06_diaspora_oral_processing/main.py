"""
06_diaspora_oral_processing - 구술 처리 모듈
메인 진입점
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import OralProcessingPipeline


def main():
    parser = argparse.ArgumentParser(description="디아스포라 구술 처리")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--num-speakers', type=int, default=None)
    parser.add_argument('--language', '-l', default='ko')
    parser.add_argument('--whisper-model', default='base')
    parser.add_argument('--no-diarization', action='store_true')
    parser.add_argument('--hf-token', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = OralProcessingPipeline(
        whisper_model=args.whisper_model,
        device=args.device,
        hf_token=args.hf_token,
        verbose=args.verbose
    )
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = pipeline.process_interview(
        input_path=input_path,
        output_dir=output_dir,
        num_speakers=args.num_speakers,
        language=args.language,
        do_diarization=not args.no_diarization,
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"oral_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
