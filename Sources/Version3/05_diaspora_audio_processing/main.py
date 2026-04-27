"""
05_diaspora_audio_processing - 음성 처리 모듈
메인 진입점

기능:
    - 음성 → 텍스트 (Whisper)
    - 다국어 처리 (한국어/영어 자동 감지)
    - 타임스탬프 정렬
    - SRT 자막 생성

사용법:
    python main.py --input data/input/audio.wav
    python main.py --input data/input/audio.wav --language ko
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AudioProcessingPipeline


def main():
    parser = argparse.ArgumentParser(description="디아스포라 음성 처리")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--language', '-l', default=None,
                        help='언어 (ko/en/auto)')
    parser.add_argument('--model', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = AudioProcessingPipeline(
        model_size=args.model,
        device=args.device,
        verbose=args.verbose
    )
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch or input_path.is_dir():
        results = pipeline.process_directory(
            input_path, output_dir, language=args.language
        )
    else:
        result = pipeline.process_file(
            input_path, output_dir, language=args.language
        )
        results = [result]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"audio_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
