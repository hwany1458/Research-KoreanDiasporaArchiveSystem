"""
01_diaspora_archive_input - 아카이브 입력 시스템
메인 진입점

기능:
    - 다양한 형식의 디아스포라 자료 입력 처리
    - 자동 자료 유형 분류 (이미지/문서/영상/음성/구술)
    - 메타데이터 추출 및 표준화
    - 후속 모듈(02~06)로의 라우팅 정보 생성

사용법:
    python main.py --input data/input/photo.jpg
    python main.py --input data/input/ --batch
    python main.py --input data/input/ --classify-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ArchiveInputPipeline


def main():
    parser = argparse.ArgumentParser(
        description="디아스포라 아카이브 입력 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True,
                        help='입력 파일 또는 디렉토리')
    parser.add_argument('--output-dir', '-o', default='data/output',
                        help='출력 디렉토리')
    parser.add_argument('--batch', action='store_true',
                        help='디렉토리 일괄 처리')
    parser.add_argument('--classify-only', action='store_true',
                        help='분류만 수행')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = ArchiveInputPipeline(verbose=args.verbose)
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch or input_path.is_dir():
        result = pipeline.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            classify_only=args.classify_only
        )
    else:
        result = pipeline.process_file(
            input_path=input_path,
            output_dir=output_dir,
            classify_only=args.classify_only
        )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifest_path = output_dir / f"manifest_{timestamp}.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n매니페스트 저장: {manifest_path}")


if __name__ == "__main__":
    main()
