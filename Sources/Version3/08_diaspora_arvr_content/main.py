"""
08_diaspora_arvr_content - AR/VR 콘텐츠 생성 모듈
메인 진입점

기능:
    - 처리된 사진/문서를 웹 기반 가상 갤러리로 생성 (HTML/JS)
    - 모바일 AR 메타데이터 생성 (AR.js 호환)
    - Unity AR Foundation / XR Toolkit용 데이터 export
    - 3D 공간 배치 자동 산출

본 모듈은 발표 시연용 골격으로, 완전한 Unity 통합은 별도의
Unity 프로젝트에서 처리됩니다 (Unity 2022.3 LTS 기반).

사용법:
    python main.py --input data/input/processed/ --type web
    python main.py --input data/input/processed/ --type unity
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ARVRContentPipeline


def main():
    parser = argparse.ArgumentParser(description="디아스포라 AR/VR 콘텐츠 생성")
    parser.add_argument('--input', '-i', required=True,
                        help='03 모듈의 결과 디렉토리 또는 통합 JSON')
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--type', '-t',
                        choices=['web', 'unity', 'ar_marker', 'all'],
                        default='web',
                        help='생성할 콘텐츠 유형')
    parser.add_argument('--gallery-title', default='디아스포라 디지털 아카이브')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = ARVRContentPipeline(verbose=args.verbose)
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = pipeline.generate(
        input_path=input_path,
        output_dir=output_dir,
        content_type=args.type,
        gallery_title=args.gallery_title,
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"arvr_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
