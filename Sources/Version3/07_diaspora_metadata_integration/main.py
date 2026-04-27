"""
07_diaspora_metadata_integration - 메타데이터 통합 모듈
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MetadataIntegrationPipeline


def main():
    parser = argparse.ArgumentParser(description="디아스포라 메타데이터 통합")
    parser.add_argument('--input', '-i', required=True,
                        help='02~06 모듈의 JSON 결과들이 있는 디렉토리')
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--no-neo4j', action='store_true')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687')
    parser.add_argument('--neo4j-user', default='neo4j')
    parser.add_argument('--neo4j-password', default='password')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = MetadataIntegrationPipeline(
        use_neo4j=not args.no_neo4j,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        verbose=args.verbose
    )
    
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = pipeline.integrate(input_dir, output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"integration_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
