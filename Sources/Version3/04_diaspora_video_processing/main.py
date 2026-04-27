"""
04_diaspora_video_processing - 영상 처리 모듈
메인 진입점

기능:
    - 영상 메타데이터 추출
    - 핵심 프레임 추출 (uniform sampling)
    - 장면 분할 (PySceneDetect)
    - 음성 트랙 분리 (ffmpeg)
    - 자막 생성 (Whisper 연동)

사용법:
    python main.py --input data/input/family.mp4
    python main.py --input data/input/family.mp4 --num-keyframes 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import VideoProcessingPipeline


def main():
    parser = argparse.ArgumentParser(description="디아스포라 영상 처리")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output-dir', '-o', default='data/output')
    parser.add_argument('--num-keyframes', type=int, default=10,
                        help='추출할 키프레임 수')
    parser.add_argument('--no-scenes', action='store_true',
                        help='장면 분할 스킵')
    parser.add_argument('--no-audio', action='store_true',
                        help='음성 추출 스킵')
    parser.add_argument('--no-subtitle', action='store_true',
                        help='자막 생성 스킵')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pipeline = VideoProcessingPipeline(
        device=args.device,
        verbose=args.verbose
    )
    
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = pipeline.process_video(
        input_path=input_path,
        output_dir=output_dir,
        num_keyframes=args.num_keyframes,
        do_scene_detection=not args.no_scenes,
        do_audio_extraction=not args.no_audio,
        do_subtitle=not args.no_subtitle,
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"video_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
