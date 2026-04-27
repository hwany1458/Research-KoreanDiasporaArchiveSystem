#!/usr/bin/env python3
"""
main.py
AI 기반 이미지 복원 시스템 - 메인 실행 파일

사용법:
    python main.py --input input.jpg --output output.jpg
    python main.py --config config.yaml --input ./data/input --output ./data/output --batch
    
Author: YongHwan Kim
Date: 2026
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import ImageRestorationPipeline, ProcessingOptions


def print_banner():
    """배너 출력"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   AI 기반 한인 디아스포라 기록 유산 디지털화 시스템          ║
║   Image Processing Module v1.0.0                             ║
║                                                              ║
║   - 초해상도 복원 (Real-ESRGAN)                              ║
║   - 얼굴 향상 (GFPGAN)                                       ║
║   - 흑백 컬러화 (DeOldify)                                   ║
║   - 이미지 분석 (BLIP, CLIP, face_recognition)               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="AI 기반 이미지 복원 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 필수 인자
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="입력 이미지 또는 디렉토리 경로")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="출력 이미지 또는 디렉토리 경로")
    
    # 설정
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="설정 파일 경로 (기본: config.yaml)")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="연산 장치")
    
    # 처리 모드
    parser.add_argument("--batch", "-b", action="store_true",
                        help="배치 처리 모드 (디렉토리 입력)")
    parser.add_argument("--report", "-r", type=str, default=None,
                        help="처리 보고서 JSON 출력 경로")
    
    # 단계별 활성화/비활성화
    parser.add_argument("--no-sr", action="store_true",
                        help="초해상도 변환 비활성화")
    parser.add_argument("--no-face", action="store_true",
                        help="얼굴 향상 비활성화")
    parser.add_argument("--no-color", action="store_true",
                        help="흑백 컬러화 비활성화")
    parser.add_argument("--no-analysis", action="store_true",
                        help="이미지 분석 비활성화")
    
    # 추가 옵션
    parser.add_argument("--save-intermediate", action="store_true",
                        help="중간 단계 결과 저장")
    parser.add_argument("--force", action="store_true",
                        help="조건 무시하고 모든 단계 강제 실행")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 출력")
    
    args = parser.parse_args()
    
    # 설정 파일 확인
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: 설정 파일을 찾을 수 없음: {config_path}")
        print("기본 설정을 사용합니다.")
        config_path = None
    
    # 처리 옵션 구성
    options = ProcessingOptions(
        enable_super_resolution=not args.no_sr,
        enable_face_enhancement=not args.no_face,
        enable_colorization=not args.no_color,
        enable_analysis=not args.no_analysis,
        save_intermediate=args.save_intermediate,
        conditional_execution=not args.force
    )
    
    # 파이프라인 초기화
    try:
        pipeline = ImageRestorationPipeline(
            device=args.device,
            config_path=str(config_path) if config_path else None,
            options=options if config_path is None else None,
            lazy_load=True
        )
    except Exception as e:
        print(f"Error: 파이프라인 초기화 실패: {e}")
        sys.exit(1)
    
    # 입력 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 경로를 찾을 수 없음: {input_path}")
        sys.exit(1)
    
    # 진행 상황 콜백
    def stage_progress(stage: str, current: int, total: int):
        if args.verbose:
            print(f"  [{current}/{total}] {stage} 처리 중...")
    
    # 처리 수행
    if args.batch or input_path.is_dir():
        # 배치 처리
        if input_path.is_dir():
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            input_paths = []
            for ext in extensions:
                input_paths.extend(input_path.glob(f"*{ext}"))
                input_paths.extend(input_path.glob(f"*{ext.upper()}"))
        else:
            input_paths = [input_path]
        
        if not input_paths:
            print(f"Error: 처리할 이미지를 찾을 수 없음: {input_path}")
            sys.exit(1)
        
        print(f"\n총 {len(input_paths)}개 이미지 처리 시작...\n")
        
        def batch_progress(current, total, result):
            status = "✓" if result.success else "✗"
            stages = ", ".join(result.stages_applied) if result.stages_applied else "없음"
            print(f"[{current:3d}/{total}] {status} {Path(result.input_path).name}")
            if args.verbose:
                print(f"         적용: {stages}")
                print(f"         시간: {result.total_time:.2f}초")
        
        results = pipeline.process_batch(
            input_paths,
            args.output,
            progress_callback=batch_progress
        )
        
        # 요약
        success = sum(1 for r in results if r.success)
        total_time = sum(r.total_time for r in results)
        
        print("\n" + "=" * 60)
        print("처리 완료")
        print("=" * 60)
        print(f"성공: {success}/{len(results)}")
        print(f"총 처리 시간: {total_time:.2f}초")
        print(f"평균 처리 시간: {total_time/len(results):.2f}초/장")
        
    else:
        # 단일 이미지 처리
        print(f"\n이미지 처리 중: {input_path}\n")
        
        result = pipeline.process(
            input_path,
            args.output,
            progress_callback=stage_progress
        )
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("처리 결과")
        print("=" * 60)
        
        if result.success:
            print(f"✓ 처리 성공")
            print(f"  출력: {result.output_path}")
            print(f"  원본 크기: {result.original_size[0]}x{result.original_size[1]}")
            print(f"  최종 크기: {result.final_size[0]}x{result.final_size[1]}")
            print(f"  적용된 단계: {', '.join(result.stages_applied) or '없음'}")
            
            if result.stages_skipped:
                print(f"  건너뛴 단계:")
                for s in result.stages_skipped:
                    print(f"    - {s}")
            
            if result.caption:
                print(f"  캡션: {result.caption}")
            
            if result.scenes:
                print(f"  장면: {', '.join([f'{s[0]}({s[1]:.2f})' for s in result.scenes[:3]])}")
            
            if result.face_count > 0:
                print(f"  감지된 얼굴: {result.face_count}개")
            
            print(f"  처리 시간: {result.total_time:.2f}초")
            
            if args.verbose and result.stage_times:
                print(f"  단계별 시간:")
                for stage, t in result.stage_times.items():
                    print(f"    - {stage}: {t:.2f}초")
        else:
            print(f"✗ 처리 실패")
            for error in result.errors:
                print(f"  에러: {error}")
        
        results = [result]
    
    # 보고서 저장
    if args.report:
        pipeline.save_report(results, args.report)
        print(f"\n보고서 저장: {args.report}")
    
    print()


if __name__ == "__main__":
    main()
