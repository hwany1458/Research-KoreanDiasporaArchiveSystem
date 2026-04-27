"""
batch_process.py
이미지 처리 모듈 일괄 처리 스크립트

기능:
    - data/input/ 폴더의 모든 이미지를 자동 처리
    - 각 결과를 data/output/batch_<timestamp>/restored/ 에 저장
    - 원본+결과를 나란히 배치한 비교 이미지를 comparisons/ 에 저장 (학위논문 figure)
    - 각 처리에 대한 상세 JSON 리포트를 reports/ 에 저장
    - 전체 일괄 처리 요약을 batch_summary.json/.md 로 저장

정책:
    - 흑백 사진: 컬러화 자동 적용
    - 컬러 사진: 컬러화 자동 스킵 (원본 색상 보존)
    - 처리 실패 시 스킵하고 다음 이미지로 진행 (graceful)

사용법:
    python batch_process.py
    python batch_process.py --input-dir data/input --output-dir data/output --verbose
    python batch_process.py --limit 5         # 최대 5장만 처리 (테스트용)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# 프로젝트 루트 경로 추가 (src/ 모듈 import용)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# 지원 이미지 확장자
# ──────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ──────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────────
def collect_input_images(input_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """input_dir에서 처리할 이미지 파일들을 수집 (재귀 X)."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if limit is not None:
        images = images[:limit]
    
    return images


def make_comparison_image(
    original_pil: Image.Image,
    restored_pil: Image.Image,
    label_height: int = 40,
    margin: int = 10
) -> Image.Image:
    """
    원본과 복원 결과를 나란히 배치한 비교 이미지 생성.
    학위논문 figure에 적합한 형태.
    """
    from PIL import ImageDraw, ImageFont
    
    # 두 이미지의 높이를 맞추기 위해 동일한 높이로 리사이즈
    target_height = max(original_pil.height, restored_pil.height)
    
    def resize_keep_ratio(img: Image.Image, h: int) -> Image.Image:
        if img.height == h:
            return img
        new_w = int(img.width * h / img.height)
        return img.resize((new_w, h), Image.Resampling.LANCZOS)
    
    orig_resized = resize_keep_ratio(original_pil, target_height)
    rest_resized = resize_keep_ratio(restored_pil, target_height)
    
    # 캔버스 크기
    canvas_w = orig_resized.width + rest_resized.width + margin * 3
    canvas_h = target_height + label_height + margin * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    
    # 이미지 배치
    canvas.paste(orig_resized, (margin, margin))
    canvas.paste(rest_resized, (margin * 2 + orig_resized.width, margin))
    
    # 라벨
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("malgun.ttf", 28)  # Windows 한글 폰트
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    label_y = target_height + margin + 5
    draw.text(
        (margin + orig_resized.width // 2 - 50, label_y),
        "Original",
        fill='black',
        font=font
    )
    draw.text(
        (margin * 2 + orig_resized.width + rest_resized.width // 2 - 50, label_y),
        "Restored",
        fill='black',
        font=font
    )
    
    return canvas


def serialize_for_json(obj: Any) -> Any:
    """JSON 직렬화 가능한 형태로 변환."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Image.Image):
        return f"<PIL.Image {obj.size} {obj.mode}>"
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """JSON 파일로 저장 (한글 안전, indent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialize_for_json(data), f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────
# 메인 일괄 처리 클래스
# ──────────────────────────────────────────────────────────────────
class BatchProcessor:
    """
    일괄 처리 오케스트레이터.
    
    main.py의 ImageProcessingPipeline을 재사용하여 여러 이미지를 처리하고,
    결과·비교 이미지·JSON 리포트를 모두 생성한다.
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        device: str = 'cuda',
        verbose: bool = False,
        save_comparison: bool = True,
        save_per_image_report: bool = True
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.save_comparison = save_comparison
        self.save_per_image_report = save_per_image_report
        
        # 출력 하위 디렉토리
        self.restored_dir = output_dir / 'restored'
        self.comparison_dir = output_dir / 'comparisons'
        self.report_dir = output_dir / 'reports'
        
        for d in [self.restored_dir, self.comparison_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 파이프라인 (지연 로드)
        self._pipeline = None
    
    def _get_pipeline(self):
        """파이프라인 지연 로드 (모델 로딩이 무거우므로 한 번만)."""
        if self._pipeline is None:
            print("=" * 70)
            print("이미지 처리 파이프라인 초기화 중...")
            print("=" * 70)
            from src.pipeline import ImageProcessingPipeline
            self._pipeline = ImageProcessingPipeline(
                device=self.device,
                lazy_load=True
            )
        return self._pipeline
    
    def _process_single(
        self,
        input_path: Path,
        idx: int,
        total: int
    ) -> Dict[str, Any]:
        """단일 이미지 처리. 실패해도 dict 형태로 결과 반환."""
        stem = input_path.stem
        record: Dict[str, Any] = {
            'index': idx,
            'total': total,
            'input_path': str(input_path),
            'input_filename': input_path.name,
            'started_at': datetime.now().isoformat(),
            'success': False,
            'error': None,
        }
        
        print(f"\n[{idx}/{total}] 처리 중: {input_path.name}")
        print("-" * 70)
        
        t_start = time.time()
        
        try:
            pipeline = self._get_pipeline()
            
            # 출력 경로
            restored_path = self.restored_dir / f"{stem}_restored.jpg"
            
            # 파이프라인 실행
            result = pipeline.process(
                input_path=str(input_path),
                output_path=str(restored_path),
                verbose=self.verbose
            )
            
            elapsed = time.time() - t_start
            
            # 비교 이미지 생성
            comparison_path = None
            if self.save_comparison:
                try:
                    comparison_path = self._save_comparison(
                        input_path, restored_path, stem
                    )
                except Exception as e:
                    print(f"  ⚠ 비교 이미지 생성 실패: {e}")
            
            # 처리 결과 기록
            record.update({
                'success': True,
                'output_path': str(restored_path),
                'comparison_path': str(comparison_path) if comparison_path else None,
                'elapsed_seconds': round(elapsed, 2),
                'pipeline_result': self._extract_pipeline_summary(result),
            })
            
            # 개별 JSON 리포트
            if self.save_per_image_report:
                report_path = self.report_dir / f"{stem}.json"
                write_json(report_path, record)
                record['report_path'] = str(report_path)
            
            # 콘솔 요약
            self._print_single_summary(record)
            
        except Exception as e:
            elapsed = time.time() - t_start
            err_msg = f"{type(e).__name__}: {e}"
            record.update({
                'success': False,
                'error': err_msg,
                'traceback': traceback.format_exc(),
                'elapsed_seconds': round(elapsed, 2),
            })
            print(f"  ✗ 처리 실패: {err_msg}")
            if self.verbose:
                traceback.print_exc()
        
        record['finished_at'] = datetime.now().isoformat()
        return record
    
    def _save_comparison(
        self,
        original_path: Path,
        restored_path: Path,
        stem: str
    ) -> Path:
        """원본+복원 비교 이미지 저장."""
        # 한글 경로 안전 로드
        original_data = np.fromfile(str(original_path), dtype=np.uint8)
        if original_data.size == 0:
            raise ValueError(f"원본 이미지를 읽을 수 없음: {original_path}")
        import cv2
        original_bgr = cv2.imdecode(original_data, cv2.IMREAD_COLOR)
        original_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        
        restored_data = np.fromfile(str(restored_path), dtype=np.uint8)
        if restored_data.size == 0:
            raise ValueError(f"복원 이미지를 읽을 수 없음: {restored_path}")
        restored_bgr = cv2.imdecode(restored_data, cv2.IMREAD_COLOR)
        restored_pil = Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
        
        comparison = make_comparison_image(original_pil, restored_pil)
        
        comparison_path = self.comparison_dir / f"{stem}_comparison.jpg"
        comparison.save(str(comparison_path), quality=92)
        return comparison_path
    
    def _extract_pipeline_summary(self, result: Any) -> Dict[str, Any]:
        """
        pipeline.process()의 반환 객체에서 핵심 정보만 추출.
        (PIL Image 등 직렬화 불가 객체는 메타데이터만 보존)
        """
        if not isinstance(result, dict):
            return {'note': '파이프라인 결과 형식 미지원', 'type': type(result).__name__}
        
        summary: Dict[str, Any] = {}
        
        # 흔한 키들
        for key in ['original_size', 'final_size', 'applied_stages',
                    'skipped_stages', 'skip_reasons', 'caption', 'scene',
                    'num_faces', 'num_detected_faces', 'processing_time',
                    'stage_times', 'backend']:
            if key in result:
                summary[key] = result[key]
        
        # 단계별 결과(상세) 보존: PIL.Image는 사이즈만
        for key, val in result.items():
            if key in summary:
                continue
            if isinstance(val, Image.Image):
                summary[f'{key}_size'] = val.size
            elif isinstance(val, (str, int, float, bool, type(None))):
                summary[key] = val
            elif isinstance(val, dict):
                summary[key] = {
                    k: (v.size if isinstance(v, Image.Image) else v)
                    for k, v in val.items()
                    if isinstance(v, (str, int, float, bool, type(None), tuple, list))
                    or isinstance(v, Image.Image)
                }
            elif isinstance(val, (list, tuple)):
                summary[key] = [
                    (v.size if isinstance(v, Image.Image) else v)
                    for v in val
                    if isinstance(v, (str, int, float, bool, type(None)))
                    or isinstance(v, Image.Image)
                ]
        
        return summary
    
    def _print_single_summary(self, record: Dict[str, Any]) -> None:
        """단일 이미지 처리 결과 콘솔 요약."""
        ps = record.get('pipeline_result', {})
        elapsed = record.get('elapsed_seconds', 0)
        
        print(f"  ✓ 처리 완료 ({elapsed:.1f}초)")
        if 'original_size' in ps and 'final_size' in ps:
            print(f"    크기: {ps['original_size']} → {ps['final_size']}")
        if 'applied_stages' in ps:
            print(f"    적용: {ps['applied_stages']}")
        if 'skipped_stages' in ps and ps['skipped_stages']:
            print(f"    건너뜀: {ps['skipped_stages']}")
        if 'caption' in ps:
            print(f"    캡션: {ps['caption']}")
        if 'num_faces' in ps:
            print(f"    얼굴: {ps['num_faces']}개")
    
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """일괄 처리 실행."""
        # 입력 수집
        images = collect_input_images(self.input_dir, limit=limit)
        if not images:
            print(f"⚠ 처리할 이미지가 없습니다: {self.input_dir}")
            return {'total': 0, 'records': []}
        
        print("=" * 70)
        print(f"일괄 처리 시작")
        print("=" * 70)
        print(f"  입력 디렉토리: {self.input_dir}")
        print(f"  출력 디렉토리: {self.output_dir}")
        print(f"  처리 대상   : {len(images)}장")
        print(f"  비교 이미지 : {'생성' if self.save_comparison else '스킵'}")
        print(f"  개별 리포트 : {'생성' if self.save_per_image_report else '스킵'}")
        print("=" * 70)
        
        records: List[Dict[str, Any]] = []
        batch_start = time.time()
        
        for idx, img_path in enumerate(images, start=1):
            rec = self._process_single(img_path, idx, len(images))
            records.append(rec)
        
        batch_elapsed = time.time() - batch_start
        
        # 일괄 처리 요약
        succeeded = [r for r in records if r['success']]
        failed = [r for r in records if not r['success']]
        
        summary = {
            'batch_started_at': datetime.fromtimestamp(batch_start).isoformat(),
            'batch_finished_at': datetime.now().isoformat(),
            'batch_elapsed_seconds': round(batch_elapsed, 2),
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'total': len(images),
            'succeeded': len(succeeded),
            'failed': len(failed),
            'avg_seconds_per_image': (
                round(sum(r.get('elapsed_seconds', 0) for r in succeeded)
                      / len(succeeded), 2)
                if succeeded else None
            ),
            'records': records,
        }
        
        # JSON 요약 저장
        summary_json_path = self.output_dir / 'batch_summary.json'
        write_json(summary_json_path, summary)
        
        # Markdown 요약 저장 (사람이 읽기 편함)
        summary_md_path = self.output_dir / 'batch_summary.md'
        self._write_markdown_summary(summary_md_path, summary)
        
        # 콘솔 최종 요약
        print()
        print("=" * 70)
        print("일괄 처리 완료")
        print("=" * 70)
        print(f"  총 처리      : {len(images)}장")
        print(f"  성공         : {len(succeeded)}장")
        print(f"  실패         : {len(failed)}장")
        print(f"  총 소요 시간 : {batch_elapsed:.1f}초")
        if succeeded:
            print(f"  평균 처리시간: {summary['avg_seconds_per_image']:.1f}초/장")
        print(f"  요약 (JSON)  : {summary_json_path}")
        print(f"  요약 (MD)    : {summary_md_path}")
        if failed:
            print()
            print(f"  ⚠ 실패한 이미지 ({len(failed)}장):")
            for r in failed:
                print(f"    - {r['input_filename']}: {r['error']}")
        print("=" * 70)
        
        return summary
    
    def _write_markdown_summary(
        self,
        path: Path,
        summary: Dict[str, Any]
    ) -> None:
        """학위논문 자료로 활용 가능한 마크다운 요약."""
        records = summary['records']
        succeeded = [r for r in records if r['success']]
        failed = [r for r in records if not r['success']]
        
        lines = []
        lines.append("# 일괄 이미지 복원 처리 결과")
        lines.append("")
        lines.append(f"- **처리 일시**: {summary['batch_started_at']}")
        lines.append(f"- **입력 디렉토리**: `{summary['input_dir']}`")
        lines.append(f"- **출력 디렉토리**: `{summary['output_dir']}`")
        lines.append(f"- **총 처리**: {summary['total']}장")
        lines.append(f"- **성공/실패**: {summary['succeeded']} / {summary['failed']}")
        lines.append(f"- **총 소요 시간**: {summary['batch_elapsed_seconds']:.1f}초")
        if summary.get('avg_seconds_per_image') is not None:
            lines.append(
                f"- **평균 처리 시간**: {summary['avg_seconds_per_image']:.1f}초/장"
            )
        lines.append("")
        
        # 결과 테이블
        lines.append("## 처리 결과 표")
        lines.append("")
        lines.append("| # | 파일명 | 상태 | 원본 크기 | 최종 크기 | 적용 단계 | 건너뜀 | 시간(초) |")
        lines.append("|---|--------|------|-----------|-----------|-----------|--------|----------|")
        for r in records:
            ps = r.get('pipeline_result', {}) if r['success'] else {}
            status = '✓' if r['success'] else '✗'
            orig = ps.get('original_size', '-')
            final = ps.get('final_size', '-')
            applied = ', '.join(ps.get('applied_stages', [])) if ps.get('applied_stages') else '-'
            skipped_list = ps.get('skipped_stages', [])
            if isinstance(skipped_list, list):
                skipped = ', '.join(skipped_list) if skipped_list else '-'
            elif isinstance(skipped_list, dict):
                skipped = ', '.join(skipped_list.keys()) if skipped_list else '-'
            else:
                skipped = '-'
            elapsed = r.get('elapsed_seconds', 0)
            lines.append(
                f"| {r['index']} | `{r['input_filename']}` | {status} "
                f"| {orig} | {final} | {applied} | {skipped} | {elapsed:.1f} |"
            )
        lines.append("")
        
        # 실패 상세
        if failed:
            lines.append("## 실패한 이미지")
            lines.append("")
            for r in failed:
                lines.append(f"- **{r['input_filename']}**: `{r['error']}`")
            lines.append("")
        
        # 정성 분석을 위한 캡션·장면 표
        if succeeded:
            lines.append("## 이미지 분석 결과")
            lines.append("")
            lines.append("| # | 파일명 | 캡션 | 장면 | 얼굴 |")
            lines.append("|---|--------|------|------|------|")
            for r in succeeded:
                ps = r.get('pipeline_result', {})
                caption = ps.get('caption', '-')
                scene = ps.get('scene', '-')
                num_faces = ps.get('num_faces', '-')
                lines.append(
                    f"| {r['index']} | `{r['input_filename']}` "
                    f"| {caption} | {scene} | {num_faces} |"
                )
            lines.append("")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines), encoding='utf-8')


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="이미지 처리 모듈 일괄 처리 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python batch_process.py
    python batch_process.py --limit 3
    python batch_process.py --input-dir data/input --output-dir data/output --verbose
    python batch_process.py --no-comparison    # 비교 이미지 안 만듦 (속도 ↑)
"""
    )
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='입력 이미지 디렉토리 (기본: data/input)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='출력 디렉토리 (기본: data/output/batch_<timestamp>)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='연산 장치')
    parser.add_argument('--limit', type=int, default=None,
                        help='최대 처리 이미지 수 (테스트용)')
    parser.add_argument('--no-comparison', action='store_true',
                        help='비교 이미지 생성 스킵 (속도 향상)')
    parser.add_argument('--no-report', action='store_true',
                        help='개별 JSON 리포트 생성 스킵')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 입력 디렉토리
    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    
    # 출력 디렉토리 (timestamp 기반)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / 'data' / 'output' / f'batch_{timestamp}'
    else:
        output_dir = Path(args.output_dir).resolve()
    
    # 실행
    processor = BatchProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        device=args.device,
        verbose=args.verbose,
        save_comparison=not args.no_comparison,
        save_per_image_report=not args.no_report,
    )
    
    summary = processor.run(limit=args.limit)
    
    # 종료 코드 (실패 있으면 1)
    sys.exit(0 if summary.get('failed', 0) == 0 else 1)


if __name__ == "__main__":
    main()
