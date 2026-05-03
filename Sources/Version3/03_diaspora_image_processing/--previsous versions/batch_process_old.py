"""
batch_process.py
이미지 처리 모듈 일괄 처리 스크립트

설계 철학:
    원본 시스템의 ImageRestorationPipeline.process_batch()를 직접 호출하여
    배치 처리 및 JSON 리포트 생성은 그대로 활용한다. 이 스크립트는 그 위에
    학위논문 자료로 활용 가능한 추가 산출물(비교 이미지, 마크다운 요약)을 생성한다.

기능:
    - data/input/ 폴더의 모든 이미지를 자동 처리 (process_batch 호출)
    - 각 결과를 data/output/batch_<timestamp>/restored/ 에 저장
    - 원본+결과 나란히 배치한 비교 이미지를 comparisons/ 에 저장
    - 원본 시스템의 JSON 리포트 (batch_summary.json) 활용
    - 학위논문 figure로 활용 가능한 마크다운 요약 (batch_summary.md) 추가 생성

정책:
    - 흑백 사진: 컬러화 자동 적용 (is_grayscale 판정)
    - 컬러 사진: 컬러화 자동 스킵 (원본 색상 보존)
    - 처리 실패 시 스킵하고 다음 이미지로 진행

사용법:
    python batch_process.py
    python batch_process.py --input-dir data/input --verbose
    python batch_process.py --limit 3        # 처음 3장만 처리 (테스트용)
    python batch_process.py --no-comparison  # 비교 이미지 생략 (속도 ↑)
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

# 프로젝트 루트 (이 파일의 위치 기준)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# 지원 이미지 확장자
# ──────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ──────────────────────────────────────────────────────────────────
# 한글 경로 안전 I/O 헬퍼
# ──────────────────────────────────────────────────────────────────
def _imread_unicode(path) -> Optional[np.ndarray]:
    """Windows 한글 경로 대응 이미지 로더."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# 비교 이미지 생성 (학위논문 figure용)
# ──────────────────────────────────────────────────────────────────
def make_comparison_image(
    original_pil: Image.Image,
    restored_pil: Image.Image,
    label_height: int = 50,
    margin: int = 12
) -> Image.Image:
    """
    원본과 복원 결과를 나란히 배치한 비교 이미지 생성.
    
    두 이미지의 높이를 맞춰 동일 크기로 표시하고,
    각 이미지 아래에 'Original' / 'Restored' 라벨을 추가한다.
    학위논문 figure에 그대로 사용할 수 있는 형태.
    """
    from PIL import ImageDraw, ImageFont
    
    # 두 이미지의 높이를 맞추기 위해 동일 높이로 리사이즈
    target_height = max(original_pil.height, restored_pil.height)
    # 비교 이미지가 너무 커지지 않도록 상한
    if target_height > 2400:
        target_height = 2400
    
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
    
    # 라벨 폰트
    draw = ImageDraw.Draw(canvas)
    font = None
    for font_name in ["malgun.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 32)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    
    label_y = target_height + margin + 8
    
    def draw_centered(text: str, x_center: int) -> None:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = len(text) * 16
        draw.text((x_center - text_w // 2, label_y), text, fill='black', font=font)
    
    draw_centered(
        "Original",
        x_center=margin + orig_resized.width // 2
    )
    draw_centered(
        "Restored",
        x_center=margin * 2 + orig_resized.width + rest_resized.width // 2
    )
    
    return canvas


def save_comparison(
    input_path: Path,
    restored_path: Path,
    comparison_dir: Path
) -> Optional[Path]:
    """비교 이미지를 저장하고 경로를 반환. 실패하면 None."""
    try:
        original_bgr = _imread_unicode(input_path)
        restored_bgr = _imread_unicode(restored_path)
        
        if original_bgr is None:
            print(f"  ⚠ 원본 이미지를 읽을 수 없어 비교 이미지 생략: {input_path.name}")
            return None
        if restored_bgr is None:
            print(f"  ⚠ 복원 이미지를 읽을 수 없어 비교 이미지 생략: {restored_path.name}")
            return None
        
        original_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        restored_pil = Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
        
        comparison = make_comparison_image(original_pil, restored_pil)
        
        comparison_path = comparison_dir / f"{input_path.stem}_comparison.jpg"
        comparison.save(str(comparison_path), quality=92)
        return comparison_path
    except Exception as e:
        print(f"  ⚠ 비교 이미지 생성 실패: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# 마크다운 요약 (학위논문 자료용)
# ──────────────────────────────────────────────────────────────────
def write_markdown_summary(
    md_path: Path,
    results: list,
    batch_started_at: str,
    batch_finished_at: str,
    batch_elapsed: float,
    input_dir: Path,
    output_dir: Path,
    comparison_paths: dict
) -> None:
    """학위논문 자료로 활용 가능한 마크다운 요약 생성."""
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    avg_time = (
        sum(r.total_time for r in succeeded) / len(succeeded)
        if succeeded else 0.0
    )
    
    lines = []
    lines.append("# 일괄 이미지 복원 처리 결과")
    lines.append("")
    lines.append(f"- **시작 시각**: {batch_started_at}")
    lines.append(f"- **종료 시각**: {batch_finished_at}")
    lines.append(f"- **입력 디렉토리**: `{input_dir}`")
    lines.append(f"- **출력 디렉토리**: `{output_dir}`")
    lines.append(f"- **총 처리**: {len(results)}장")
    lines.append(f"- **성공/실패**: {len(succeeded)} / {len(failed)}")
    lines.append(f"- **총 소요 시간**: {batch_elapsed:.1f}초")
    if succeeded:
        lines.append(f"- **평균 처리 시간**: {avg_time:.1f}초/장")
    lines.append("")
    
    # 처리 결과 표
    lines.append("## 처리 결과")
    lines.append("")
    lines.append("| # | 파일명 | 상태 | 원본 크기 | 최종 크기 | 적용 단계 | 건너뜀 | 시간(초) |")
    lines.append("|---|--------|------|-----------|-----------|-----------|--------|----------|")
    for i, r in enumerate(results, start=1):
        filename = Path(r.input_path).name
        status = '✓' if r.success else '✗'
        orig = f"{r.original_size[0]}×{r.original_size[1]}" if r.original_size != (0, 0) else '-'
        final = f"{r.final_size[0]}×{r.final_size[1]}" if r.final_size != (0, 0) else '-'
        applied = ', '.join(r.stages_applied) if r.stages_applied else '-'
        skipped = ', '.join(r.stages_skipped) if r.stages_skipped else '-'
        lines.append(
            f"| {i} | `{filename}` | {status} "
            f"| {orig} | {final} | {applied} | {skipped} | {r.total_time:.1f} |"
        )
    lines.append("")
    
    # 단계별 평균 처리 시간
    if succeeded:
        stage_totals: dict = {}
        stage_counts: dict = {}
        for r in succeeded:
            for stage, t in r.stage_times.items():
                stage_totals[stage] = stage_totals.get(stage, 0.0) + t
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        if stage_totals:
            lines.append("## 단계별 평균 처리 시간")
            lines.append("")
            lines.append("| 단계 | 평균 시간 (초) | 적용 횟수 |")
            lines.append("|------|----------------|-----------|")
            for stage, total in stage_totals.items():
                count = stage_counts[stage]
                avg = total / count if count else 0
                lines.append(f"| {stage} | {avg:.2f} | {count} |")
            lines.append("")
    
    # 이미지 분석 결과
    if succeeded:
        lines.append("## 이미지 분석 결과")
        lines.append("")
        lines.append("| # | 파일명 | 캡션 | 주요 장면 | 얼굴 수 |")
        lines.append("|---|--------|------|-----------|---------|")
        for i, r in enumerate(results, start=1):
            if not r.success:
                continue
            filename = Path(r.input_path).name
            caption = (r.caption or '-').replace('|', '\\|')
            top_scene = (
                f"{r.scenes[0][0]} ({r.scenes[0][1]:.2f})"
                if r.scenes else '-'
            )
            lines.append(
                f"| {i} | `{filename}` | {caption} | {top_scene} | {r.face_count} |"
            )
        lines.append("")
    
    # 비교 이미지 링크 (생성된 경우)
    if comparison_paths:
        lines.append("## 비교 이미지")
        lines.append("")
        lines.append("학위논문 figure로 활용 가능한 원본/복원 비교 이미지:")
        lines.append("")
        for filename, comp_path in comparison_paths.items():
            try:
                rel_path = Path(comp_path).relative_to(md_path.parent)
            except ValueError:
                rel_path = Path(comp_path)
            lines.append(f"- `{filename}` → [{rel_path}]({rel_path})")
        lines.append("")
    
    # 실패 상세
    if failed:
        lines.append("## 실패한 이미지")
        lines.append("")
        for r in failed:
            filename = Path(r.input_path).name
            errors = '; '.join(r.errors) if r.errors else '알 수 없는 오류'
            lines.append(f"- **{filename}**: `{errors}`")
        lines.append("")
    
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines), encoding='utf-8')


# ──────────────────────────────────────────────────────────────────
# 입력 이미지 수집
# ──────────────────────────────────────────────────────────────────
def collect_input_images(input_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """input_dir에서 처리할 이미지 파일들을 수집 (재귀 X, 정렬됨)."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if limit is not None:
        images = images[:limit]
    
    return images


# ──────────────────────────────────────────────────────────────────
# 메인 일괄 처리 함수
# ──────────────────────────────────────────────────────────────────
def run_batch(
    input_dir: Path,
    output_dir: Path,
    device: str = 'cuda',
    limit: Optional[int] = None,
    save_comparison: bool = True,
    save_per_image_report: bool = True,
    verbose: bool = False
) -> int:
    """
    일괄 처리 실행.
    
    Returns:
        실패한 이미지 수 (종료 코드용; 0이면 모두 성공)
    """
    # 입력 수집
    images = collect_input_images(input_dir, limit=limit)
    if not images:
        print(f"⚠ 처리할 이미지가 없습니다: {input_dir}")
        return 0
    
    # 출력 하위 디렉토리
    restored_dir = output_dir / 'restored'
    comparison_dir = output_dir / 'comparisons'
    report_dir = output_dir / 'reports'
    for d in [restored_dir, comparison_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("일괄 처리 시작")
    print("=" * 70)
    print(f"  입력 디렉토리: {input_dir}")
    print(f"  출력 디렉토리: {output_dir}")
    print(f"  처리 대상   : {len(images)}장")
    print(f"  비교 이미지 : {'생성' if save_comparison else '스킵'}")
    print(f"  개별 리포트 : {'생성' if save_per_image_report else '스킵'}")
    print("=" * 70)
    
    # ──────────────────────────────────────────────────────────
    # 파이프라인 초기화 (모델은 1회만 로드)
    # ──────────────────────────────────────────────────────────
    print("\n[1/3] 파이프라인 초기화...")
    from src.pipeline import ImageRestorationPipeline
    
    pipeline = ImageRestorationPipeline(
        device=device,
        lazy_load=True
    )
    
    # ──────────────────────────────────────────────────────────
    # 진행 콜백
    # ──────────────────────────────────────────────────────────
    def progress_cb(current: int, total: int, result) -> None:
        """원본 process_batch의 진행 콜백."""
        status = '✓' if result.success else '✗'
        filename = Path(result.input_path).name
        elapsed = result.total_time
        
        print(f"\n[{current}/{total}] {status} {filename}  ({elapsed:.1f}초)")
        if result.success:
            if result.original_size != (0, 0) and result.final_size != (0, 0):
                print(f"      크기: {result.original_size} → {result.final_size}")
            if result.stages_applied:
                print(f"      적용: {', '.join(result.stages_applied)}")
            if result.stages_skipped:
                print(f"      건너뜀: {', '.join(result.stages_skipped)}")
            if result.caption:
                print(f"      캡션: {result.caption}")
            if result.face_count:
                print(f"      얼굴: {result.face_count}개")
        else:
            for err in result.errors:
                print(f"      에러: {err}")
    
    # ──────────────────────────────────────────────────────────
    # 일괄 처리 실행 (원본 process_batch 호출)
    # ──────────────────────────────────────────────────────────
    print("\n[2/3] 일괄 처리 실행...")
    batch_start = time.time()
    batch_started_at = datetime.now().isoformat()
    
    results = pipeline.process_batch(
        input_paths=images,
        output_dir=restored_dir,
        progress_callback=progress_cb
    )
    
    batch_elapsed = time.time() - batch_start
    batch_finished_at = datetime.now().isoformat()
    
    # ──────────────────────────────────────────────────────────
    # 추가 산출물 생성
    # ──────────────────────────────────────────────────────────
    print("\n[3/3] 추가 산출물 생성 (비교 이미지·리포트)...")
    
    # 비교 이미지 생성
    comparison_paths: dict = {}
    if save_comparison:
        for r in results:
            if not r.success:
                continue
            input_path = Path(r.input_path)
            restored_path = Path(r.output_path)
            comp_path = save_comparison_safe(
                input_path, restored_path, comparison_dir
            )
            if comp_path is not None:
                comparison_paths[input_path.name] = str(comp_path)
    
    # 개별 JSON 리포트 (원본 save_report 활용)
    if save_per_image_report:
        for r in results:
            stem = Path(r.input_path).stem
            report_path = report_dir / f"{stem}.json"
            try:
                pipeline.save_report(r, report_path)
            except Exception as e:
                print(f"  ⚠ 개별 리포트 저장 실패 ({stem}): {e}")
    
    # 전체 JSON 요약 (원본 save_report로 List 처리)
    summary_json = output_dir / 'batch_summary.json'
    try:
        pipeline.save_report(results, summary_json)
    except Exception as e:
        print(f"⚠ 전체 JSON 요약 저장 실패: {e}")
    
    # 마크다운 요약
    summary_md = output_dir / 'batch_summary.md'
    try:
        write_markdown_summary(
            md_path=summary_md,
            results=results,
            batch_started_at=batch_started_at,
            batch_finished_at=batch_finished_at,
            batch_elapsed=batch_elapsed,
            input_dir=input_dir,
            output_dir=output_dir,
            comparison_paths=comparison_paths,
        )
    except Exception as e:
        print(f"⚠ 마크다운 요약 저장 실패: {e}")
    
    # ──────────────────────────────────────────────────────────
    # 최종 콘솔 요약
    # ──────────────────────────────────────────────────────────
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print()
    print("=" * 70)
    print("일괄 처리 완료")
    print("=" * 70)
    print(f"  총 처리      : {len(results)}장")
    print(f"  성공         : {len(succeeded)}장")
    print(f"  실패         : {len(failed)}장")
    print(f"  총 소요 시간 : {batch_elapsed:.1f}초")
    if succeeded:
        avg_time = sum(r.total_time for r in succeeded) / len(succeeded)
        print(f"  평균 처리시간: {avg_time:.1f}초/장")
    print(f"  복원 이미지  : {restored_dir}")
    if save_comparison:
        print(f"  비교 이미지  : {comparison_dir}")
    print(f"  요약 (JSON)  : {summary_json}")
    print(f"  요약 (MD)    : {summary_md}")
    
    if failed:
        print()
        print(f"  ⚠ 실패한 이미지 ({len(failed)}장):")
        for r in failed:
            filename = Path(r.input_path).name
            errors = '; '.join(r.errors) if r.errors else '알 수 없는 오류'
            print(f"    - {filename}: {errors}")
    print("=" * 70)
    
    return len(failed)


def save_comparison_safe(
    input_path: Path,
    restored_path: Path,
    comparison_dir: Path
) -> Optional[Path]:
    """save_comparison wrapper with safety check."""
    if not restored_path.exists():
        print(f"  ⚠ 복원 결과 파일이 없어 비교 이미지 생략: {restored_path}")
        return None
    return save_comparison(input_path, restored_path, comparison_dir)


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
    python batch_process.py --input-dir data/input --verbose
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
    failed_count = run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        device=args.device,
        limit=args.limit,
        save_comparison=not args.no_comparison,
        save_per_image_report=not args.no_report,
        verbose=args.verbose,
    )
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
