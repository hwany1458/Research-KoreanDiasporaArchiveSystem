"""
evaluate.py
정량 평가 메인 스크립트

전체 평가 워크플로우를 자동화합니다:
    1. GT 이미지에 합성 열화 적용 → 열화된 입력 생성
    2. 3가지 방법으로 복원: Bicubic / Real-ESRGAN / 우리 통합 시스템
    3. 각 방법의 결과를 GT와 비교하여 PSNR/SSIM/LPIPS 산출
    4. 통계적 유의성 검정 (Wilcoxon signed-rank)
    5. (선택) 실제 디아스포라 사진에 대한 No-reference 평가
    6. 학위논문 자료로 사용 가능한 JSON/MD 리포트 생성

산출물:
    data/eval/
    ├── gt/                     ← 사용자가 미리 준비한 GT 이미지
    ├── degraded/               ← 자동 생성: 열화된 입력
    ├── restored_bicubic/       ← Bicubic 복원
    ├── restored_realesrgan/    ← Real-ESRGAN 복원
    ├── restored_ours/          ← 우리 통합 시스템 복원
    └── reports/
        ├── eval_report_<ts>.json
        └── eval_report_<ts>.md

사용법:
    python -m evaluation.evaluate --gt-dir data/eval/gt
    python -m evaluation.evaluate --gt-dir data/eval/gt --skip-degrade  # 이미 열화된 경우
    python -m evaluation.evaluate --gt-dir data/eval/gt --no-reference data/input  # 실제 사진도 평가
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트를 import 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 같은 패키지의 모듈
from evaluation.degrade import SyntheticDegradation
from evaluation.metrics import MetricsEvaluator, _imread_unicode
from evaluation.baselines import (
    BicubicBaseline, RealESRGANBaseline, FullPipelineBaseline
)
from evaluation.statistical_test import compare_methods


# ──────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────
def collect_images(directory: Path, extensions=None) -> List[Path]:
    """디렉토리에서 이미지 파일 수집 (정렬됨)."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if not directory.is_dir():
        return []
    return sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])


def find_pair(gt_path: Path, restored_dir: Path,
              suffix_hint: str = '') -> Optional[Path]:
    """
    GT 파일에 대응되는 복원 결과 파일 찾기.
    
    GT가 'photo01.png'이면 restored_dir에서 'photo01_*.jpg' 류를 찾음.
    """
    stem = gt_path.stem
    # 정확 일치 우선
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        candidate = restored_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # 접두사 매칭
    for p in restored_dir.iterdir():
        if p.is_file() and p.stem.startswith(stem):
            return p
    return None


# ──────────────────────────────────────────────────────────────────
# 평가 워크플로우
# ──────────────────────────────────────────────────────────────────
class EvaluationPipeline:
    """전체 평가 워크플로우 오케스트레이터."""
    
    def __init__(
        self,
        gt_dir: Path,
        eval_root: Path,
        device: str = 'cuda',
        seed: int = 42,
        verbose: bool = False
    ):
        self.gt_dir = gt_dir
        self.eval_root = eval_root
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        # 출력 하위 디렉토리
        self.degraded_dir = eval_root / 'degraded'
        self.bicubic_dir = eval_root / 'restored_bicubic'
        self.realesrgan_dir = eval_root / 'restored_realesrgan'
        self.ours_dir = eval_root / 'restored_ours'
        self.report_dir = eval_root / 'reports'
        
        for d in [self.degraded_dir, self.bicubic_dir, self.realesrgan_dir,
                  self.ours_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────────────
    # Step 1: 합성 열화
    # ─────────────────────────────────────────────
    def step_degrade(self) -> Dict:
        print("\n" + "=" * 70)
        print("[Step 1/5] 합성 열화 적용 (GT → 열화된 입력)")
        print("=" * 70)
        
        degrader = SyntheticDegradation(seed=self.seed)
        result = degrader.degrade_directory(
            input_dir=self.gt_dir,
            output_dir=self.degraded_dir
        )
        print(f"  완료: {result['succeeded']}/{result['total']}장")
        return result
    
    # ─────────────────────────────────────────────
    # Step 2-4: 각 baseline으로 복원
    # ─────────────────────────────────────────────
    def step_baseline_bicubic(self) -> Dict:
        print("\n" + "=" * 70)
        print("[Step 2/5] Baseline: Bicubic interpolation 적용")
        print("=" * 70)
        
        baseline = BicubicBaseline(scale=4)
        result = baseline.upscale_directory(
            input_dir=self.degraded_dir,
            output_dir=self.bicubic_dir
        )
        print(f"  완료: {result['succeeded']}/{result['total']}장")
        return result
    
    def step_baseline_realesrgan(self) -> Dict:
        print("\n" + "=" * 70)
        print("[Step 3/5] Baseline: Real-ESRGAN 단독 적용")
        print("=" * 70)
        
        baseline = RealESRGANBaseline(device=self.device)
        result = baseline.upscale_directory(
            input_dir=self.degraded_dir,
            output_dir=self.realesrgan_dir
        )
        print(f"  완료: {result['succeeded']}/{result['total']}장")
        return result
    
    def step_full_pipeline(self) -> Dict:
        print("\n" + "=" * 70)
        print("[Step 4/5] 우리 통합 시스템 적용")
        print("=" * 70)
        
        baseline = FullPipelineBaseline(device=self.device)
        result = baseline.process_directory(
            input_dir=self.degraded_dir,
            output_dir=self.ours_dir
        )
        print(f"  완료: {result['succeeded']}/{result['total']}장")
        return result
    
    # ─────────────────────────────────────────────
    # Step 5: 정량 지표 계산 + 통계 검정
    # ─────────────────────────────────────────────
    def step_metrics(self) -> Dict:
        print("\n" + "=" * 70)
        print("[Step 5/5] 정량 지표 계산 및 통계 검정")
        print("=" * 70)
        
        # MetricsEvaluator 초기화 (LPIPS 모델 로딩)
        evaluator = MetricsEvaluator(
            device=self.device,
            enable_lpips=True,
            enable_brisque=False  # full-reference 모드에선 불필요
        )
        
        # GT 이미지 수집
        gt_images = collect_images(self.gt_dir)
        if not gt_images:
            print("  ⚠ GT 이미지가 없습니다.")
            return {}
        
        # 각 방법별 결과 수집
        method_dirs = {
            'Bicubic': self.bicubic_dir,
            'Real-ESRGAN': self.realesrgan_dir,
            'Ours': self.ours_dir
        }
        
        # {method_name: {metric: [values...]}}
        method_metrics: Dict[str, Dict[str, List[float]]] = {
            name: {'psnr': [], 'ssim': [], 'lpips': []}
            for name in method_dirs.keys()
        }
        
        # 각 GT에 대해 모든 방법의 결과를 평가
        per_image_records = []
        
        for i, gt_path in enumerate(gt_images, start=1):
            print(f"\n  [{i}/{len(gt_images)}] {gt_path.name}")
            
            record = {'gt_filename': gt_path.name, 'methods': {}}
            
            for method_name, method_dir in method_dirs.items():
                pred_path = find_pair(gt_path, method_dir)
                if pred_path is None:
                    print(f"    ⚠ {method_name}: 결과 파일 없음")
                    continue
                
                metrics = evaluator.evaluate_pair(gt_path, pred_path)
                
                if 'error' in metrics:
                    print(f"    ⚠ {method_name}: {metrics['error']}")
                    continue
                
                # 값 누적
                for metric_key in ['psnr', 'ssim', 'lpips']:
                    if metric_key in metrics:
                        method_metrics[method_name][metric_key].append(
                            metrics[metric_key]
                        )
                
                # 출력
                psnr = metrics.get('psnr', float('nan'))
                ssim = metrics.get('ssim', float('nan'))
                lpips = metrics.get('lpips', float('nan'))
                print(f"    {method_name:<14} PSNR={psnr:6.2f}  "
                      f"SSIM={ssim:.4f}  LPIPS={lpips:.4f}")
                
                record['methods'][method_name] = {
                    'pred_filename': pred_path.name,
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpips
                }
            
            per_image_records.append(record)
        
        # 통계 검정: 각 지표별로
        print("\n" + "─" * 70)
        print("통계 검정 (Wilcoxon signed-rank, vs Bicubic)")
        print("─" * 70)
        
        statistical_results = {}
        
        for metric_name, higher_is_better in [
            ('psnr', True),
            ('ssim', True),
            ('lpips', False)
        ]:
            method_results_for_metric = {
                name: data[metric_name]
                for name, data in method_metrics.items()
                if data[metric_name]
            }
            
            if not method_results_for_metric:
                continue
            
            comp = compare_methods(
                method_results=method_results_for_metric,
                metric_name=metric_name.upper(),
                reference_method='Bicubic',
                higher_is_better=higher_is_better
            )
            print(comp['summary_table'])
            statistical_results[metric_name] = comp
        
        return {
            'per_image_records': per_image_records,
            'method_metrics': method_metrics,
            'statistical_results': statistical_results
        }
    
    # ─────────────────────────────────────────────
    # No-reference 평가 (선택, GT 없는 실제 사진용)
    # ─────────────────────────────────────────────
    def step_no_reference(
        self,
        nr_input_dir: Path,
        nr_output_subdir: Path
    ) -> Dict:
        """실제 디아스포라 사진에 대한 No-reference 평가."""
        print("\n" + "=" * 70)
        print("[보너스] No-reference 평가 (실제 디아스포라 사진)")
        print("=" * 70)
        
        if not nr_input_dir.is_dir():
            print(f"  ⚠ No-reference 입력 디렉토리 없음: {nr_input_dir}")
            return {}
        
        # 우리 시스템으로 복원
        nr_output_subdir.mkdir(parents=True, exist_ok=True)
        full_pipeline = FullPipelineBaseline(device=self.device)
        full_pipeline.process_directory(
            input_dir=nr_input_dir,
            output_dir=nr_output_subdir
        )
        
        # NR 평가기
        evaluator = MetricsEvaluator(
            device=self.device,
            enable_lpips=False,
            enable_brisque=True
        )
        
        if evaluator.nr_metrics is None:
            print("  ⚠ No-reference 지표 미사용 가능")
            return {}
        
        nr_records = []
        original_brisques = []
        restored_brisques = []
        
        original_images = collect_images(nr_input_dir)
        
        for orig_path in original_images:
            # 원본 BRISQUE
            orig_metrics = evaluator.evaluate_no_reference(orig_path)
            # 복원 결과 BRISQUE
            restored_path = find_pair(orig_path, nr_output_subdir)
            
            record = {
                'filename': orig_path.name,
                'original': orig_metrics
            }
            
            if restored_path is not None:
                restored_metrics = evaluator.evaluate_no_reference(restored_path)
                record['restored'] = restored_metrics
                
                if 'brisque' in orig_metrics and 'brisque' in restored_metrics:
                    original_brisques.append(orig_metrics['brisque'])
                    restored_brisques.append(restored_metrics['brisque'])
            
            nr_records.append(record)
        
        # 통계
        if original_brisques and restored_brisques:
            comp = compare_methods(
                method_results={
                    'Original': original_brisques,
                    'Restored (Ours)': restored_brisques
                },
                metric_name='BRISQUE',
                reference_method='Original',
                higher_is_better=False  # BRISQUE는 낮을수록 좋음
            )
            print(comp['summary_table'])
            return {
                'nr_records': nr_records,
                'statistical_results': {'brisque': comp}
            }
        
        return {'nr_records': nr_records}
    
    # ─────────────────────────────────────────────
    # 리포트 생성
    # ─────────────────────────────────────────────
    def write_reports(
        self,
        all_results: Dict,
        timestamp: str
    ) -> Dict[str, Path]:
        """JSON 및 마크다운 리포트 작성."""
        # JSON 리포트
        json_path = self.report_dir / f"eval_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2,
                     default=str)  # numpy 등 직렬화
        
        # 마크다운 리포트
        md_path = self.report_dir / f"eval_report_{timestamp}.md"
        self._write_markdown(md_path, all_results, timestamp)
        
        return {'json': json_path, 'md': md_path}
    
    def _write_markdown(self, path: Path, results: Dict, timestamp: str):
        """학위논문 자료로 활용 가능한 마크다운 리포트."""
        lines = []
        lines.append(f"# 정량 평가 리포트")
        lines.append("")
        lines.append(f"- **평가 일시**: {timestamp}")
        lines.append(f"- **GT 디렉토리**: `{self.gt_dir}`")
        lines.append(f"- **평가 루트**: `{self.eval_root}`")
        lines.append(f"- **장치**: {self.device}")
        lines.append(f"- **시드**: {self.seed}")
        lines.append("")
        
        # 1. 워크플로우 요약
        lines.append("## 1. 평가 워크플로우")
        lines.append("")
        lines.append("```")
        lines.append("GT 컬러 이미지")
        lines.append("    ↓ 합성 열화 (downscale 4x + blur + noise + grayscale + JPEG)")
        lines.append("열화된 입력 (저해상도 흑백)")
        lines.append("    ├─→ Bicubic interpolation (Baseline 1)")
        lines.append("    ├─→ Real-ESRGAN 단독 (Baseline 2)")
        lines.append("    └─→ 우리 통합 시스템 (Ours)")
        lines.append("                  ↓")
        lines.append("            GT와 비교하여 PSNR/SSIM/LPIPS 산출")
        lines.append("                  ↓")
        lines.append("            Wilcoxon signed-rank test (vs Bicubic)")
        lines.append("```")
        lines.append("")
        
        # 2. 정량 지표 결과
        if 'metrics' in results and 'statistical_results' in results['metrics']:
            stat_results = results['metrics']['statistical_results']
            
            lines.append("## 2. 정량 지표 결과 (Full-Reference)")
            lines.append("")
            
            for metric_name in ['psnr', 'ssim', 'lpips']:
                if metric_name not in stat_results:
                    continue
                
                comp = stat_results[metric_name]
                arrow = '↑' if comp['higher_is_better'] else '↓'
                lines.append(f"### 2.{['psnr', 'ssim', 'lpips'].index(metric_name) + 1} "
                           f"{metric_name.upper()} ({arrow})")
                lines.append("")
                lines.append("| Method | n | Mean | Std | Median | p (vs Bicubic) | Sig. |")
                lines.append("|--------|---|------|-----|--------|----------------|------|")
                
                ref_method = comp['reference_method']
                
                for name, desc in comp['descriptive'].items():
                    if name == ref_method:
                        p_str = "(reference)"
                        sig = "—"
                    else:
                        test_key = f"{name}_vs_{ref_method}"
                        if test_key in comp['pairwise_tests']:
                            t = comp['pairwise_tests'][test_key]
                            effective_p = t.get('bonferroni_p') or t.get('p_value')
                            p_str = f"{effective_p:.4f}" if effective_p else "N/A"
                            
                            p = t.get('p_value', 1.0)
                            if p < 0.001:
                                sig = "***"
                            elif p < 0.01:
                                sig = "**"
                            elif p < 0.05:
                                sig = "*"
                            else:
                                sig = "n.s."
                        else:
                            p_str = "N/A"
                            sig = "—"
                    
                    best_marker = " ★" if name == comp.get('best_method') else ""
                    lines.append(
                        f"| {name}{best_marker} | {desc['n']} | "
                        f"{desc['mean']:.4f} | {desc['std']:.4f} | "
                        f"{desc['median']:.4f} | {p_str} | {sig} |"
                    )
                lines.append("")
            
            lines.append("**유의수준**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` not significant")
            lines.append("")
            if any(stat_results[m].get('bonferroni_n', 1) > 1
                   for m in stat_results):
                lines.append("Bonferroni 다중 검정 보정 적용.")
                lines.append("")
        
        # 3. No-reference 평가 (있는 경우)
        if 'no_reference' in results and results['no_reference']:
            nr = results['no_reference']
            lines.append("## 3. No-Reference 평가 (실제 디아스포라 사진)")
            lines.append("")
            
            if 'statistical_results' in nr:
                for metric_name, comp in nr['statistical_results'].items():
                    arrow = '↑' if comp['higher_is_better'] else '↓'
                    lines.append(f"### {metric_name.upper()} ({arrow})")
                    lines.append("")
                    lines.append("| Method | n | Mean | Std | Median |")
                    lines.append("|--------|---|------|-----|--------|")
                    for name, desc in comp['descriptive'].items():
                        lines.append(
                            f"| {name} | {desc['n']} | "
                            f"{desc['mean']:.4f} | {desc['std']:.4f} | "
                            f"{desc['median']:.4f} |"
                        )
                    lines.append("")
        
        # 4. 학위논문 활용 가이드
        lines.append("## 4. 학위논문 활용 가이드")
        lines.append("")
        lines.append("- **본 결과의 권장 배치**: 제5장 평가 챕터 / 정량 평가 절")
        lines.append("- **mandatory 이슈 대응**:")
        lines.append("  - *통계적 유효성*: Wilcoxon signed-rank test 적용")
        lines.append("  - *공정한 baseline 비교*: Bicubic / Real-ESRGAN 단독 / 통합 시스템 점진적 비교")
        lines.append("  - *데이터셋 대표성*: GT는 일반 데이터셋 + 디아스포라 컬러사진 혼합 권장")
        lines.append("- **표본 크기**: 메모리 권장값 n≥30. 현재 n이 부족하면 GT 추가 권장.")
        lines.append("")
        
        path.write_text('\n'.join(lines), encoding='utf-8')
    
    # ─────────────────────────────────────────────
    # 메인 진입점
    # ─────────────────────────────────────────────
    def run(
        self,
        skip_degrade: bool = False,
        skip_baselines: bool = False,
        nr_input_dir: Optional[Path] = None
    ) -> Dict:
        all_results = {
            'started_at': datetime.now().isoformat(),
            'gt_dir': str(self.gt_dir),
            'eval_root': str(self.eval_root),
            'device': self.device,
            'seed': self.seed
        }
        
        try:
            if not skip_degrade:
                all_results['degrade'] = self.step_degrade()
            
            if not skip_baselines:
                all_results['bicubic'] = self.step_baseline_bicubic()
                all_results['realesrgan'] = self.step_baseline_realesrgan()
                all_results['ours'] = self.step_full_pipeline()
            
            all_results['metrics'] = self.step_metrics()
            
            # No-reference (선택)
            if nr_input_dir is not None:
                nr_output = self.eval_root / 'restored_ours_no_ref'
                all_results['no_reference'] = self.step_no_reference(
                    nr_input_dir, nr_output
                )
        except Exception as e:
            all_results['error'] = str(e)
            all_results['traceback'] = traceback.format_exc()
            print(f"\n✗ 평가 중단: {e}")
            traceback.print_exc()
        
        all_results['finished_at'] = datetime.now().isoformat()
        
        # 리포트 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_paths = self.write_reports(all_results, timestamp)
        
        print("\n" + "=" * 70)
        print("평가 완료")
        print("=" * 70)
        print(f"  JSON 리포트: {report_paths['json']}")
        print(f"  MD 리포트  : {report_paths['md']}")
        print("=" * 70)
        
        return all_results


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="이미지 처리 모듈 정량 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 1. data/eval/gt/ 폴더에 GT 이미지(컬러 고화질)를 두고
    python -m evaluation.evaluate --gt-dir data/eval/gt
    
    # 2. 이미 합성 열화/baseline 처리가 끝나서 지표만 다시 계산하려면
    python -m evaluation.evaluate --gt-dir data/eval/gt --skip-degrade --skip-baselines
    
    # 3. 실제 디아스포라 사진(GT 없음)에 대한 No-reference 평가도 함께
    python -m evaluation.evaluate --gt-dir data/eval/gt --no-reference data/input
"""
    )
    parser.add_argument('--gt-dir', type=str, default='data/eval/gt',
                        help='GT 이미지 디렉토리')
    parser.add_argument('--eval-root', type=str, default='data/eval',
                        help='평가 산출물 루트 디렉토리')
    parser.add_argument('--no-reference', type=str, default=None,
                        help='No-reference 평가용 실제 디아스포라 사진 디렉토리 (선택)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='재현성 시드')
    parser.add_argument('--skip-degrade', action='store_true',
                        help='합성 열화 단계 스킵 (이미 열화된 경우)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Baseline 처리 단계 스킵')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    gt_dir = (PROJECT_ROOT / args.gt_dir).resolve() \
        if not Path(args.gt_dir).is_absolute() else Path(args.gt_dir)
    eval_root = (PROJECT_ROOT / args.eval_root).resolve() \
        if not Path(args.eval_root).is_absolute() else Path(args.eval_root)
    
    nr_input_dir = None
    if args.no_reference:
        nr_input_dir = (
            (PROJECT_ROOT / args.no_reference).resolve()
            if not Path(args.no_reference).is_absolute()
            else Path(args.no_reference)
        )
    
    if not gt_dir.is_dir():
        print(f"✗ GT 디렉토리가 없습니다: {gt_dir}")
        print(f"  GT 이미지(컬러 고화질)를 해당 디렉토리에 두고 다시 실행하세요.")
        sys.exit(1)
    
    pipeline = EvaluationPipeline(
        gt_dir=gt_dir,
        eval_root=eval_root,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose
    )
    
    pipeline.run(
        skip_degrade=args.skip_degrade,
        skip_baselines=args.skip_baselines,
        nr_input_dir=nr_input_dir
    )


if __name__ == "__main__":
    main()
