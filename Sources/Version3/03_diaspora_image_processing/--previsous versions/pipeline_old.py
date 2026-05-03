"""
pipeline.py
통합 이미지 처리 파이프라인

모든 이미지 처리 모듈을 조건부로 실행하는 통합 파이프라인입니다.
입력 이미지의 상태를 분석하여 필요한 처리만 선택적으로 적용합니다.

처리 순서:
1. 손상 복원 (향후 구현)
2. 초해상도 변환 (Real-ESRGAN)
3. 얼굴 향상 (GFPGAN)
4. 흑백 컬러화 (DeOldify)
5. 이미지 분석 및 메타데이터 생성 (BLIP, CLIP, face_recognition)

Author: YongHwan Kim
Date: 2026
"""

import os
import yaml
import json
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image
import numpy as np

# 모듈 imports
from .modules.super_resolution import SuperResolutionModule
from .modules.face_enhancement import FaceEnhancementModule
from .modules.colorization import ColorizationModule
from .modules.image_analysis import ImageAnalysisModule


@dataclass
class ProcessingOptions:
    """처리 옵션 데이터 클래스"""
    
    # 초해상도 옵션
    enable_super_resolution: bool = True
    sr_scale: float = 4.0
    sr_min_resolution: int = 512
    sr_max_resolution: int = 2048
    
    # 얼굴 향상 옵션
    enable_face_enhancement: bool = True
    face_upscale: int = 2
    face_min_size: int = 64
    
    # 컬러화 옵션
    enable_colorization: bool = True
    colorize_model: str = 'artistic'
    colorize_render_factor: int = 35
    colorize_saturation_threshold: float = 0.1
    
    # 이미지 분석 옵션
    enable_analysis: bool = True
    enable_captioning: bool = True
    enable_classification: bool = True
    enable_face_detection: bool = True
    
    # 조건부 실행
    conditional_execution: bool = True
    
    # 중간 결과 저장
    save_intermediate: bool = False


@dataclass
class ProcessingResult:
    """처리 결과 데이터 클래스"""
    
    success: bool = True
    input_path: str = ""
    output_path: str = ""
    
    # 원본 정보
    original_size: tuple = (0, 0)
    final_size: tuple = (0, 0)
    
    # 각 단계 결과
    stages_applied: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    
    # 분석 결과
    caption: Optional[str] = None
    scenes: List[tuple] = field(default_factory=list)
    face_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 성능 정보
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # 에러
    errors: List[str] = field(default_factory=list)


class ImageRestorationPipeline:
    """
    통합 이미지 복원 파이프라인
    
    디아스포라 기록 유산 이미지를 자동으로 분석하고 복원합니다.
    각 처리 단계는 이미지 상태에 따라 조건부로 실행됩니다.
    
    Attributes:
        device (str): 연산 장치
        options (ProcessingOptions): 처리 옵션
        sr_module: 초해상도 모듈
        face_module: 얼굴 향상 모듈
        color_module: 컬러화 모듈
        analysis_module: 분석 모듈
        
    Example:
        >>> pipeline = ImageRestorationPipeline(device='cuda')
        >>> result = pipeline.process("old_photo.jpg", "restored_photo.jpg")
        >>> print(f"적용된 단계: {result.stages_applied}")
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        config_path: Optional[str] = None,
        options: Optional[ProcessingOptions] = None,
        lazy_load: bool = True
    ):
        """
        파이프라인 초기화
        
        Args:
            device: 연산 장치 ('cuda', 'cpu')
            config_path: 설정 파일 경로 (YAML)
            options: 처리 옵션 (config_path보다 우선)
            lazy_load: 지연 로딩 사용 여부
        """
        self.device = device
        self.lazy_load = lazy_load
        
        # 설정 로드
        if options:
            self.options = options
        elif config_path:
            self.options = self._load_config(config_path)
        else:
            self.options = ProcessingOptions()
        
        # 모듈 초기화 (지연 로딩 시 None으로 시작)
        self._sr_module = None
        self._face_module = None
        self._color_module = None
        self._analysis_module = None
        
        if not lazy_load:
            self._initialize_modules()
        
        print("=" * 50)
        print("이미지 복원 파이프라인 초기화 완료")
        print("=" * 50)
        print(f"장치: {device}")
        print(f"지연 로딩: {lazy_load}")
        print(f"활성화된 단계:")
        print(f"  - 초해상도: {self.options.enable_super_resolution}")
        print(f"  - 얼굴 향상: {self.options.enable_face_enhancement}")
        print(f"  - 컬러화: {self.options.enable_colorization}")
        print(f"  - 이미지 분석: {self.options.enable_analysis}")
        print("=" * 50)
    
    def _load_config(self, config_path: str) -> ProcessingOptions:
        """YAML 설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        options = ProcessingOptions()
        
        # 초해상도 설정
        if 'super_resolution' in config:
            sr = config['super_resolution']
            options.enable_super_resolution = sr.get('enabled', True)
            options.sr_scale = sr.get('scale', 4)
            if 'auto_trigger' in sr:
                options.sr_min_resolution = sr['auto_trigger'].get('min_resolution', 512)
                options.sr_max_resolution = sr['auto_trigger'].get('max_resolution', 2048)
        
        # 얼굴 향상 설정
        if 'face_enhancement' in config:
            face = config['face_enhancement']
            options.enable_face_enhancement = face.get('enabled', True)
            options.face_upscale = face.get('upscale', 2)
        
        # 컬러화 설정
        if 'colorization' in config:
            color = config['colorization']
            options.enable_colorization = color.get('enabled', True)
            options.colorize_model = color.get('model_type', 'artistic')
            options.colorize_render_factor = color.get('render_factor', 35)
        
        # 분석 설정
        if 'image_analysis' in config:
            analysis = config['image_analysis']
            if 'captioning' in analysis:
                options.enable_captioning = analysis['captioning'].get('enabled', True)
            if 'scene_classification' in analysis:
                options.enable_classification = analysis['scene_classification'].get('enabled', True)
            if 'face_detection' in analysis:
                options.enable_face_detection = analysis['face_detection'].get('enabled', True)
        
        # 파이프라인 설정
        if 'pipeline' in config:
            pipe = config['pipeline']
            options.conditional_execution = pipe.get('conditional_execution', True)
            options.save_intermediate = pipe.get('save_intermediate', False)
        
        return options
    
    def _initialize_modules(self):
        """모든 모듈 초기화"""
        if self.options.enable_super_resolution and self._sr_module is None:
            self._sr_module = SuperResolutionModule(device=self.device)
        
        if self.options.enable_face_enhancement and self._face_module is None:
            self._face_module = FaceEnhancementModule(device=self.device)
        
        if self.options.enable_colorization and self._color_module is None:
            self._color_module = ColorizationModule(device=self.device)
        
        if self.options.enable_analysis and self._analysis_module is None:
            self._analysis_module = ImageAnalysisModule(
                device=self.device,
                enable_captioning=self.options.enable_captioning,
                enable_classification=self.options.enable_classification,
                enable_face_analysis=self.options.enable_face_detection
            )
    
    @property
    def sr_module(self) -> Optional[SuperResolutionModule]:
        """초해상도 모듈 (지연 로딩)"""
        if self._sr_module is None and self.options.enable_super_resolution:
            self._sr_module = SuperResolutionModule(device=self.device)
        return self._sr_module
    
    @property
    def face_module(self) -> Optional[FaceEnhancementModule]:
        """얼굴 향상 모듈 (지연 로딩)"""
        if self._face_module is None and self.options.enable_face_enhancement:
            self._face_module = FaceEnhancementModule(device=self.device)
        return self._face_module
    
    @property
    def color_module(self) -> Optional[ColorizationModule]:
        """컬러화 모듈 (지연 로딩)"""
        if self._color_module is None and self.options.enable_colorization:
            self._color_module = ColorizationModule(device=self.device)
        return self._color_module
    
    @property
    def analysis_module(self) -> Optional[ImageAnalysisModule]:
        """분석 모듈 (지연 로딩)"""
        if self._analysis_module is None and self.options.enable_analysis:
            self._analysis_module = ImageAnalysisModule(device=self.device)
        return self._analysis_module
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        options: Optional[ProcessingOptions] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> ProcessingResult:
        """
        이미지 처리 수행
        
        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (None이면 자동 생성)
            options: 처리 옵션 (None이면 기본값 사용)
            progress_callback: 진행 상황 콜백 (stage_name, current, total)
            
        Returns:
            ProcessingResult: 처리 결과
        """
        start_time = time.time()
        options = options or self.options
        
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_restored{input_path.suffix}"
        output_path = Path(output_path)
        
        result = ProcessingResult(
            input_path=str(input_path),
            output_path=str(output_path)
        )
        
        try:
            # 이미지 로드
            current_image = Image.open(input_path)
            result.original_size = current_image.size
            
            total_stages = 4
            current_stage = 0
            
            # 중간 결과 저장 경로
            intermediate_dir = None
            if options.save_intermediate:
                intermediate_dir = output_path.parent / f"{output_path.stem}_stages"
                intermediate_dir.mkdir(exist_ok=True)
            
            # === 1단계: 초해상도 변환 ===
            current_stage += 1
            if progress_callback:
                progress_callback("super_resolution", current_stage, total_stages)
            
            if options.enable_super_resolution and self.sr_module:
                stage_start = time.time()
                
                should_process, reason = self.sr_module.should_process(
                    current_image,
                    min_resolution=options.sr_min_resolution,
                    max_resolution=options.sr_max_resolution
                )
                
                if should_process or not options.conditional_execution:
                    sr_result = self.sr_module.enhance(
                        current_image,
                        outscale=options.sr_scale
                    )
                    current_image = sr_result['enhanced']
                    result.stages_applied.append('super_resolution')
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "01_super_resolution.jpg")
                else:
                    result.stages_skipped.append(f'super_resolution: {reason}')
                
                result.stage_times['super_resolution'] = time.time() - stage_start
            
            # === 2단계: 얼굴 향상 ===
            current_stage += 1
            if progress_callback:
                progress_callback("face_enhancement", current_stage, total_stages)
            
            if options.enable_face_enhancement and self.face_module:
                stage_start = time.time()
                
                should_process, reason, num_faces = self.face_module.should_process(
                    current_image,
                    min_face_size=options.face_min_size
                )
                
                if should_process or not options.conditional_execution:
                    face_result = self.face_module.enhance(current_image)
                    current_image = face_result['enhanced']
                    result.stages_applied.append('face_enhancement')
                    result.face_count = face_result['num_faces']
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "02_face_enhancement.jpg")
                else:
                    result.stages_skipped.append(f'face_enhancement: {reason}')
                
                result.stage_times['face_enhancement'] = time.time() - stage_start
            
            # === 3단계: 흑백 컬러화 ===
            current_stage += 1
            if progress_callback:
                progress_callback("colorization", current_stage, total_stages)
            
            if options.enable_colorization and self.color_module:
                stage_start = time.time()
                
                should_process, reason = self.color_module.should_process(
                    current_image,
                    saturation_threshold=options.colorize_saturation_threshold
                )
                
                if should_process or not options.conditional_execution:
                    color_result = self.color_module.colorize(
                        current_image,
                        render_factor=options.colorize_render_factor
                    )
                    
                    if not color_result.get('skipped'):
                        current_image = color_result['colorized']
                        result.stages_applied.append('colorization')
                    else:
                        result.stages_skipped.append(f"colorization: {color_result.get('reason')}")
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "03_colorization.jpg")
                else:
                    result.stages_skipped.append(f'colorization: {reason}')
                
                result.stage_times['colorization'] = time.time() - stage_start
            
            # === 4단계: 이미지 분석 ===
            current_stage += 1
            if progress_callback:
                progress_callback("analysis", current_stage, total_stages)
            
            if options.enable_analysis and self.analysis_module:
                stage_start = time.time()
                
                analysis_result = self.analysis_module.analyze(current_image)
                
                result.caption = analysis_result.get('caption')
                result.scenes = analysis_result.get('scenes', [])
                if result.face_count == 0:
                    result.face_count = len(analysis_result.get('faces', []))
                result.metadata = analysis_result.get('metadata', {})
                result.stages_applied.append('analysis')
                
                result.stage_times['analysis'] = time.time() - stage_start
            
            # 최종 결과 저장
            result.final_size = current_image.size
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                current_image.save(output_path, quality=95)
            else:
                current_image.save(output_path)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            import traceback
            traceback.print_exc()
        
        result.total_time = time.time() - start_time
        return result
    
    def process_batch(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        options: Optional[ProcessingOptions] = None,
        progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None
    ) -> List[ProcessingResult]:
        """
        배치 이미지 처리
        
        Args:
            input_paths: 입력 이미지 경로 리스트
            output_dir: 출력 디렉토리
            options: 처리 옵션
            progress_callback: 진행 콜백 (current, total, result)
            
        Returns:
            처리 결과 리스트
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total = len(input_paths)
        
        for i, input_path in enumerate(input_paths):
            input_path = Path(input_path)
            output_path = output_dir / f"{input_path.stem}_restored{input_path.suffix}"
            
            result = self.process(input_path, output_path, options)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
        
        return results
    
    def save_report(
        self,
        results: Union[ProcessingResult, List[ProcessingResult]],
        output_path: Union[str, Path]
    ) -> None:
        """
        처리 결과 보고서 저장
        
        Args:
            results: 처리 결과 또는 결과 리스트
            output_path: 보고서 출력 경로 (JSON)
        """
        if isinstance(results, ProcessingResult):
            results = [results]
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_images': len(results),
            'successful': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success),
            'results': []
        }
        
        for r in results:
            report['results'].append({
                'input': r.input_path,
                'output': r.output_path,
                'success': r.success,
                'original_size': r.original_size,
                'final_size': r.final_size,
                'stages_applied': r.stages_applied,
                'stages_skipped': r.stages_skipped,
                'caption': r.caption,
                'scenes': r.scenes,
                'face_count': r.face_count,
                'metadata': r.metadata,
                'total_time': r.total_time,
                'stage_times': r.stage_times,
                'errors': r.errors
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================
# CLI 인터페이스
# ============================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI 기반 이미지 복원 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 이미지 처리
  python -m src.pipeline -i input.jpg -o output.jpg
  
  # 배치 처리
  python -m src.pipeline -i ./input_dir -o ./output_dir --batch
  
  # 설정 파일 사용
  python -m src.pipeline -i input.jpg -o output.jpg --config config.yaml
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="입력 이미지 또는 디렉토리 경로")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="출력 이미지 또는 디렉토리 경로")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="설정 파일 경로 (YAML)")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="연산 장치 (cuda/cpu)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="배치 처리 모드")
    parser.add_argument("--report", "-r", type=str, default=None,
                        help="처리 보고서 출력 경로 (JSON)")
    
    # 개별 단계 활성화/비활성화
    parser.add_argument("--no-sr", action="store_true",
                        help="초해상도 비활성화")
    parser.add_argument("--no-face", action="store_true",
                        help="얼굴 향상 비활성화")
    parser.add_argument("--no-color", action="store_true",
                        help="컬러화 비활성화")
    parser.add_argument("--no-analysis", action="store_true",
                        help="이미지 분석 비활성화")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="중간 결과 저장")
    
    args = parser.parse_args()
    
    # 옵션 구성
    options = ProcessingOptions(
        enable_super_resolution=not args.no_sr,
        enable_face_enhancement=not args.no_face,
        enable_colorization=not args.no_color,
        enable_analysis=not args.no_analysis,
        save_intermediate=args.save_intermediate
    )
    
    # 파이프라인 초기화
    pipeline = ImageRestorationPipeline(
        device=args.device,
        config_path=args.config,
        options=options if not args.config else None
    )
    
    # 진행 상황 콜백
    def progress_callback(stage: str, current: int, total: int):
        print(f"[{current}/{total}] {stage} 처리 중...")
    
    # 처리 수행
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 배치 처리
        if input_path.is_dir():
            input_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        else:
            input_paths = [input_path]
        
        def batch_progress(current, total, result):
            status = "✓" if result.success else "✗"
            print(f"[{current}/{total}] {status} {result.input_path}")
        
        results = pipeline.process_batch(
            input_paths,
            args.output,
            progress_callback=batch_progress
        )
        
        # 요약 출력
        success = sum(1 for r in results if r.success)
        print(f"\n처리 완료: {success}/{len(results)} 성공")
        
    else:
        # 단일 이미지 처리
        result = pipeline.process(
            args.input,
            args.output,
            progress_callback=progress_callback
        )
        
        # 결과 출력
        print("\n" + "=" * 50)
        print("처리 결과")
        print("=" * 50)
        print(f"성공: {result.success}")
        print(f"원본 크기: {result.original_size}")
        print(f"최종 크기: {result.final_size}")
        print(f"적용된 단계: {result.stages_applied}")
        print(f"건너뛴 단계: {result.stages_skipped}")
        print(f"캡션: {result.caption}")
        print(f"장면: {result.scenes[:3] if result.scenes else 'N/A'}")
        print(f"얼굴 수: {result.face_count}")
        print(f"총 처리 시간: {result.total_time:.2f}초")
        
        results = [result]
    
    # 보고서 저장
    if args.report:
        pipeline.save_report(results, args.report)
        print(f"\n보고서 저장: {args.report}")


if __name__ == "__main__":
    main()
