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
    sr_min_resolution: int = 0       # ← 0으로 변경: 모든 이미지 SR 대상
    sr_max_resolution: int = 99999   # ← 99999: 상한 없음
    
    # 얼굴 향상 옵션
    enable_face_enhancement: bool = True
    face_upscale: int = 2
    face_min_size: int = 0           # ← 0으로 변경: 얼굴 크기 무관하게 실행
    
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
    
    # 조건부 실행 ← False로 변경: 조건 무시하고 항상 실행
    conditional_execution: bool = False
    
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
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        config_path: Optional[str] = None,
        options: Optional[ProcessingOptions] = None,
        lazy_load: bool = True
    ):
        self.device = device
        self.lazy_load = lazy_load
        
        # 설정 로드 (config보다 options 우선)
        if options:
            self.options = options
        elif config_path:
            self.options = self._load_config(config_path)
        else:
            self.options = ProcessingOptions()
        
        # 모듈 초기화
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
        
        if 'super_resolution' in config:
            sr = config['super_resolution']
            options.enable_super_resolution = sr.get('enabled', True)
            options.sr_scale = sr.get('scale', 4)
            # auto_trigger 무시 — 항상 실행
            options.sr_min_resolution = 0
            options.sr_max_resolution = 99999
        
        if 'face_enhancement' in config:
            face = config['face_enhancement']
            options.enable_face_enhancement = face.get('enabled', True)
            options.face_upscale = face.get('upscale', 2)
            options.face_min_size = 0   # 크기 조건 무시
        
        if 'colorization' in config:
            color = config['colorization']
            options.enable_colorization = color.get('enabled', True)
            options.colorize_model = color.get('model_type', 'artistic')
            options.colorize_render_factor = color.get('render_factor', 35)
        
        if 'image_analysis' in config:
            analysis = config['image_analysis']
            if 'captioning' in analysis:
                options.enable_captioning = analysis['captioning'].get('enabled', True)
            if 'scene_classification' in analysis:
                options.enable_classification = analysis['scene_classification'].get('enabled', True)
            if 'face_detection' in analysis:
                options.enable_face_detection = analysis['face_detection'].get('enabled', True)
        
        # pipeline.conditional_execution 무시 — 항상 False(강제 실행)
        options.conditional_execution = False
        
        if 'pipeline' in config:
            pipe = config['pipeline']
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
        if self._sr_module is None and self.options.enable_super_resolution:
            self._sr_module = SuperResolutionModule(device=self.device)
        return self._sr_module
    
    @property
    def face_module(self) -> Optional[FaceEnhancementModule]:
        if self._face_module is None and self.options.enable_face_enhancement:
            self._face_module = FaceEnhancementModule(device=self.device)
        return self._face_module
    
    @property
    def color_module(self) -> Optional[ColorizationModule]:
        if self._color_module is None and self.options.enable_colorization:
            self._color_module = ColorizationModule(device=self.device)
        return self._color_module
    
    @property
    def analysis_module(self) -> Optional[ImageAnalysisModule]:
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
            current_image = Image.open(input_path).convert('RGB')
            result.original_size = current_image.size
            
            total_stages = 4
            current_stage = 0
            
            intermediate_dir = None
            if options.save_intermediate:
                intermediate_dir = output_path.parent / f"{output_path.stem}_stages"
                intermediate_dir.mkdir(exist_ok=True)
            
            # =============================================
            # 1단계: 초해상도 변환 (항상 실행)
            # =============================================
            current_stage += 1
            if progress_callback:
                progress_callback("super_resolution", current_stage, total_stages)
            
            if options.enable_super_resolution and self.sr_module:
                stage_start = time.time()
                try:
                    print(f"  [1/4] 초해상도 변환 중... (원본: {current_image.size})")
                    sr_result = self.sr_module.enhance(
                        current_image,
                        outscale=options.sr_scale
                    )
                    current_image = sr_result['enhanced']
                    result.stages_applied.append('super_resolution')
                    print(f"  ✓ 초해상도 완료: {sr_result['original_size']} → {sr_result['enhanced_size']}")
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "01_super_resolution.jpg")
                except Exception as e:
                    print(f"  ✗ 초해상도 실패: {e}")
                    result.stages_skipped.append(f'super_resolution: 오류 - {e}')
                
                result.stage_times['super_resolution'] = time.time() - stage_start
            
            # =============================================
            # 2단계: 얼굴 향상 (항상 실행 — 감지 실패해도 시도)
            # =============================================
            current_stage += 1
            if progress_callback:
                progress_callback("face_enhancement", current_stage, total_stages)
            
            if options.enable_face_enhancement and self.face_module:
                stage_start = time.time()
                try:
                    print(f"  [2/4] 얼굴 향상 중...")
                    face_result = self.face_module.enhance(current_image)
                    current_image = face_result['enhanced']
                    result.stages_applied.append('face_enhancement')
                    result.face_count = face_result['num_faces']
                    print(f"  ✓ 얼굴 향상 완료 (감지된 얼굴: {face_result['num_faces']}개)")
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "02_face_enhancement.jpg")
                except Exception as e:
                    print(f"  ✗ 얼굴 향상 실패: {e}")
                    result.stages_skipped.append(f'face_enhancement: 오류 - {e}')
                
                result.stage_times['face_enhancement'] = time.time() - stage_start
            
            # =============================================
            # 3단계: 흑백 컬러화 (흑백 여부 판단 후 실행)
            # =============================================
            current_stage += 1
            if progress_callback:
                progress_callback("colorization", current_stage, total_stages)
            
            if options.enable_colorization and self.color_module:
                stage_start = time.time()
                try:
                    is_gray, gray_info = self.color_module.is_grayscale(
                        current_image,
                        saturation_threshold=options.colorize_saturation_threshold
                    )
                    print(f"  [3/4] 컬러화 중... (흑백 판정: {is_gray}, {gray_info.get('reason','')})")
                    
                    if is_gray or not options.conditional_execution:
                        color_result = self.color_module.colorize(
                            current_image,
                            render_factor=options.colorize_render_factor
                        )
                        current_image = color_result['colorized']
                        result.stages_applied.append('colorization')
                        print(f"  ✓ 컬러화 완료 (방법: {color_result.get('method','unknown')})")
                    else:
                        result.stages_skipped.append(f"colorization: 이미 컬러 이미지")
                        print(f"  - 컬러화 건너뜀: 이미 컬러 이미지")
                    
                    if intermediate_dir:
                        current_image.save(intermediate_dir / "03_colorization.jpg")
                except Exception as e:
                    print(f"  ✗ 컬러화 실패: {e}")
                    result.stages_skipped.append(f'colorization: 오류 - {e}')
                
                result.stage_times['colorization'] = time.time() - stage_start
            
            # =============================================
            # 4단계: 이미지 분석
            # =============================================
            current_stage += 1
            if progress_callback:
                progress_callback("analysis", current_stage, total_stages)
            
            if options.enable_analysis and self.analysis_module:
                stage_start = time.time()
                try:
                    print(f"  [4/4] 이미지 분석 중...")
                    analysis_result = self.analysis_module.analyze(current_image)
                    result.caption = analysis_result.get('caption')
                    result.scenes = analysis_result.get('scenes', [])
                    if result.face_count == 0:
                        result.face_count = len(analysis_result.get('faces', []))
                    result.metadata = analysis_result.get('metadata', {})
                    result.stages_applied.append('analysis')
                    print(f"  ✓ 분석 완료")
                except Exception as e:
                    print(f"  ✗ 분석 실패: {e}")
                    result.stages_skipped.append(f'analysis: 오류 - {e}')
                
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
        progress_callback: Optional[Callable[[int, int, 'ProcessingResult'], None]] = None
    ) -> List[ProcessingResult]:
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
    
    parser = argparse.ArgumentParser(description="AI 기반 이미지 복원 파이프라인")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--batch", "-b", action="store_true")
    parser.add_argument("--report", "-r", type=str, default=None)
    parser.add_argument("--no-sr", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")
    
    args = parser.parse_args()
    
    options = ProcessingOptions(
        enable_super_resolution=not args.no_sr,
        enable_face_enhancement=not args.no_face,
        enable_colorization=not args.no_color,
        enable_analysis=not args.no_analysis,
        save_intermediate=args.save_intermediate
    )
    
    pipeline = ImageRestorationPipeline(
        device=args.device,
        config_path=args.config,
        options=options if not args.config else None
    )
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        if input_path.is_dir():
            input_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        else:
            input_paths = [input_path]
        
        results = pipeline.process_batch(input_paths, args.output)
        success = sum(1 for r in results if r.success)
        print(f"\n처리 완료: {success}/{len(results)} 성공")
    else:
        result = pipeline.process(args.input, args.output)
        print(f"\n성공: {result.success}, 크기: {result.original_size} → {result.final_size}")
        print(f"적용: {result.stages_applied}")
    
    if args.report:
        pipeline.save_report(results if args.batch else [result], args.report)


if __name__ == "__main__":
    main()
