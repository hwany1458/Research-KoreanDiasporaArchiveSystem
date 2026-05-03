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
import gc
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image
import numpy as np

# torch는 VRAM 정리를 위해 사용. CUDA 미사용 환경 대응 위해 try/except.
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# 모듈 imports
from .modules.super_resolution import SuperResolutionModule
from .modules.face_enhancement import FaceEnhancementModule
from .modules.colorization import ColorizationModule
from .modules.image_analysis import ImageAnalysisModule


# =============================================================================
# VRAM 관리 유틸리티 (8GB GPU 환경 지원)
# =============================================================================
#
# v2 패치 (2026-05-02): CUDA 13.x 드라이버 + PyTorch cu124 빌드 환경에서
# torch.cuda.empty_cache() 자체가 RuntimeError(OOM)을 일으키는 사례를 방어하기
# 위해 모든 CUDA cleanup 호출을 try/except로 감싸 graceful degradation을
# 적용한다. cleanup 실패 시 처리는 계속 진행되며, 다음 단계의 lazy_load와
# Python gc.collect()가 메모리를 회수한다.
#
def _safe_cuda_cleanup(verbose: bool = False, label: str = "") -> bool:
    """
    CUDA 캐시 정리를 안전하게 수행한다.
    
    드라이버/CUDA 런타임 ABI mismatch 환경(예: NVIDIA 591.74 + CUDA 13.1
    드라이버에 PyTorch 2.4.1+cu124 사용)에서 torch.cuda.empty_cache() 자체가
    OOM RuntimeError를 발생시킬 수 있다. 이 함수는 그러한 실패를 무시하고
    파이프라인 진행을 보장한다.
    
    Args:
        verbose: True 시 실패 메시지 출력
        label: 디버그용 레이블
        
    Returns:
        cleanup 성공 여부
    """
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        return False
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        # CUDA 13 ABI mismatch 또는 이미 OOM 상태에서 cleanup 실패
        if verbose:
            sys.stdout.write(
                f"  [VRAM cleanup skipped at {label}: {type(e).__name__}: {str(e)[:80]}]\n"
            )
            sys.stdout.flush()
        return False
    except Exception as e:
        # 기타 예외도 무시 (드라이버 mismatch, 이미 손상된 컨텍스트 등)
        if verbose:
            sys.stdout.write(
                f"  [VRAM cleanup error at {label}: {type(e).__name__}]\n"
            )
            sys.stdout.flush()
        return False


def _safe_mem_info():
    """
    torch.cuda.mem_get_info()를 안전하게 호출한다.
    실패 시 None 반환.
    """
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        return None
    try:
        return torch.cuda.mem_get_info()
    except Exception:
        return None


def _release_vram(label: str = "", verbose: bool = False):
    """
    각 단계 사이의 VRAM/메모리 정리 함수.
    
    8GB VRAM GPU(예: RTX 4070 Laptop)에서 SR + 얼굴 복원 + 컬러화를
    동시 실행 시 누적 OOM을 방지하기 위해 사용한다.
    
    Args:
        label: 디버그 출력용 단계 레이블
        verbose: True 시 VRAM 사용량 출력
    """
    # Python GC: 참조가 끊긴 텐서를 즉시 회수
    gc.collect()
    
    # CUDA 캐시 정리 (실패해도 계속 진행)
    _safe_cuda_cleanup(verbose=verbose, label=label)
    
    if verbose:
        info = _safe_mem_info()
        if info is not None:
            free, total = info
            used = total - free
            sys.stdout.write(
                f"  [VRAM after {label}] "
                f"used={used/1024**3:.2f}GB / total={total/1024**3:.2f}GB "
                f"(free={free/1024**3:.2f}GB)\n"
            )
            sys.stdout.flush()


def _unload_module(pipeline_obj, attr_name: str, verbose: bool = False):
    """
    파이프라인이 보유하던 모듈을 메모리에서 완전히 언로드한다.
    
    각 처리 단계가 끝난 직후 해당 단계의 모델 가중치를 GPU/RAM에서 제거하여
    다음 단계의 모델 적재에 필요한 공간을 확보한다. 다음 사진 처리 시
    lazy_load=True 메커니즘에 의해 자동으로 다시 로드된다.
    
    Args:
        pipeline_obj: ImageRestorationPipeline 인스턴스
        attr_name: 언로드할 내부 모듈 속성명 (예: '_sr_module')
        verbose: True 시 언로드 메시지 출력
    """
    module = getattr(pipeline_obj, attr_name, None)
    if module is None:
        return
    
    # 모듈 내부의 PyTorch 모델을 명시적으로 CPU로 옮기고 삭제 시도.
    # 일부 모듈은 .model 또는 .pipeline 속성을 보유함.
    for inner_attr in ['model', 'pipeline', 'face_helper', 'gfpganer',
                       'upsampler', 'blip_model', 'clip_model']:
        inner = getattr(module, inner_attr, None)
        if inner is None:
            continue
        try:
            # PyTorch nn.Module인 경우
            if hasattr(inner, 'to'):
                inner.to('cpu')
        except Exception:
            pass
        try:
            del inner
        except Exception:
            pass
    
    # 모듈 자체 제거
    setattr(pipeline_obj, attr_name, None)
    del module
    
    # GC + CUDA 캐시 정리 (실패해도 계속 진행)
    gc.collect()
    _safe_cuda_cleanup(verbose=verbose, label=f"unload {attr_name}")
    
    if verbose:
        sys.stdout.write(f"  [unloaded {attr_name}]\n")
        sys.stdout.flush()
        # VRAM 사용량도 함께 출력 (학위논문 부록 측정용)
        info = _safe_mem_info()
        if info is not None:
            free, total = info
            used = total - free
            sys.stdout.write(
                f"  [VRAM after unload {attr_name}] "
                f"used={used/1024**3:.2f}GB / total={total/1024**3:.2f}GB "
                f"(free={free/1024**3:.2f}GB)\n"
            )
            sys.stdout.flush()


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
    
    # ===== 저사양 GPU(예: 8GB VRAM) 지원 옵션 =====
    # low_vram_mode: 각 단계 종료 후 모듈을 메모리에서 언로드하여
    #   다음 단계가 필요로 하는 VRAM을 확보. 8GB 이하 GPU에서 권장.
    #   단점: 매 사진마다 모듈 재로드 → 처리 시간 약 5~15초 증가.
    low_vram_mode: bool = False
    # vram_verbose: True 시 단계별 VRAM 사용량 stdout 출력
    vram_verbose: bool = False
    
    # sr_tile_size: Real-ESRGAN의 타일 처리 크기 (픽셀).
    #   None이면 자동: low_vram_mode=True → 128, low_vram_mode=False → 256
    #   직접 지정 시: 사용자 값 우선
    #   타일이 작을수록 VRAM 적게 사용하나 처리 시간 증가.
    sr_tile_size: Optional[int] = None
    
    # ===== 단계 순서 옵션 (ablation study 지원) =====
    # pipeline_order: SR / Face / Color 3단계의 실행 순서를 결정한다.
    #   Analysis는 항상 마지막에 실행되므로 이 리스트에 포함하지 않는다.
    #   허용되는 단계명: 'super_resolution' (또는 'sr'),
    #                   'face_enhancement' (또는 'face'),
    #                   'colorization' (또는 'color')
    #
    # 학술적으로 의미 있는 ablation 후보:
    #   ['super_resolution', 'face_enhancement', 'colorization']  (default)
    #     - 디테일 먼저 살리고 → 얼굴 복원 → 색상 추정
    #     - 단점: SR 후 입력이 커져 색상 추정 일관성 저하 가능
    #   ['colorization', 'face_enhancement', 'super_resolution']  (옵션 B)
    #     - 흑백 작은 입력에서 색상 추정 → 색이 있는 얼굴 복원 → 마지막 SR
    #     - 학계의 일반적 권장 순서, 색상 일관성 양호 예상
    #   ['face_enhancement', 'colorization', 'super_resolution']
    #     - 원본에서 얼굴 먼저 복원 후 컬러화, 마지막 SR
    #
    # 단계 이름은 모두 소문자, 쉼표 또는 공백으로 구분된 문자열로
    # CLI에서 --pipeline-order "color,face,sr" 형태로도 지정 가능.
    pipeline_order: List[str] = field(default_factory=lambda: [
        'super_resolution', 'face_enhancement', 'colorization'
    ])
    
    def get_normalized_pipeline_order(self) -> List[str]:
        """
        pipeline_order를 정규화된 형태(canonical name)로 반환한다.
        
        - 단계명 별칭 처리: 'sr' → 'super_resolution',
                           'face' → 'face_enhancement',
                           'color' → 'colorization'
        - 중복 제거 및 누락 검출
        - 잘못된 단계명에 대해 ValueError 발생
        
        Returns:
            ['super_resolution', 'face_enhancement', 'colorization'] 의 순열
        """
        canonical_map = {
            'sr': 'super_resolution',
            'super_resolution': 'super_resolution',
            'super-resolution': 'super_resolution',
            'face': 'face_enhancement',
            'face_enhancement': 'face_enhancement',
            'face-enhancement': 'face_enhancement',
            'color': 'colorization',
            'colorize': 'colorization',
            'colorization': 'colorization',
        }
        valid_set = {'super_resolution', 'face_enhancement', 'colorization'}
        
        normalized = []
        seen = set()
        for stage in self.pipeline_order:
            key = stage.strip().lower()
            if key not in canonical_map:
                raise ValueError(
                    f"알 수 없는 단계명: '{stage}'. "
                    f"허용: 'sr'/'super_resolution', 'face'/'face_enhancement', "
                    f"'color'/'colorization'"
                )
            canon = canonical_map[key]
            if canon in seen:
                raise ValueError(f"단계 '{canon}'이(가) 중복되었습니다.")
            seen.add(canon)
            normalized.append(canon)
        
        # 누락된 단계가 있으면 default 위치에 끼워 넣지 않고 그대로 반환.
        # (일부 단계만 사용하는 ablation도 enable_* 플래그로 가능)
        return normalized

    def get_effective_sr_tile_size(self) -> int:
        """
        실제 적용될 SR 타일 크기를 반환한다.
        
        우선순위:
        1. sr_tile_size가 명시적으로 설정되어 있으면 그 값 사용
        2. low_vram_mode=True 이면 128 (메모리 절약)
        3. 기본값 256 (속도와 메모리의 균형)
        """
        if self.sr_tile_size is not None:
            return self.sr_tile_size
        if self.low_vram_mode:
            return 128  # 8GB GPU 환경: 메모리 단편화 회피
        return 256  # 16GB+ GPU 환경: 기본값


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
            # 저사양 GPU 옵션
            options.low_vram_mode = pipe.get('low_vram_mode', False)
            options.vram_verbose = pipe.get('vram_verbose', False)
        
        return options
    
    def _initialize_modules(self):
        """모든 모듈 초기화"""
        if self.options.enable_super_resolution and self._sr_module is None:
            self._sr_module = self._create_sr_module()
        
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
    
    def _create_sr_module(self) -> SuperResolutionModule:
        """
        Real-ESRGAN 초해상도 모듈을 생성하고 동적 tile_size를 적용한다.
        
        low_vram_mode=True일 때는 메모리 단편화 회피를 위해 작은 타일(128)을,
        평소에는 기본 타일(256)을 사용한다.
        
        호환성: SuperResolutionModule이 tile_size 매개변수를 지원하는지 여부에
        따라 다음 순서로 적용을 시도한다:
        1. 생성자에 tile_size 매개변수 전달 시도
        2. 실패 시 기본 생성자 후 속성 직접 설정 (.tile_size)
        3. 그것도 실패 시 .upsampler.tile 속성 설정 (Real-ESRGAN 내부 객체)
        4. 모두 실패 시 경고만 출력하고 기본값으로 진행
        """
        effective_tile = self.options.get_effective_sr_tile_size()
        
        if self.options.vram_verbose:
            sys.stdout.write(
                f"  [SR module] tile_size={effective_tile} "
                f"(low_vram_mode={self.options.low_vram_mode})\n"
            )
            sys.stdout.flush()
        
        # 시도 1: 생성자에 tile_size 키워드 전달
        try:
            return SuperResolutionModule(device=self.device, tile_size=effective_tile)
        except TypeError:
            # 생성자가 tile_size 매개변수를 받지 않음
            pass
        except Exception:
            # 다른 예외도 무시하고 fallback
            pass
        
        # 시도 2: 기본 생성자 후 속성 설정
        sr = SuperResolutionModule(device=self.device)
        
        # 시도 2a: 인스턴스에 직접 tile_size 속성 설정
        try:
            if hasattr(sr, 'tile_size'):
                sr.tile_size = effective_tile
                return sr
        except Exception:
            pass
        
        # 시도 2b: Real-ESRGAN의 upsampler.tile 속성 (가장 흔한 구조)
        try:
            if hasattr(sr, 'upsampler') and hasattr(sr.upsampler, 'tile'):
                sr.upsampler.tile = effective_tile
                if self.options.vram_verbose:
                    sys.stdout.write(
                        f"  [SR module] tile_size 적용됨 (upsampler.tile = {effective_tile})\n"
                    )
                    sys.stdout.flush()
                return sr
        except Exception:
            pass
        
        # 시도 2c: tile 속성 (내부 명명 규칙이 다를 경우)
        try:
            if hasattr(sr, 'tile'):
                sr.tile = effective_tile
                return sr
        except Exception:
            pass
        
        # 모든 시도 실패: 경고 출력하고 기본값으로 진행
        if self.options.vram_verbose:
            sys.stdout.write(
                f"  [SR module] WARNING: tile_size={effective_tile} 적용 실패. "
                f"super_resolution.py의 기본값으로 진행.\n"
            )
            sys.stdout.flush()
        return sr
    
    @property
    def sr_module(self) -> Optional[SuperResolutionModule]:
        """초해상도 모듈 (지연 로딩, low_vram_mode 시 자동 작은 tile 적용)"""
        if self._sr_module is None and self.options.enable_super_resolution:
            self._sr_module = self._create_sr_module()
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
    
    # ──────────────────────────────────────────────────────────
    # 단계별 헬퍼 메서드 (process()의 dispatcher가 호출)
    # 각 헬퍼는 다음 약속을 따른다:
    #   - 입력: current_image (PIL.Image), result (ProcessingResult),
    #           options (ProcessingOptions), intermediate_dir (Path|None),
    #           stage_index (int, 1-based, 중간 파일명에 사용)
    #   - 출력: 갱신된 current_image (PIL.Image)를 반환
    #   - 부수효과: result.stages_applied / stages_skipped /
    #              stage_times / face_count 등을 갱신,
    #              intermediate_dir이 주어지면 중간 결과 저장,
    #              VRAM 정리는 호출자(dispatcher)가 담당.
    # ──────────────────────────────────────────────────────────
    def _run_stage_super_resolution(
        self, current_image, result, options, intermediate_dir, stage_index
    ):
        """SR 단계 실행 (입력: PIL.Image → 출력: PIL.Image)."""
        if not (options.enable_super_resolution and self.sr_module):
            return current_image
        
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
                current_image.save(
                    intermediate_dir / f"{stage_index:02d}_super_resolution.jpg"
                )
        else:
            result.stages_skipped.append(f'super_resolution: {reason}')
        
        result.stage_times['super_resolution'] = time.time() - stage_start
        return current_image
    
    def _run_stage_face_enhancement(
        self, current_image, result, options, intermediate_dir, stage_index
    ):
        """얼굴 향상 단계 실행."""
        if not (options.enable_face_enhancement and self.face_module):
            return current_image
        
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
                current_image.save(
                    intermediate_dir / f"{stage_index:02d}_face_enhancement.jpg"
                )
        else:
            result.stages_skipped.append(f'face_enhancement: {reason}')
        
        result.stage_times['face_enhancement'] = time.time() - stage_start
        return current_image
    
    def _run_stage_colorization(
        self, current_image, result, options, intermediate_dir, stage_index
    ):
        """컬러화 단계 실행."""
        if not (options.enable_colorization and self.color_module):
            return current_image
        
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
                result.stages_skipped.append(
                    f"colorization: {color_result.get('reason')}"
                )
            
            if intermediate_dir:
                current_image.save(
                    intermediate_dir / f"{stage_index:02d}_colorization.jpg"
                )
        else:
            result.stages_skipped.append(f'colorization: {reason}')
        
        result.stage_times['colorization'] = time.time() - stage_start
        return current_image
    
    def _run_stage_analysis(
        self, current_image, result, options, intermediate_dir, stage_index
    ):
        """이미지 분석 단계 실행 (항상 마지막)."""
        if not (options.enable_analysis and self.analysis_module):
            return current_image
        
        stage_start = time.time()
        analysis_result = self.analysis_module.analyze(current_image)
        
        result.caption = analysis_result.get('caption')
        result.scenes = analysis_result.get('scenes', [])
        if result.face_count == 0:
            result.face_count = len(analysis_result.get('faces', []))
        result.metadata = analysis_result.get('metadata', {})
        result.stages_applied.append('analysis')
        result.stage_times['analysis'] = time.time() - stage_start
        return current_image
    
    # 단계명 → (헬퍼 메서드, VRAM 정리 시 unload 대상 속성명)
    _STAGE_DISPATCH = {
        'super_resolution': ('_run_stage_super_resolution', '_sr_module'),
        'face_enhancement': ('_run_stage_face_enhancement', '_face_module'),
        'colorization':     ('_run_stage_colorization',     '_color_module'),
    }
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        options: Optional[ProcessingOptions] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> ProcessingResult:
        """
        이미지 처리 수행
        
        단계 실행 순서는 options.pipeline_order에 의해 결정된다.
        Analysis는 항상 마지막에 실행되며, pipeline_order에는 SR / Face /
        Color 3단계의 순열을 지정한다.
        
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
        
        # pipeline_order 검증 및 정규화
        try:
            order = options.get_normalized_pipeline_order()
        except ValueError as exc:
            result = ProcessingResult(
                input_path=str(input_path),
                output_path="",
                success=False
            )
            result.errors.append(f"pipeline_order 오류: {exc}")
            result.total_time = time.time() - start_time
            return result
        
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
            
            # 총 단계 수 = 사용자 지정 순열 + analysis (마지막 고정)
            total_stages = len(order) + (1 if options.enable_analysis else 0)
            current_stage = 0
            
            # 중간 결과 저장 경로
            intermediate_dir = None
            if options.save_intermediate:
                intermediate_dir = output_path.parent / f"{output_path.stem}_stages"
                intermediate_dir.mkdir(exist_ok=True)
                # 적용된 순서를 README로 남겨 ablation 분석 시 추적 가능
                try:
                    (intermediate_dir / "_pipeline_order.txt").write_text(
                        " → ".join(order) + "\n",
                        encoding="utf-8"
                    )
                except Exception:
                    pass  # 메타 정보는 저장 실패해도 본 처리는 진행
            
            # === pipeline_order 순서대로 실행 ===
            for stage_name in order:
                current_stage += 1
                if progress_callback:
                    progress_callback(stage_name, current_stage, total_stages)
                
                method_name, unload_attr = self._STAGE_DISPATCH[stage_name]
                stage_method = getattr(self, method_name)
                
                current_image = stage_method(
                    current_image, result, options,
                    intermediate_dir, current_stage
                )
                
                # VRAM 정리: low_vram_mode면 모듈 자체 언로드, 아니면 캐시만
                if options.low_vram_mode:
                    _unload_module(self, unload_attr, verbose=options.vram_verbose)
                else:
                    _release_vram(stage_name, verbose=options.vram_verbose)
            
            # === Analysis 단계 (항상 마지막) ===
            if options.enable_analysis:
                current_stage += 1
                if progress_callback:
                    progress_callback("analysis", current_stage, total_stages)
                
                current_image = self._run_stage_analysis(
                    current_image, result, options,
                    intermediate_dir, current_stage
                )
                
                # VRAM 정리: BLIP/CLIP 모델 해제
                if options.low_vram_mode:
                    _unload_module(self, '_analysis_module',
                                   verbose=options.vram_verbose)
                else:
                    _release_vram("analysis", verbose=options.vram_verbose)
            
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
    parser.add_argument("--low-vram", action="store_true",
                        help="저사양 GPU(8GB 이하) 모드: 단계별로 모듈을 언로드하여 OOM 방지")
    parser.add_argument("--vram-verbose", action="store_true",
                        help="단계별 VRAM 사용량 출력")
    parser.add_argument(
        "--pipeline-order", type=str, default=None,
        help=(
            "단계 실행 순서 (ablation 실험용). 쉼표로 구분된 단계명. "
            "단계: 'sr'/'super_resolution', 'face'/'face_enhancement', "
            "'color'/'colorization'. analysis는 항상 마지막에 실행됨. "
            "기본값: 'sr,face,color'. "
            "예시: --pipeline-order \"color,face,sr\" (옵션 B)"
        )
    )
    
    args = parser.parse_args()
    
    # pipeline_order 파싱 (CLI 문자열 → 리스트)
    if args.pipeline_order is not None:
        order_list = [
            s.strip() for s in args.pipeline_order.replace(';', ',').split(',')
            if s.strip()
        ]
    else:
        order_list = ['super_resolution', 'face_enhancement', 'colorization']
    
    # 옵션 구성
    options = ProcessingOptions(
        enable_super_resolution=not args.no_sr,
        enable_face_enhancement=not args.no_face,
        enable_colorization=not args.no_color,
        enable_analysis=not args.no_analysis,
        save_intermediate=args.save_intermediate,
        low_vram_mode=args.low_vram,
        vram_verbose=args.vram_verbose,
        pipeline_order=order_list
    )
    
    # pipeline_order 사전 검증 (잘못된 값이면 즉시 에러)
    try:
        normalized = options.get_normalized_pipeline_order()
        if args.pipeline_order is not None:
            print(f"  파이프라인 순서: {' → '.join(normalized)}"
                  + (" → analysis" if not args.no_analysis else ""))
    except ValueError as exc:
        parser.error(f"--pipeline-order 오류: {exc}")
    
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
