"""
test_ddcolor.py (디버그 강화 버전)
DDColor 단독 동작 테스트 - silent crash 추적 강화

기존 버전과의 차이:
    - 단계마다 print + sys.stdout.flush() (silent 종료 추적)
    - VRAM 모니터링 (각 단계 전후)
    - 모든 예외에 traceback 출력
    - faulthandler 활성화 (segfault 추적)
    - device='cpu' 옵션 권장 안내

사용법:
    python test_ddcolor.py --input data/input/UC11951204.jpg --output data/output/test_ddcolor.jpg
    python test_ddcolor.py --input data/input/UC11951204.jpg --output data/output/test_ddcolor_cpu.jpg --device cpu
"""

import argparse
import faulthandler
import gc
import os
import sys
import time
import traceback
from pathlib import Path

# segfault 등 silent crash 추적 활성화
faulthandler.enable()


def log(msg: str):
    """즉시 출력 (버퍼링 방지)"""
    print(msg, flush=True)


def vram_status(label: str = ""):
    """VRAM 상태 출력"""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            log(f"  [VRAM {label}] 사용: {used/1024**3:.2f}GB / {total/1024**3:.2f}GB (가용: {free/1024**3:.2f}GB)")
    except Exception:
        pass


def check_environment():
    """환경 확인"""
    log("=" * 60)
    log("환경 확인")
    log("=" * 60)
    
    try:
        import torch
        log(f"✓ PyTorch: {torch.__version__}")
        log(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  CUDA 디바이스: {torch.cuda.get_device_name(0)}")
            vram_status("초기")
    except ImportError:
        log("✗ PyTorch가 설치되지 않음")
        return False
    
    try:
        import modelscope
        log(f"✓ modelscope: {modelscope.__version__}")
    except ImportError:
        log("✗ modelscope가 설치되지 않음")
        return False
    
    try:
        import cv2
        log(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        log("✗ opencv-python이 설치되지 않음")
        return False
    
    log("")
    return True


def colorize_with_ddcolor(input_path: str, output_path: str, device: str = 'cuda'):
    """DDColor 컬러화 (디버그 강화)"""
    import cv2
    import numpy as np
    import torch
    
    log("=" * 60)
    log("[1/7] 입력 파일 확인")
    log("=" * 60)
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    log(f"✓ 입력 파일 존재: {input_path}")
    log(f"  크기: {os.path.getsize(input_path) / 1024:.1f} KB")
    log("")
    
    # CUDA 가용성 재확인
    if device == 'cuda':
        if not torch.cuda.is_available():
            log("⚠ CUDA 사용 불가 → CPU로 전환")
            device = 'cpu'
        else:
            vram_status("로딩 전")
    
    log("=" * 60)
    log(f"[2/7] modelscope 모듈 import (device={device})")
    log("=" * 60)
    
    try:
        from modelscope.outputs import OutputKeys
        log("  ✓ OutputKeys import")
        from modelscope.pipelines import pipeline
        log("  ✓ pipeline import")
        from modelscope.utils.constant import Tasks
        log("  ✓ Tasks import")
    except Exception as e:
        log(f"  ✗ modelscope import 실패: {e}")
        traceback.print_exc()
        raise
    log("")
    
    log("=" * 60)
    log("[3/7] DDColor 파이프라인 로딩 ★ silent crash 가능성 높음")
    log("=" * 60)
    log("(가중치는 캐시되어 있어야 함, 실제 로드 시간 1~5분)")
    
    t0 = time.time()
    
    try:
        img_colorization = pipeline(
            Tasks.image_colorization,
            model='damo/cv_ddcolor_image-colorization',
            device=device
        )
        t1 = time.time()
        log(f"  ✓ 파이프라인 로드 완료 ({t1 - t0:.1f}초)")
        if device == 'cuda':
            vram_status("로딩 후")
    except torch.cuda.OutOfMemoryError as e:
        log("")
        log("=" * 60)
        log("✗ VRAM 부족 (OOM)")
        log("=" * 60)
        log(f"  에러: {e}")
        log("  해결: --device cpu 옵션 사용")
        log(f"  명령: python test_ddcolor.py --input {input_path} --output {output_path} --device cpu")
        raise
    except Exception as e:
        log("")
        log("=" * 60)
        log(f"✗ 파이프라인 로드 실패: {type(e).__name__}")
        log("=" * 60)
        log(f"  에러: {e}")
        traceback.print_exc()
        raise
    log("")
    
    log("=" * 60)
    log("[4/7] 입력 이미지 디코딩 (한글 경로 대응)")
    log("=" * 60)
    
    try:
        img_data = np.fromfile(input_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("imdecode 실패")
        log(f"✓ 이미지 디코딩 완료")
        log(f"  shape: {img_bgr.shape}")
        log(f"  dtype: {img_bgr.dtype}")
    except Exception as e:
        log(f"✗ 이미지 디코딩 실패: {e}")
        traceback.print_exc()
        raise
    log("")
    
    log("=" * 60)
    log("[5/7] DDColor 추론")
    log("=" * 60)
    if device == 'cuda':
        vram_status("추론 전")
    
    t2 = time.time()
    try:
        result = img_colorization(img_bgr)
        output_bgr = result[OutputKeys.OUTPUT_IMG]
        t3 = time.time()
        log(f"✓ 추론 완료 ({t3 - t2:.1f}초)")
        log(f"  출력 shape: {output_bgr.shape}")
        if device == 'cuda':
            vram_status("추론 후")
    except torch.cuda.OutOfMemoryError as e:
        log(f"✗ 추론 중 VRAM OOM: {e}")
        log("  해결: --device cpu")
        raise
    except Exception as e:
        log(f"✗ 추론 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    log("")
    
    log("=" * 60)
    log("[6/7] 결과 저장")
    log("=" * 60)
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        ext = os.path.splitext(output_path)[1] or '.jpg'
        success, encoded = cv2.imencode(ext, output_bgr)
        if not success:
            raise RuntimeError("imencode 실패")
        encoded.tofile(output_path)
        log(f"✓ 저장 완료: {output_path}")
        log(f"  파일 크기: {os.path.getsize(output_path) / 1024:.1f} KB")
    except Exception as e:
        log(f"✗ 저장 실패: {e}")
        traceback.print_exc()
        raise
    log("")
    
    log("=" * 60)
    log("[7/7] 완료")
    log("=" * 60)
    log(f"✓ 총 소요 시간: {time.time() - t0:.1f}초")
    
    # 메모리 정리
    del img_colorization
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return output_bgr


def main():
    parser = argparse.ArgumentParser(description="DDColor 단독 동작 테스트 (디버그 강화)")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    log("")
    log("█" * 60)
    log("█ DDColor 디버그 테스트")
    log("█" * 60)
    log("")
    
    if not check_environment():
        sys.exit(1)
    
    try:
        colorize_with_ddcolor(args.input, args.output, args.device)
        log("")
        log("=" * 60)
        log("✓ DDColor 테스트 성공")
        log("=" * 60)
    except Exception as e:
        log("")
        log("=" * 60)
        log("✗ DDColor 테스트 실패")
        log("=" * 60)
        log(f"최종 에러: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
