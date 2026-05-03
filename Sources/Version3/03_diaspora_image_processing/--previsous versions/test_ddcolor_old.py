"""
test_ddcolor.py
DDColor 단독 동작 테스트

목적:
    colorization.py를 수정하기 전에 DDColor가 현재 환경에서
    정상 동작하는지를 확인한다.

사용법:
    python test_ddcolor.py --input data/input/UC11950610.jpg --output data/output/UC11950610_ddcolor.jpg

성공 시:
    - 컬러화된 이미지가 output에 저장됨
    - 처리 시간, 가중치 위치 등 정보 출력

실패 시:
    - 에러 메시지를 보고 다음 단계 결정
"""

import argparse
import os
import sys
import time
from pathlib import Path


def check_environment():
    """환경 확인"""
    print("=" * 60)
    print("환경 확인")
    print("=" * 60)
    
    # PyTorch & CUDA
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 디바이스: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch가 설치되지 않음")
        return False
    
    # modelscope
    try:
        import modelscope
        print(f"✓ modelscope: {modelscope.__version__}")
    except ImportError:
        print("✗ modelscope가 설치되지 않음. pip install modelscope")
        return False
    
    # OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ opencv-python이 설치되지 않음")
        return False
    
    print()
    return True


def colorize_with_ddcolor(input_path: str, output_path: str, device: str = 'cuda'):
    """
    DDColor를 사용해 이미지 컬러화
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        device: 'cuda' or 'cpu'
    """
    import cv2
    import numpy as np
    import torch
    
    # 입력 파일 확인
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    
    print("=" * 60)
    print("DDColor 모델 로딩")
    print("=" * 60)
    print("(최초 실행 시 가중치 자동 다운로드 - 약 1.5GB, 시간 소요됨)")
    print()
    
    t0 = time.time()
    
    # modelscope를 통한 DDColor 로드
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    
    # cuda가 사용 불가하면 자동으로 cpu로 fallback
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA 사용 불가 → CPU로 전환")
        device = 'cpu'
    
    img_colorization = pipeline(
        Tasks.image_colorization,
        model='damo/cv_ddcolor_image-colorization',
        device=device
    )
    
    t1 = time.time()
    print(f"✓ 모델 로딩 완료 ({t1 - t0:.1f}초)")
    print()
    
    # 가중치 위치 확인 (modelscope 캐시)
    cache_dir = os.path.expanduser('~/.cache/modelscope/hub')
    if os.path.isdir(cache_dir):
        print(f"  modelscope 캐시 디렉토리: {cache_dir}")
    
    # ──────────────────────────────────────────
    # 한글 경로 대응 이미지 로드
    # ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("이미지 컬러화")
    print("=" * 60)
    print(f"입력: {input_path}")
    
    # 한글 경로 대응
    img_data = np.fromfile(input_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {input_path}")
    
    print(f"  원본 크기: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    
    # 추론
    t2 = time.time()
    result = img_colorization(img_bgr)
    output_bgr = result[OutputKeys.OUTPUT_IMG]
    t3 = time.time()
    
    print(f"✓ 컬러화 완료 ({t3 - t2:.1f}초)")
    print(f"  출력 크기: {output_bgr.shape[1]}x{output_bgr.shape[0]}")
    
    # ──────────────────────────────────────────
    # 한글 경로 대응 저장
    # ──────────────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    ext = os.path.splitext(output_path)[1] or '.jpg'
    success, encoded = cv2.imencode(ext, output_bgr)
    if not success:
        raise RuntimeError("이미지 인코딩 실패")
    encoded.tofile(output_path)
    
    print()
    print(f"✓ 저장 완료: {output_path}")
    print(f"  총 소요 시간: {t3 - t0:.1f}초")
    
    return output_bgr


def main():
    parser = argparse.ArgumentParser(description="DDColor 단독 동작 테스트")
    parser.add_argument('--input', '-i', required=True, help='입력 이미지 경로')
    parser.add_argument('--output', '-o', required=True, help='출력 이미지 경로')
    parser.add_argument('--device', '-d', default='cuda',
                        choices=['cuda', 'cpu'], help='연산 장치')
    args = parser.parse_args()
    
    # 환경 확인
    if not check_environment():
        sys.exit(1)
    
    # 컬러화 실행
    try:
        colorize_with_ddcolor(args.input, args.output, args.device)
        print()
        print("=" * 60)
        print("✓ DDColor 테스트 성공")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ DDColor 테스트 실패")
        print("=" * 60)
        print(f"에러: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
