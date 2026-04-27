"""
evaluation
정량 평가 패키지

학위논문의 mandatory 이슈 (통계적 유효성, 공정한 baseline 비교, 
재현성 확보)에 대응하는 평가 코드 모음입니다.

주요 모듈:
    - degrade.py: 합성 열화 (GT → 저화질 입력)
    - metrics.py: PSNR/SSIM/LPIPS/BRISQUE 계산
    - baselines.py: Bicubic, Real-ESRGAN, 우리 통합 시스템
    - statistical_test.py: Wilcoxon, Bonferroni 보정
    - evaluate.py: 메인 평가 워크플로우
"""

__version__ = '1.0.0'
__author__ = 'YongHwan Kim'
