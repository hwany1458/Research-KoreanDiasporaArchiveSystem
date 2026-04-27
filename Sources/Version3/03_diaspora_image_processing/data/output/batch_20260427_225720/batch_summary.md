# 일괄 이미지 복원 처리 결과

- **시작 시각**: 2026-04-27T22:57:24.931792
- **종료 시각**: 2026-04-27T22:58:36.382343
- **입력 디렉토리**: `C:\Users\user\Downloads\YouKyoung\diaspora_thesis_code\03_diaspora_image_processing\data\input`
- **출력 디렉토리**: `C:\Users\user\Downloads\YouKyoung\diaspora_thesis_code\03_diaspora_image_processing\data\output\batch_20260427_225720`
- **총 처리**: 3장
- **성공/실패**: 3 / 0
- **총 소요 시간**: 71.5초
- **평균 처리 시간**: 23.8초/장

## 처리 결과

| # | 파일명 | 상태 | 원본 크기 | 최종 크기 | 적용 단계 | 건너뜀 | 시간(초) |
|---|--------|------|-----------|-----------|-----------|--------|----------|
| 1 | `family01.jpg` | ✓ | 2678×2008 | 2678×2008 | face_enhancement, colorization, analysis | super_resolution: 메모리 안전 임계값 초과 (입력 2678x2008 = 5.4MP, 4배 후 86.0MP > 24.0MP) | 25.4 |
| 2 | `KADA-Edwinlee091.jpg` | ✓ | 1200×844 | 4800×3376 | super_resolution, face_enhancement, colorization, analysis | - | 26.3 |
| 3 | `UC11906884.jpg` | ✓ | 1200×845 | 4800×3380 | super_resolution, face_enhancement, colorization, analysis | - | 19.7 |

## 단계별 평균 처리 시간

| 단계 | 평균 시간 (초) | 적용 횟수 |
|------|----------------|-----------|
| super_resolution | 3.29 | 3 |
| face_enhancement | 9.33 | 3 |
| colorization | 0.79 | 3 |
| analysis | 3.97 | 3 |

## 이미지 분석 결과

| # | 파일명 | 캡션 | 주요 장면 | 얼굴 수 |
|---|--------|------|-----------|---------|
| 1 | `family01.jpg` | there is a man and a woman standing on a beach next to the ocean | family trip (0.37) | 3 |
| 2 | `KADA-Edwinlee091.jpg` | this is a group of people posing for a picture in a restaurant | family meal (0.37) | 14 |
| 3 | `UC11906884.jpg` | there is a group of people holding up signs at a protest | street scene (0.48) | 5 |

## 비교 이미지

학위논문 figure로 활용 가능한 원본/복원 비교 이미지:

- `family01.jpg` → [comparisons\family01_comparison.jpg](comparisons\family01_comparison.jpg)
- `KADA-Edwinlee091.jpg` → [comparisons\KADA-Edwinlee091_comparison.jpg](comparisons\KADA-Edwinlee091_comparison.jpg)
- `UC11906884.jpg` → [comparisons\UC11906884_comparison.jpg](comparisons\UC11906884_comparison.jpg)
