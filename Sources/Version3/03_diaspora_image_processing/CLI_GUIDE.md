# CLI 사용 가이드

본 문서는 본 시스템에서 제공하는 모든 명령어 진입점과 그 옵션들을 정리한 가이드입니다.

## 진입점 스크립트 목록

| 카테고리 | 스크립트 | 역할 |
|---------|----------|------|
| **메인** | `main.py` | 단일 이미지 전체 파이프라인 처리 |
| **메인** | `batch_process.py` | 여러 이미지 일괄 처리 |
| **모듈** | `python -m src.modules.super_resolution` | 초해상도 단독 |
| **모듈** | `python -m src.modules.colorization` | 컬러화 단독 |
| **모듈** | `python -m src.pipeline` | 파이프라인 단독 (main.py와 거의 동일) |
| **평가** | `python -m evaluation.evaluate` | 정량 평가 (전체 워크플로우) |
| **평가** | `python -m evaluation.degrade` | 합성 열화 단독 |
| **평가** | `python -m evaluation.metrics` | 지표 계산 단독 |
| **평가** | `python -m evaluation.baselines` | Baseline 단독 적용 |
| **테스트** | `test_ddcolor.py` | DDColor 동작 검증 |

---

## 0. 공통 사전 조건

모든 명령어 실행 전 conda 환경이 활성화되어 있어야 합니다.

```powershell
conda activate diaspora
cd C:\Users\<YourName>\downloads\youkyoung\diaspora_thesis_code\03_diaspora_image_processing
```

---

## 1. main.py — 단일 이미지 처리

이미지 1장에 대해 전체 파이프라인(SR → 얼굴 복원 → 컬러화 → 분석)을 적용합니다.

### 옵션

| 옵션 | 단축형 | 필수? | 기본값 | 설명 |
|------|--------|-------|--------|------|
| `--input` | `-i` | **필수** | — | 입력 이미지 파일 경로 |
| `--output` | `-o` | **필수** | — | 출력 이미지 파일 경로 |
| `--config` | `-c` | 선택 | `None` | YAML 설정 파일 경로 |
| `--device` | `-d` | 선택 | `cuda` | 연산 장치 (`cuda` 또는 `cpu`) |
| `--batch` | `-b` | 선택 | `False` | 배치 모드로 전환 (디렉토리 입력 시 자동) |
| `--report` | `-r` | 선택 | `None` | JSON 처리 보고서 출력 경로 |
| `--no-sr` | — | 선택 | `False` | 초해상도 비활성화 |
| `--no-face` | — | 선택 | `False` | 얼굴 향상 비활성화 |
| `--no-color` | — | 선택 | `False` | 컬러화 비활성화 |
| `--no-analysis` | — | 선택 | `False` | 이미지 분석 비활성화 |
| `--save-intermediate` | — | 선택 | `False` | 단계별 중간 결과도 함께 저장 |
| `--verbose` | `-v` | 선택 | `False` | 상세 로그 출력 |

### 예시

#### 예시 1-1. 가장 기본 (전체 파이프라인 적용)
```powershell
python main.py --input data/input/photo.jpg --output data/output/photo_restored.jpg
```

#### 예시 1-2. 상세 로그 출력
```powershell
python main.py -i data/input/photo.jpg -o data/output/photo_restored.jpg --verbose
```

#### 예시 1-3. 컬러화 없이 (이미 컬러 사진일 때 강제로 끄고 싶은 경우)
```powershell
python main.py -i data/input/color_photo.jpg -o data/output/result.jpg --no-color
```

#### 예시 1-4. SR만 적용 (얼굴 복원·컬러화·분석 모두 끔)
```powershell
python main.py -i data/input/photo.jpg -o data/output/sr_only.jpg --no-face --no-color --no-analysis
```

#### 예시 1-5. CPU 모드 (GPU 없는 환경)
```powershell
python main.py -i data/input/photo.jpg -o data/output/result.jpg --device cpu
```

#### 예시 1-6. 처리 보고서 함께 저장
```powershell
python main.py -i data/input/photo.jpg -o data/output/result.jpg --report data/output/report.json
```

#### 예시 1-7. 중간 단계 결과까지 저장 (디버깅용)
```powershell
python main.py -i data/input/photo.jpg -o data/output/result.jpg --save-intermediate -v
```

---

## 2. batch_process.py — 일괄 처리

`data/input/` 폴더의 모든 이미지를 자동 처리합니다.

### 옵션

| 옵션 | 단축형 | 필수? | 기본값 | 설명 |
|------|--------|-------|--------|------|
| `--input-dir` | — | 선택 | `data/input` | 입력 이미지 디렉토리 |
| `--output-dir` | — | 선택 | `data/output/batch_<timestamp>` | 출력 디렉토리 (자동 timestamp 생성) |
| `--device` | — | 선택 | `cuda` | 연산 장치 |
| `--limit` | — | 선택 | `None` | 최대 처리 이미지 수 (테스트용) |
| `--no-comparison` | — | 선택 | `False` | 비교 이미지 생성 스킵 (속도 향상) |
| `--no-report` | — | 선택 | `False` | 개별 JSON 리포트 생성 스킵 |
| `--verbose` | `-v` | 선택 | `False` | 상세 로그 출력 |

### 출력 구조

```
data/output/batch_20260427_220000/
├── restored/                       # 복원된 결과 이미지
├── comparisons/                    # 원본+결과 비교 (학위논문 figure용)
├── reports/                        # 각 이미지별 JSON 리포트
├── batch_summary.json              # 전체 일괄 처리 요약
└── batch_summary.md                # 사람이 읽기 좋은 요약
```

### 예시

#### 예시 2-1. 기본 (모든 입력 이미지 처리)
```powershell
python batch_process.py
```

#### 예시 2-2. 처음 3장만 (빠른 테스트)
```powershell
python batch_process.py --limit 3
```

#### 예시 2-3. 비교 이미지 없이 (처리 시간 약 30% 단축)
```powershell
python batch_process.py --no-comparison
```

#### 예시 2-4. 다른 입력 디렉토리 사용
```powershell
python batch_process.py --input-dir data/eval_set --verbose
```

#### 예시 2-5. 출력 디렉토리 직접 지정
```powershell
python batch_process.py --output-dir data/output/2026_thesis_demo
```

---

## 3. 개별 모듈 단독 실행

각 모듈은 독립 실행도 가능하여, 디버깅과 ablation에 유용합니다.

### 3.1 super_resolution (초해상도 단독)

```powershell
python -m src.modules.super_resolution -i input.jpg -o output.jpg
```

내부적으로 `SuperResolutionModule`을 직접 호출하므로, 이미지에 SR만 적용한 결과를 빠르게 확인할 수 있습니다.

### 3.2 colorization (컬러화 단독)

| 옵션 | 단축형 | 기본값 | 설명 |
|------|--------|--------|------|
| `--input` | `-i` | **필수** | 입력 이미지 |
| `--output` | `-o` | `output_color.jpg` | 출력 경로 |
| `--backend` | `-b` | `None` (자동) | `ddcolor` / `deoldify` / `sepia` 강제 지정 |
| `--model` | `-m` | `artistic` | DeOldify 모델 (`artistic` 또는 `stable`) |
| `--render-factor` | `-r` | `35` | DeOldify 렌더 팩터 (7-45) |
| `--device` | `-d` | `cuda` | 연산 장치 |
| `--force` | `-f` | `False` | 컬러 이미지도 강제 처리 |

#### 예시 3-2-1. 자동 백엔드로 컬러화
```powershell
python -m src.modules.colorization -i bw_photo.jpg -o color.jpg
```

#### 예시 3-2-2. DDColor 강제 사용
```powershell
python -m src.modules.colorization -i bw_photo.jpg -o color.jpg --backend ddcolor
```

#### 예시 3-2-3. Sepia fallback 테스트 (학위논문 ablation용)
```powershell
python -m src.modules.colorization -i bw_photo.jpg -o sepia.jpg --backend sepia
```

#### 예시 3-2-4. 컬러 이미지에 강제 적용 (학습 효과 비교)
```powershell
python -m src.modules.colorization -i color_photo.jpg -o forced.jpg --force
```

---

## 4. 정량 평가 (`evaluation/` 패키지)

### 4.1 evaluate.py — 전체 평가 워크플로우 (메인)

GT 이미지부터 시작해 합성 열화 → Baseline 비교 → 통계 검정까지 자동 실행.

| 옵션 | 단축형 | 기본값 | 설명 |
|------|--------|--------|------|
| `--gt-dir` | — | `data/eval/gt` | GT 이미지 디렉토리 (깨끗한 컬러 고화질) |
| `--eval-root` | — | `data/eval` | 평가 산출물 루트 디렉토리 |
| `--no-reference` | — | `None` | No-reference 평가용 디아스포라 사진 디렉토리 |
| `--device` | — | `cuda` | 연산 장치 |
| `--seed` | — | `42` | 재현성을 위한 random seed (열화 단계용) |
| `--skip-degrade` | — | `False` | 합성 열화 단계 건너뜀 |
| `--skip-baselines` | — | `False` | Baseline 처리 단계 건너뜀 |
| `--verbose` | `-v` | `False` | 상세 로그 |

#### 예시 4-1-1. 전체 평가 실행 (가장 일반적)
```powershell
# 사전: data/eval/gt/ 폴더에 GT 이미지(컬러 고화질) 30장 이상 준비
python -m evaluation.evaluate --gt-dir data/eval/gt
```

#### 예시 4-1-2. 부분 재실행 (지표만 다시 계산)
```powershell
# 합성 열화와 baseline 처리는 이미 끝난 상태에서 지표만 다시
python -m evaluation.evaluate --gt-dir data/eval/gt --skip-degrade --skip-baselines
```

#### 예시 4-1-3. No-reference 평가도 함께 (실제 디아스포라 사진)
```powershell
python -m evaluation.evaluate --gt-dir data/eval/gt --no-reference data/input
```

#### 예시 4-1-4. 다른 GT 데이터셋으로 평가 (DIV2K 사용 등)
```powershell
python -m evaluation.evaluate --gt-dir D:/datasets/DIV2K_valid --eval-root data/eval/div2k
```

#### 예시 4-1-5. 재현성 시드 변경 (다른 열화 패턴)
```powershell
python -m evaluation.evaluate --gt-dir data/eval/gt --seed 123
```

### 4.2 degrade.py — 합성 열화 단독

GT 이미지에 인위적 열화(저해상도, 흑백, 노이즈, JPEG 압축)를 적용.

| 옵션 | 단축형 | 기본값 | 설명 |
|------|--------|--------|------|
| `--input` | `-i` | **필수** | 입력 (파일 또는 디렉토리) |
| `--output` | `-o` | **필수** | 출력 (파일 또는 디렉토리) |
| `--seed` | — | `42` | 재현성 시드 |

#### 예시 4-2-1. 단일 이미지 열화
```powershell
python -m evaluation.degrade -i clean.jpg -o degraded.jpg
```

#### 예시 4-2-2. 디렉토리 일괄 열화
```powershell
python -m evaluation.degrade -i data/eval/gt -o data/eval/degraded
```

#### 예시 4-2-3. 다른 시드로 다른 열화 패턴 만들기
```powershell
python -m evaluation.degrade -i clean.jpg -o degraded_v2.jpg --seed 999
```

### 4.3 metrics.py — 지표 계산 단독

두 이미지를 비교하여 PSNR/SSIM/LPIPS 산출. 또는 단일 이미지의 BRISQUE.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--gt` | `None` | GT 이미지 경로 (있으면 full-reference, 없으면 no-reference) |
| `--pred` | **필수** | 평가할 이미지 경로 |
| `--device` | `cuda` | 연산 장치 |

#### 예시 4-3-1. Full-reference (GT와 비교)
```powershell
python -m evaluation.metrics --gt data/eval/gt/photo01.png --pred data/eval/restored_ours/photo01.jpg
```

출력:
```json
{
  "psnr": 29.12,
  "ssim": 0.8567,
  "lpips": 0.1832
}
```

#### 예시 4-3-2. No-reference (GT 없이)
```powershell
python -m evaluation.metrics --pred data/output/result.jpg
```

출력:
```json
{
  "brisque": 32.45
}
```

### 4.4 baselines.py — Baseline 단독 적용

| 옵션 | 단축형 | 기본값 | 설명 |
|------|--------|--------|------|
| `--method` | — | **필수** | `bicubic` / `real_esrgan` / `full_pipeline` |
| `--input` | `-i` | **필수** | 입력 디렉토리 |
| `--output` | `-o` | **필수** | 출력 디렉토리 |
| `--device` | — | `cuda` | 연산 장치 |

#### 예시 4-4-1. Bicubic만 적용
```powershell
python -m evaluation.baselines --method bicubic -i data/eval/degraded -o data/eval/restored_bicubic
```

#### 예시 4-4-2. Real-ESRGAN 단독 적용
```powershell
python -m evaluation.baselines --method real_esrgan -i data/eval/degraded -o data/eval/restored_realesrgan
```

#### 예시 4-4-3. 우리 통합 시스템 (batch_process.py와 동일)
```powershell
python -m evaluation.baselines --method full_pipeline -i data/eval/degraded -o data/eval/restored_ours
```

---

## 5. test_ddcolor.py — DDColor 동작 검증

DDColor 모델 자체를 단독 테스트. 환경 점검용.

| 옵션 | 단축형 | 기본값 | 설명 |
|------|--------|--------|------|
| `--input` | `-i` | **필수** | 입력 이미지 |
| `--output` | `-o` | **필수** | 출력 이미지 |
| `--device` | `-d` | `cuda` | 연산 장치 |

### 예시

```powershell
python test_ddcolor.py -i data/input/bw.jpg -o data/output/test_color.jpg
```

DDColor 가중치가 정상 다운로드되어 추론까지 가능한지 확인할 때 사용.

---

## 6. 자주 사용하는 워크플로우 모음

### 6.1 학위논문 시연 데모 (가장 빠른 결과)

```powershell
# 활성화
conda activate diaspora

# 단일 이미지 빠른 시연
python main.py -i data/demo/wedding.jpg -o data/demo/wedding_restored.jpg
```

### 6.2 학위논문 figure용 일괄 처리

```powershell
# 비교 이미지 + 마크다운 요약 모두 생성
python batch_process.py
# 결과: data/output/batch_<timestamp>/comparisons/ 에 학위논문 figure용 이미지
# 결과: data/output/batch_<timestamp>/batch_summary.md 에 표 형식 요약
```

### 6.3 학위논문 평가 챕터 자료 생성 (정량 평가)

```powershell
# Step 1: GT 이미지 준비 (data/eval/gt/ 에 30장 이상)
# Step 2: 전체 평가 실행
python -m evaluation.evaluate --gt-dir data/eval/gt
# 결과: data/eval/reports/eval_report_<timestamp>.md 에 통계표
```

### 6.4 ablation study 워크플로우

```powershell
# (1) 컬러화 백엔드 비교
python -m src.modules.colorization -i bw.jpg -o ddcolor.jpg --backend ddcolor
python -m src.modules.colorization -i bw.jpg -o sepia.jpg --backend sepia

# (2) 모듈별 ON/OFF 비교
python main.py -i in.jpg -o sr_only.jpg --no-face --no-color --no-analysis
python main.py -i in.jpg -o no_color.jpg --no-color
python main.py -i in.jpg -o full.jpg
```

### 6.5 학회 발표 직전 캐시 워밍업

```powershell
# 모든 모델이 다운로드되어 있는지 확인 (최초 1회)
python batch_process.py --limit 1

# 발표 직전 한 번 더 빠른 테스트
python main.py -i data/demo/test.jpg -o data/demo/test_out.jpg
```

---

## 7. 옵션이 충돌할 때의 우선순위

여러 옵션을 동시에 사용할 때:

1. **`--no-XX` 옵션은 무조건 OFF 우선** — `--no-color`가 있으면 컬러화는 무조건 안 됨
2. **CLI 옵션이 `--config` 파일보다 우선** — config에 SR enabled=true여도 `--no-sr`이 있으면 SR은 꺼짐
3. **`--device cuda`가 기본이지만 GPU 없으면 자동 CPU로** — 명시적 fallback

---

## 8. 트러블슈팅

### 8.1 "ModuleNotFoundError: No module named 'evaluation'"

**원인**: 프로젝트 루트가 아닌 다른 디렉토리에서 실행.

**해결**:
```powershell
cd <project_root>  # 즉 main.py가 있는 디렉토리
python -m evaluation.evaluate ...
```

### 8.2 `python -m src.modules.xxx` 가 안 될 때

**원인**: `src/__init__.py` 또는 `src/modules/__init__.py` 가 없음.

**해결**:
```powershell
# 빈 __init__.py 파일 생성
echo. > src\__init__.py
echo. > src\modules\__init__.py
```

### 8.3 인자에 한글 경로/공백이 있을 때

따옴표로 감싸세요:
```powershell
python main.py -i "data/input/제주도 사진.jpg" -o "data/output/result.jpg"
```

### 8.4 옵션을 잊었을 때 — `--help` 사용

모든 스크립트는 `-h` 또는 `--help` 지원:

```powershell
python main.py --help
python batch_process.py --help
python -m evaluation.evaluate --help
python -m evaluation.degrade --help
```

가장 정확한 옵션 정보는 `--help`에서 확인할 수 있습니다.

---

## 9. 빠른 참조 카드 (Quick Reference)

```
# === 단일 처리 ===
python main.py -i IN.jpg -o OUT.jpg [-v] [--no-sr|--no-face|--no-color|--no-analysis]

# === 일괄 처리 ===
python batch_process.py [--limit N] [--no-comparison] [--input-dir DIR]

# === 정량 평가 ===
python -m evaluation.evaluate --gt-dir GT_DIR [--no-reference NR_DIR] [--seed N]

# === 보조 도구 ===
python -m evaluation.degrade -i IN -o OUT [--seed N]
python -m evaluation.metrics --gt GT --pred PRED
python -m evaluation.baselines --method {bicubic|real_esrgan|full_pipeline} -i DIR -o DIR

# === 컬러화 단독 ===
python -m src.modules.colorization -i IN -o OUT [--backend ddcolor|deoldify|sepia] [--force]

# === 환경 검증 ===
python test_ddcolor.py -i IN -o OUT
python -c "from src.pipeline import ImageRestorationPipeline; print('OK')"
```

복사해서 책상에 붙여두기 좋은 형태입니다.
