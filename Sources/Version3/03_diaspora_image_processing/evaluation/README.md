# 정량 평가 (Quantitative Evaluation)

본 디렉토리는 이미지 처리 모듈의 정량적 성능을 평가하는 코드를 포함합니다. 학위논문의 mandatory 이슈(통계적 유효성, 공정한 baseline 비교, 재현성)에 직접 대응하도록 설계되었습니다.

## 평가 방법론

### 핵심 난제: Ground Truth가 없다

100년 된 디아스포라 사진의 "원본 깨끗한 컬러 버전"은 존재하지 않습니다. PSNR/SSIM/LPIPS 같은 정량 지표는 GT가 필수입니다.

### 해결책: 합성 열화 (Synthetic Degradation)

학계 표준 방식(Real-ESRGAN, GFPGAN 논문이 모두 사용)을 따릅니다.

```
[GT 컬러 고화질] 
      ↓ 합성 열화 (4배 다운스케일 + 블러 + 노이즈 + 흑백화 + JPEG 압축)
[저화질 흑백 입력]
      ↓ 복원
      ├─→ Bicubic interpolation (Baseline 1)
      ├─→ Real-ESRGAN 단독 (Baseline 2)  
      └─→ 우리 통합 시스템 (Ours)
                  ↓
            [복원 결과]
                  ↓
           GT와 비교 → PSNR / SSIM / LPIPS
                  ↓
      Wilcoxon signed-rank test (vs Bicubic)
```

### 산출 지표

| 지표 | 측정 대상 | 좋은 방향 | 의미 |
|------|----------|-----------|------|
| **PSNR** | 픽셀 차이 (dB) | 높을수록 ↑ | 신호 대 잡음 비율 |
| **SSIM** | 구조적 유사도 | 1에 가까울수록 ↑ | 휘도·대비·구조 유사도 |
| **LPIPS** | 인간 지각 거리 | 0에 가까울수록 ↓ | 학습된 perceptual feature 거리 |
| **BRISQUE** | 자연성 (참조 없음) | 낮을수록 ↓ | 자연 통계로부터의 거리 |

### Baseline 단계

1. **Bicubic interpolation** — SR 분야의 가장 기본 baseline
2. **Real-ESRGAN 단독** — SR만 적용한 결과 (얼굴 복원·컬러화 없이)
3. **우리 통합 시스템** — SR + GFPGAN + DDColor + Analysis 전부

이 점진적 비교는 "통합 시스템이 단일 모델 대비 얼마나 추가 개선을 가져오는가"를 정량화합니다.

### 통계 검정

**Wilcoxon signed-rank test** (비모수, 대응 표본):
- 메모리 정책: "non-parametric tests when sample expansion is constrained"
- 정규성 가정 불필요, 작은 표본에서도 적용 가능
- 다중 비교 시 **Bonferroni 보정** 자동 적용

---

## 의존성 설치

기본 의존성은 이미 메인 시스템에 포함되어 있습니다. 평가용 추가 패키지:

```powershell
pip install scikit-image lpips piq scipy
```

**LPIPS 모델 가중치**는 최초 실행 시 자동 다운로드 (약 5MB, AlexNet backbone).

---

## 사용법

### 1. GT 이미지 준비

`data/eval/gt/` 폴더에 GT 이미지(깨끗한 컬러 고화질)를 두세요.

**권장 GT 출처:**

| 출처 | 설명 | 학위논문 가치 |
|------|------|--------------|
| **DIV2K** | SR 학계 표준, 1000장 | 학계 표준 비교 가능 |
| **Set14** | 작은 평가셋, 14장 | 빠른 검증 |
| **Urban100** | 도시 이미지, 100장 | SR 어려운 케이스 |
| **디아스포라 컬러사진** | 1980년대 이후 컬러 한인 사진 | 도메인 일치 ★ |

가장 강력한 학위논문 자료는 **DIV2K(또는 Set14) + 디아스포라 컬러사진** 두 세트로 각각 평가하는 것입니다.

### 2. 전체 평가 실행

```powershell
# GT가 data/eval/gt/ 에 있을 때
python -m evaluation.evaluate --gt-dir data/eval/gt
```

다음 단계가 자동으로 실행됩니다:
1. 합성 열화 → `data/eval/degraded/`
2. Bicubic 적용 → `data/eval/restored_bicubic/`
3. Real-ESRGAN 적용 → `data/eval/restored_realesrgan/`
4. 우리 시스템 적용 → `data/eval/restored_ours/`
5. PSNR/SSIM/LPIPS 계산 + 통계 검정
6. JSON/MD 리포트 → `data/eval/reports/`

### 3. 부분 실행 (시간 절약)

이미 열화된 입력이나 baseline 결과가 있으면:

```powershell
# 합성 열화는 한 번만 하고, 이후 평가만 다시
python -m evaluation.evaluate --gt-dir data/eval/gt --skip-degrade

# 모든 처리 결과가 이미 있고 지표만 다시 계산
python -m evaluation.evaluate --gt-dir data/eval/gt --skip-degrade --skip-baselines
```

### 4. No-reference 평가 추가 (실제 디아스포라 사진)

```powershell
python -m evaluation.evaluate \
    --gt-dir data/eval/gt \
    --no-reference data/input
```

`data/input/` 의 실제 디아스포라 흑백 사진에 대해 BRISQUE 산출. 우리 시스템 적용 전후의 자연성 변화를 측정.

### 5. 개별 모듈 단독 사용

```powershell
# 단일 이미지 열화
python -m evaluation.degrade -i some.jpg -o degraded.jpg

# 두 이미지 비교
python -m evaluation.metrics --gt gt.jpg --pred restored.jpg

# Bicubic 일괄 적용
python -m evaluation.baselines --method bicubic -i degraded/ -o bicubic/
```

---

## 출력 예시

### `eval_report_<timestamp>.md`

```markdown
# 정량 평가 리포트

## 2.1 PSNR (↑)

| Method | n | Mean | Std | Median | p (vs Bicubic) | Sig. |
|--------|---|------|-----|--------|----------------|------|
| Bicubic | 30 | 24.3215 | 2.1043 | 24.5621 | (reference) | — |
| Real-ESRGAN | 30 | 27.8421 | 1.9876 | 27.9234 | 0.0021 | ** |
| Ours ★ | 30 | 29.1245 | 1.7654 | 29.2541 | 0.0015 | ** |

유의수준: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant
```

이 표는 학위논문 평가 챕터에 거의 그대로 옮길 수 있는 형태입니다.

---

## 학위논문 활용 가이드

### 권장 본문 위치

- **제5장 평가** / **5.X 정량 평가**

### 본문 서술 템플릿

```
표 5.X는 30장의 이미지에 대한 PSNR/SSIM/LPIPS 결과이다. 
본 시스템은 Bicubic baseline 대비 PSNR 4.80 dB, SSIM 0.13, LPIPS -0.24 만큼
개선되었으며, 이 차이는 모두 통계적으로 유의했다 (Wilcoxon signed-rank,
p < 0.01, Bonferroni 보정 적용).

또한 Real-ESRGAN 단독 적용 대비에서도 PSNR 1.28 dB의 추가 개선이 관찰되어
(p = 0.034), 본 시스템의 얼굴 복원 및 컬러화 모듈이 정량적으로도
유의미한 기여를 한다는 것을 확인할 수 있다.
```

### Mandatory 이슈 매핑

| 이슈 | 본 평가의 대응 |
|------|---------------|
| **통계적 유효성** | Wilcoxon signed-rank + Bonferroni 보정 |
| **공정한 baseline 비교** | Bicubic / Real-ESRGAN / 통합 시스템의 점진적 비교 |
| **표본 크기 (n≥30)** | DIV2K로 30장 이상 GT 확보 가능 |
| **재현성** | 시드 고정, 합성 열화 파라미터 명시 |

---

## 주의사항

1. **합성 열화는 진짜 100년 된 사진의 열화와 다를 수 있습니다.** 이 한계를 학위논문에 명시하고, No-reference 평가(BRISQUE)로 보완하는 것이 정직한 방식입니다.

2. **GT 이미지의 도메인이 결과에 영향을 줍니다.** DIV2K(자연 풍경 위주)와 디아스포라 컬러사진은 다른 결과를 보일 수 있고, 이는 오히려 흥미로운 분석 포인트입니다.

3. **표본 크기가 작으면(n<10) 통계 검정 결과를 신뢰할 수 없습니다.** 가능한 한 n≥30 권장.

4. **LPIPS 모델 다운로드**: 최초 실행 시 인터넷 필요. 학회 발표 노트북에서는 사전 1회 실행 필수.
