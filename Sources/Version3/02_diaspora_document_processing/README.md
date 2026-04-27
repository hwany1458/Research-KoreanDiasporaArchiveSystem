# 02_diaspora_document_processing — 문서 처리 모듈

디아스포라 문서 자료(편지, 일기, 공문서 등)의 OCR/HTR 처리 모듈입니다.

## 기능
- 인쇄체 OCR (EasyOCR — 한글/영어/한자)
- 손글씨 HTR (TrOCR)
- 개체명 인식 (KoELECTRA)
- 정규식 기반 날짜/연도 추출 fallback

## 의존성
```powershell
pip install easyocr transformers torch
```

## 사용법

```powershell
# 인쇄체 문서
python main.py --input data/input/letter.jpg --mode ocr

# 손글씨 문서
python main.py --input data/input/diary.jpg --mode htr

# 둘 다 적용
python main.py --input data/input/mixed.jpg --mode both --verbose

# 일괄 처리
python main.py --input data/input/ --batch
```

## 출력
JSON 리포트에 OCR 결과, HTR 결과, 추출된 개체명 (인물·조직·장소·날짜) 포함
