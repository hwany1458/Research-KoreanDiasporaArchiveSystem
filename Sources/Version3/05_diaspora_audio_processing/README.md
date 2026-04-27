# 05_diaspora_audio_processing — 음성 처리 모듈

디아스포라 음성 자료 (녹음, 음성 편지 등)의 전사 모듈.

## 기능
- 음성 → 텍스트 (Whisper / Faster-Whisper)
- 한국어/영어 자동 감지
- SRT 자막 생성
- 평문 텍스트 저장

## 의존성
```powershell
pip install openai-whisper
# 또는 더 빠른 옵션:
pip install faster-whisper
```

## 사용법

```powershell
# 자동 언어 감지
python main.py --input data/input/audio.wav

# 한국어 강제
python main.py --input data/input/audio.wav --language ko

# 더 정확한 모델
python main.py --input data/input/audio.wav --model medium
```

모델 크기: tiny < base < small < medium < large

## 출력
- `<filename>.srt` — 자막 파일
- `<filename>.txt` — 전체 텍스트
- `audio_report_<timestamp>.json` — 통합 결과
