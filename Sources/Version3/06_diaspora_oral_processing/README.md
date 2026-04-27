# 06_diaspora_oral_processing — 구술 처리 모듈

디아스포라 구술사·인터뷰 자료의 통합 처리 모듈.

## 기능
- 구술 음성 전사 (Whisper)
- 화자 분리 (pyannote.audio)
- 전사 + 화자 정보 자동 병합
- 화자별 발화 정리 / 대화 형식 출력
- SRT 자막 생성

## 의존성
```powershell
pip install openai-whisper pyannote.audio
```

**중요**: pyannote.audio 사용에는 HuggingFace 토큰이 필요합니다.
1. https://huggingface.co/pyannote/speaker-diarization-3.1 약관 동의
2. https://huggingface.co/settings/tokens 에서 토큰 발급

## 사용법

```powershell
# 자동 화자 수 감지
python main.py --input data/input/interview.wav --hf-token YOUR_TOKEN

# 화자 수 지정 (인터뷰는 보통 2명)
python main.py --input data/input/interview.wav --num-speakers 2 --hf-token YOUR_TOKEN

# 화자 분리 없이 전사만
python main.py --input data/input/interview.wav --no-diarization
```

## 출력
- `<filename>_dialogue.txt` — 대화 형식 (시간순)
- `<filename>_by_speaker.txt` — 화자별 정리
- `<filename>.srt` — 자막
- `oral_report_<timestamp>.json` — 통합 결과
