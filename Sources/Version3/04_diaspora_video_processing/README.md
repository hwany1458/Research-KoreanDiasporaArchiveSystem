# 04_diaspora_video_processing — 영상 처리 모듈

디아스포라 영상 자료(홈비디오, 행사 영상 등) 처리 모듈.

## 기능
- 영상 메타데이터 추출 (길이, 해상도, FPS)
- 키프레임 추출 (uniform sampling)
- 장면 분할 (PySceneDetect 또는 frame-diff fallback)
- 음성 트랙 분리 (ffmpeg)
- 자막 자동 생성 (Whisper, 한국어)

## 의존성
```powershell
pip install opencv-python scenedetect openai-whisper
# ffmpeg 별도 설치 필요 (https://ffmpeg.org/)
```

## 사용법

```powershell
# 전체 처리
python main.py --input data/input/family.mp4 --verbose

# 키프레임만 (빠르게)
python main.py --input data/input/family.mp4 --no-scenes --no-audio --no-subtitle

# 키프레임 수 조정
python main.py --input data/input/family.mp4 --num-keyframes 20
```

## 출력
- `<videoname>_keyframes/` — 키프레임 이미지 (03 모듈로 후속 처리 가능)
- `<videoname>_audio.wav` — 추출된 음성 (16kHz mono)
- `<videoname>_subtitle.srt` — 한국어 자막
- `video_report_<timestamp>.json` — 통합 결과
