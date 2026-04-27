# 01_diaspora_archive_input — 아카이브 입력 시스템

디아스포라 자료의 입력 및 분류 모듈입니다.

## 기능
- 다양한 형식의 자료 자동 분류 (이미지/문서/영상/음성/구술)
- 파일 메타데이터 추출 (크기, 시간, 해시)
- 후속 모듈로의 라우팅 정보 생성
- 매니페스트 JSON 출력

## 자료 유형 분류

| 카테고리 | 확장자 | 라우팅 모듈 |
|---------|--------|------------|
| image | .jpg, .png, .tiff 등 | 03_diaspora_image_processing |
| document | .pdf, .docx, .hwp 등 | 02_diaspora_document_processing |
| video | .mp4, .avi, .mov 등 | 04_diaspora_video_processing |
| audio | .mp3, .wav, .flac 등 | 05_diaspora_audio_processing |
| oral | (audio + 구술 키워드) | 06_diaspora_oral_processing |

## 사용법

```powershell
# 단일 파일
python main.py --input data/input/photo.jpg

# 디렉토리 일괄
python main.py --input data/input/ --batch

# 분류만
python main.py --input data/input/ --classify-only --verbose
```

## 출력
`data/output/manifest_<timestamp>.json` 에 분류 결과 저장
