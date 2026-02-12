
## 🔧 모듈별 주요 기능

### 1. 이미지 처리 (image/restoration.py)

- ImageRestorer: 통합 이미지 복원 파이프라인
- SuperResolution: Real-ESRGAN 초해상도
- FaceEnhancer: GFPGAN 얼굴 향상
- ImageColorizer: DeOldify 컬러화
- ImageCaptioner: BLIP-2 캡셔닝
- SceneClassifier: CLIP 장면 분류

### 2. 문서 처리 (document/ocr.py)

- DocumentProcessor: 통합 문서 처리
- PrintedTextOCR: EasyOCR/Tesseract
- HandwritingRecognizer: TrOCR 손글씨
- LayoutAnalyzer: LayoutLMv3 레이아웃
- NamedEntityRecognizer: 개체명 인식

### 3. 동영상 처리 (video/processing.py)

- VideoProcessor: 통합 비디오 처리
- SceneDetector: PySceneDetect 장면 분할
- KeyframeExtractor: 핵심 프레임 추출
- VideoEnhancer: 품질 향상
- VideoStabilizer: 손떨림 보정
- SubtitleGenerator: Whisper 자막 생성

### 4. 음성 처리 (audio/transcription.py)

- AudioProcessor: 통합 음성 처리
- SpeechRecognizer: Whisper 음성 인식
- SpeakerDiarizer: pyannote 화자 분리
- CodeSwitchingDetector: 코드 스위칭 감지
- EmotionAnalyzer: 감정 분석

### 5. 구술 처리 (oral_history/processing.py)

- OralHistoryProcessor: 통합 구술 분석
- TopicModeler: BERTopic 주제 분할
- QuoteExtractor: 핵심 인용구 추출
- TimelineExtractor: 타임라인 생성
- NarrativeAnalyzer: 내러티브 구조 분석
- SentimentAnalyzer: 감정 분석

