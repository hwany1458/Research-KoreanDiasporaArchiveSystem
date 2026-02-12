"""
음성 처리 모듈 (Audio Processing Module)

한인 디아스포라 기록유산의 음성 인식 및 분석을 위한 AI 기반 모듈

주요 기능:
- 음성 전처리 (Audio Preprocessing): 노이즈 제거, 음량 정규화
- 음성 인식 (Speech Recognition): Whisper
- 화자 분리 (Speaker Diarization): pyannote.audio
- 다국어 처리 (Multilingual Processing)
- 코드 스위칭 감지 (Code-Switching Detection)
- 타임스탬프 정렬 (Timestamp Alignment)
- 감정 분석 (Emotion Analysis)

Author: Diaspora Archive Project
License: MIT
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """오디오 포맷"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"
    UNKNOWN = "unknown"


class Language(Enum):
    """언어"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    RUSSIAN = "ru"
    MIXED = "mixed"


class EmotionType(Enum):
    """감정 유형"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    NOSTALGIC = "nostalgic"


@dataclass
class AudioMetadata:
    """오디오 메타데이터"""
    duration: float
    sample_rate: int
    channels: int
    format: AudioFormat
    bit_depth: int = 16
    file_size: int = 0
    is_mono: bool = True
    has_speech: bool = True
    noise_level: float = 0.0
    silence_ratio: float = 0.0


@dataclass
class TranscriptionSegment:
    """전사 세그먼트"""
    start_time: float
    end_time: float
    text: str
    confidence: float
    language: Language = Language.KOREAN
    speaker: Optional[str] = None
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """화자 세그먼트"""
    speaker_id: str
    speaker_label: str
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """전사 결과"""
    full_text: str
    segments: List[TranscriptionSegment]
    speakers: List[str]
    speaker_segments: List[SpeakerSegment]
    detected_languages: List[Language]
    duration: float
    processing_time: float
    word_count: int
    confidence: float


class AudioPreprocessor:
    """오디오 전처리"""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        logger.info(f"AudioPreprocessor initialized (target_sr: {target_sample_rate})")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """오디오 로드"""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            return audio, sr
        except ImportError:
            logger.warning("librosa not available, using scipy")
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32) / 32768.0
            return audio, sr
    
    def save_audio(self, audio: np.ndarray, path: str, sample_rate: int):
        """오디오 저장"""
        try:
            import soundfile as sf
            sf.write(path, audio, sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(path, sample_rate, audio_int)
    
    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        denoise: bool = True,
        normalize: bool = True,
        remove_silence: bool = False
    ) -> np.ndarray:
        """오디오 전처리"""
        result = audio.copy()
        
        # 노이즈 제거
        if denoise:
            result = self._reduce_noise(result, sample_rate)
        
        # 음량 정규화
        if normalize:
            result = self._normalize_audio(result)
        
        # 묵음 제거 (선택적)
        if remove_silence:
            result = self._remove_silence(result, sample_rate)
        
        return result
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """노이즈 제거"""
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.8)
        except ImportError:
            logger.warning("noisereduce not available, skipping noise reduction")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """음량 정규화"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
            audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def _remove_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_db: int = 30
    ) -> np.ndarray:
        """묵음 제거"""
        try:
            import librosa
            intervals = librosa.effects.split(audio, top_db=top_db)
            if len(intervals) == 0:
                return audio
            
            non_silent = []
            for start, end in intervals:
                non_silent.append(audio[start:end])
            
            return np.concatenate(non_silent) if non_silent else audio
        except ImportError:
            return audio
    
    def get_metadata(self, audio_path: str) -> AudioMetadata:
        """오디오 메타데이터 추출"""
        audio, sr = self.load_audio(audio_path)
        
        duration = len(audio) / sr
        file_size = os.path.getsize(audio_path)
        
        # 묵음 비율 계산
        threshold = 0.01
        silence_samples = np.sum(np.abs(audio) < threshold)
        silence_ratio = silence_samples / len(audio)
        
        # 노이즈 레벨 추정
        noise_level = self._estimate_noise_level(audio)
        
        # 포맷 감지
        ext = os.path.splitext(audio_path)[1].lower()
        format_map = {
            ".wav": AudioFormat.WAV,
            ".mp3": AudioFormat.MP3,
            ".flac": AudioFormat.FLAC,
            ".ogg": AudioFormat.OGG,
            ".m4a": AudioFormat.M4A
        }
        audio_format = format_map.get(ext, AudioFormat.UNKNOWN)
        
        return AudioMetadata(
            duration=duration,
            sample_rate=sr,
            channels=1,
            format=audio_format,
            file_size=file_size,
            is_mono=True,
            has_speech=silence_ratio < 0.9,
            noise_level=noise_level,
            silence_ratio=silence_ratio
        )
    
    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """노이즈 레벨 추정"""
        # 짧은 구간들의 RMS 중 최소값을 노이즈로 추정
        window_size = 1024
        rms_values = []
        
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if rms_values:
            return float(np.percentile(rms_values, 10))
        return 0.0


class SpeechRecognizer:
    """음성 인식 (Whisper)"""
    
    LANGUAGE_MAP = {
        "ko": Language.KOREAN,
        "en": Language.ENGLISH,
        "ja": Language.JAPANESE,
        "zh": Language.CHINESE,
        "ru": Language.RUSSIAN
    }
    
    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None
        logger.info(f"SpeechRecognizer initialized: Whisper {model_size} on {device}")
    
    def load_model(self):
        """Whisper 모델 로드"""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper {self.model_size} loaded successfully")
        except ImportError:
            logger.error("Whisper not installed")
            raise
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> List[TranscriptionSegment]:
        """음성 전사"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": False
        }
        
        if language:
            options["language"] = language
        
        result = self.model.transcribe(audio_path, **options)
        
        segments = []
        detected_lang = result.get("language", "ko")
        lang_enum = self.LANGUAGE_MAP.get(detected_lang, Language.KOREAN)
        
        for seg in result["segments"]:
            words = []
            if word_timestamps and "words" in seg:
                for word in seg["words"]:
                    words.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "probability": word.get("probability", 0.0)
                    })
            
            segment = TranscriptionSegment(
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob", 0),
                language=lang_enum,
                words=words
            )
            segments.append(segment)
        
        return segments
    
    def detect_language(self, audio_path: str) -> Tuple[Language, float]:
        """언어 감지"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        import whisper
        
        # 오디오 로드
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # 멜 스펙트로그램
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # 언어 감지
        _, probs = self.model.detect_language(mel)
        
        # 최고 확률 언어
        best_lang = max(probs, key=probs.get)
        best_prob = probs[best_lang]
        
        lang_enum = self.LANGUAGE_MAP.get(best_lang, Language.KOREAN)
        
        return lang_enum, best_prob


class SpeakerDiarizer:
    """화자 분리 (pyannote.audio)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipeline = None
        logger.info(f"SpeakerDiarizer initialized on {device}")
    
    def load_model(self, auth_token: Optional[str] = None):
        """pyannote 파이프라인 로드"""
        try:
            from pyannote.audio import Pipeline
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            
            logger.info("pyannote pipeline loaded successfully")
            
        except ImportError:
            logger.warning("pyannote.audio not installed")
        except Exception as e:
            logger.error(f"Failed to load pyannote: {e}")
    
    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> List[SpeakerSegment]:
        """화자 분리"""
        if self.pipeline is None:
            logger.warning("Diarization pipeline not loaded, using fallback")
            return self._fallback_diarization(audio_path)
        
        diarization = self.pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        segments = []
        speaker_map = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker_{len(speaker_map) + 1}"
            
            segment = SpeakerSegment(
                speaker_id=speaker,
                speaker_label=speaker_map[speaker],
                start_time=turn.start,
                end_time=turn.end,
                confidence=0.9
            )
            segments.append(segment)
        
        return segments
    
    def _fallback_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """폴백 화자 분리 (에너지 기반)"""
        preprocessor = AudioPreprocessor()
        audio, sr = preprocessor.load_audio(audio_path)
        
        # 간단한 에너지 기반 세그멘테이션
        window_size = int(sr * 0.5)  # 500ms
        hop_size = int(sr * 0.25)    # 250ms
        
        segments = []
        current_speaker = "Speaker_1"
        segment_start = 0.0
        
        for i in range(0, len(audio) - window_size, hop_size):
            time = i / sr
            
            # 에너지 계산
            window = audio[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            
            # 묵음 감지 (화자 전환 가능 지점)
            if energy < 0.01:
                if time - segment_start > 0.5:
                    segments.append(SpeakerSegment(
                        speaker_id=current_speaker,
                        speaker_label=current_speaker,
                        start_time=segment_start,
                        end_time=time,
                        confidence=0.5
                    ))
                    segment_start = time
        
        # 마지막 세그먼트
        total_duration = len(audio) / sr
        if total_duration - segment_start > 0.5:
            segments.append(SpeakerSegment(
                speaker_id=current_speaker,
                speaker_label=current_speaker,
                start_time=segment_start,
                end_time=total_duration,
                confidence=0.5
            ))
        
        return segments if segments else [SpeakerSegment(
            speaker_id="Speaker_1",
            speaker_label="Speaker_1",
            start_time=0.0,
            end_time=total_duration,
            confidence=0.5
        )]


class CodeSwitchingDetector:
    """코드 스위칭 감지"""
    
    def __init__(self):
        logger.info("CodeSwitchingDetector initialized")
    
    def detect(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[Dict[str, Any]]:
        """코드 스위칭 구간 감지"""
        switch_points = []
        
        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            
            if prev_seg.language != curr_seg.language:
                switch_points.append({
                    "time": curr_seg.start_time,
                    "from_language": prev_seg.language.value,
                    "to_language": curr_seg.language.value,
                    "prev_text": prev_seg.text[-50:] if len(prev_seg.text) > 50 else prev_seg.text,
                    "curr_text": curr_seg.text[:50] if len(curr_seg.text) > 50 else curr_seg.text
                })
        
        return switch_points
    
    def analyze_mixing(
        self,
        text: str
    ) -> Dict[str, float]:
        """텍스트 내 언어 혼용 분석"""
        import re
        
        # 각 언어 문자 수 계산
        korean = len(re.findall(r'[가-힣]', text))
        english = len(re.findall(r'[a-zA-Z]', text))
        japanese = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
        chinese = len(re.findall(r'[\u4E00-\u9FFF]', text))
        russian = len(re.findall(r'[а-яА-Я]', text))
        
        total = korean + english + japanese + chinese + russian
        
        if total == 0:
            return {"korean": 1.0}
        
        return {
            "korean": korean / total,
            "english": english / total,
            "japanese": japanese / total,
            "chinese": chinese / total,
            "russian": russian / total
        }


class EmotionAnalyzer:
    """음성 감정 분석"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        logger.info(f"EmotionAnalyzer initialized on {device}")
    
    def load_model(self):
        """감정 분석 모델 로드"""
        try:
            from transformers import pipeline
            
            self.model = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Emotion analysis model loaded")
            
        except ImportError:
            logger.warning("Transformers not available for emotion analysis")
    
    def analyze(self, audio_path: str) -> List[Dict[str, Any]]:
        """감정 분석"""
        if self.model is None:
            return self._rule_based_analysis(audio_path)
        
        try:
            results = self.model(audio_path)
            
            emotions = []
            for result in results:
                emotions.append({
                    "emotion": result["label"],
                    "confidence": result["score"]
                })
            
            return emotions
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return self._rule_based_analysis(audio_path)
    
    def _rule_based_analysis(self, audio_path: str) -> List[Dict[str, Any]]:
        """규칙 기반 감정 분석 (폴백)"""
        preprocessor = AudioPreprocessor()
        audio, sr = preprocessor.load_audio(audio_path)
        
        # 간단한 특성 추출
        energy = np.sqrt(np.mean(audio ** 2))
        
        # 음높이 변화 (기본적인 추정)
        try:
            import librosa
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        except:
            pitch_mean = 0
            pitch_std = 0
        
        # 규칙 기반 감정 추정
        if energy > 0.1 and pitch_std > 50:
            emotion = "happy"
            confidence = 0.6
        elif energy < 0.02:
            emotion = "sad"
            confidence = 0.5
        elif energy > 0.15 and pitch_mean > 200:
            emotion = "angry"
            confidence = 0.5
        else:
            emotion = "neutral"
            confidence = 0.7
        
        return [{"emotion": emotion, "confidence": confidence}]


class TranscriptAligner:
    """전사 텍스트와 화자 분리 결과 정렬"""
    
    def __init__(self):
        logger.info("TranscriptAligner initialized")
    
    def align(
        self,
        transcription_segments: List[TranscriptionSegment],
        speaker_segments: List[SpeakerSegment]
    ) -> List[TranscriptionSegment]:
        """전사와 화자 정보 정렬"""
        if not speaker_segments:
            return transcription_segments
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            seg_mid = (trans_seg.start_time + trans_seg.end_time) / 2
            
            # 해당 시간의 화자 찾기
            speaker = self._find_speaker_at_time(seg_mid, speaker_segments)
            
            aligned_seg = TranscriptionSegment(
                start_time=trans_seg.start_time,
                end_time=trans_seg.end_time,
                text=trans_seg.text,
                confidence=trans_seg.confidence,
                language=trans_seg.language,
                speaker=speaker,
                words=trans_seg.words
            )
            aligned_segments.append(aligned_seg)
        
        return aligned_segments
    
    def _find_speaker_at_time(
        self,
        time: float,
        speaker_segments: List[SpeakerSegment]
    ) -> Optional[str]:
        """특정 시간의 화자 찾기"""
        for seg in speaker_segments:
            if seg.start_time <= time <= seg.end_time:
                return seg.speaker_label
        return None


class AudioProcessor:
    """통합 음성 처리 파이프라인"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        
        # 모듈 초기화
        self.preprocessor = AudioPreprocessor()
        self.recognizer = SpeechRecognizer(
            model_size=self.config.get("whisper_model", "large-v3"),
            device=self.device
        )
        self.diarizer = SpeakerDiarizer(device=self.device)
        self.code_switching_detector = CodeSwitchingDetector()
        self.emotion_analyzer = EmotionAnalyzer(device=self.device)
        self.aligner = TranscriptAligner()
        
        self.models_loaded = False
        logger.info(f"AudioProcessor initialized on {self.device}")
    
    def load_models(self, hf_token: Optional[str] = None):
        """모든 모델 로드"""
        try:
            self.recognizer.load_model()
            self.diarizer.load_model(auth_token=hf_token)
            self.emotion_analyzer.load_model()
            
            self.models_loaded = True
            logger.info("All audio models loaded")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def process(
        self,
        audio_path: str,
        language: Optional[str] = None,
        diarize: bool = True,
        detect_emotion: bool = False,
        preprocess: bool = True
    ) -> TranscriptionResult:
        """음성 처리"""
        import time
        start_time = time.time()
        
        # 메타데이터
        metadata = self.preprocessor.get_metadata(audio_path)
        logger.info(f"Audio: {metadata.duration:.1f}s, {metadata.sample_rate}Hz")
        
        # 전처리 (필요시)
        processed_path = audio_path
        if preprocess:
            processed_path = self._preprocess_audio(audio_path)
        
        # 음성 인식
        transcription_segments = self.recognizer.transcribe(
            processed_path,
            language=language,
            word_timestamps=True
        )
        
        # 화자 분리
        speaker_segments = []
        if diarize and metadata.duration > 10:  # 10초 이상인 경우만
            speaker_segments = self.diarizer.diarize(processed_path)
        
        # 전사와 화자 정렬
        if speaker_segments:
            transcription_segments = self.aligner.align(
                transcription_segments,
                speaker_segments
            )
        
        # 언어 감지
        detected_languages = list(set(seg.language for seg in transcription_segments))
        
        # 코드 스위칭 감지
        code_switches = self.code_switching_detector.detect(transcription_segments)
        if code_switches:
            logger.info(f"Detected {len(code_switches)} code-switching points")
        
        # 감정 분석 (선택적)
        if detect_emotion:
            emotions = self.emotion_analyzer.analyze(processed_path)
            logger.info(f"Primary emotion: {emotions[0] if emotions else 'unknown'}")
        
        # 전체 텍스트 조합
        full_text = self._combine_segments(transcription_segments)
        
        # 화자 목록
        speakers = list(set(seg.speaker for seg in transcription_segments if seg.speaker))
        
        # 평균 신뢰도
        avg_confidence = np.mean([seg.confidence for seg in transcription_segments]) if transcription_segments else 0.0
        
        # 단어 수
        word_count = len(full_text.split())
        
        # 임시 파일 정리
        if processed_path != audio_path and os.path.exists(processed_path):
            os.remove(processed_path)
        
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            full_text=full_text,
            segments=transcription_segments,
            speakers=speakers,
            speaker_segments=speaker_segments,
            detected_languages=detected_languages,
            duration=metadata.duration,
            processing_time=processing_time,
            word_count=word_count,
            confidence=avg_confidence
        )
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """오디오 전처리 및 임시 파일 생성"""
        audio, sr = self.preprocessor.load_audio(audio_path)
        processed = self.preprocessor.preprocess(
            audio, sr,
            denoise=True,
            normalize=True
        )
        
        # 임시 파일 저장
        temp_path = tempfile.mktemp(suffix=".wav")
        self.preprocessor.save_audio(processed, temp_path, sr)
        
        return temp_path
    
    def _combine_segments(self, segments: List[TranscriptionSegment]) -> str:
        """세그먼트 텍스트 조합"""
        texts = []
        current_speaker = None
        
        for seg in segments:
            if seg.speaker and seg.speaker != current_speaker:
                texts.append(f"\n[{seg.speaker}]")
                current_speaker = seg.speaker
            texts.append(seg.text)
        
        return " ".join(texts).strip()
    
    def batch_process(
        self,
        audio_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> List[TranscriptionResult]:
        """배치 처리"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, path in enumerate(audio_paths):
            logger.info(f"Processing {i+1}/{len(audio_paths)}: {path}")
            
            try:
                result = self.process(path, **kwargs)
                results.append(result)
                
                # 결과 저장
                self._save_result(result, path, output_dir)
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        return results
    
    def _save_result(self, result: TranscriptionResult, audio_path: str, output_dir: str):
        """결과 저장"""
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 텍스트 저장
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.full_text)
        
        # JSON 저장
        json_path = os.path.join(output_dir, f"{base_name}.json")
        
        json_data = {
            "duration": result.duration,
            "word_count": result.word_count,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "speakers": result.speakers,
            "detected_languages": [l.value for l in result.detected_languages],
            "segments": [
                {
                    "start": seg.start_time,
                    "end": seg.end_time,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "language": seg.language.value,
                    "confidence": seg.confidence
                }
                for seg in result.segments
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # SRT 자막 저장
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        self._save_srt(result.segments, srt_path)
    
    def _save_srt(self, segments: List[TranscriptionSegment], path: str):
        """SRT 자막 저장"""
        lines = []
        
        for i, seg in enumerate(segments, 1):
            start_str = self._format_srt_time(seg.start_time)
            end_str = self._format_srt_time(seg.end_time)
            
            text = seg.text
            if seg.speaker:
                text = f"[{seg.speaker}] {text}"
            
            lines.append(str(i))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(text)
            lines.append("")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    
    def _format_srt_time(self, seconds: float) -> str:
        """SRT 타임스탬프 포맷"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# CLI 인터페이스
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diaspora Archive Audio Processing")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--language", default=None, help="Language code (ko, en, etc.)")
    parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--emotion", action="store_true", help="Analyze emotion")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--hf-token", help="HuggingFace token for pyannote")
    
    args = parser.parse_args()
    
    processor = AudioProcessor({
        "device": args.device,
        "whisper_model": args.model
    })
    processor.load_models(hf_token=args.hf_token)
    
    process_options = {
        "language": args.language,
        "diarize": not args.no_diarize,
        "detect_emotion": args.emotion,
        "preprocess": not args.no_preprocess
    }
    
    if os.path.isdir(args.input):
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in audio_extensions
        ]
        output_dir = args.output or os.path.join(args.input, "transcripts")
        results = processor.batch_process(audio_paths, output_dir, **process_options)
        print(f"Processed {len(results)} audio files")
    else:
        result = processor.process(args.input, **process_options)
        
        print(f"\n=== Audio Processing Result ===")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Word count: {result.word_count}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Speakers: {', '.join(result.speakers) if result.speakers else 'Unknown'}")
        print(f"Languages: {', '.join(l.value for l in result.detected_languages)}")
        print(f"Processing time: {result.processing_time:.1f}s")
        print(f"\n--- Transcript ---")
        print(result.full_text[:1000] + "..." if len(result.full_text) > 1000 else result.full_text)
        
        if args.output:
            processor._save_result(result, args.input, args.output)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
