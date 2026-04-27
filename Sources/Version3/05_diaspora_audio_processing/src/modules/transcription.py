"""
src/modules/transcription.py
음성 인식 모듈 (Whisper / Faster-Whisper)
"""

from pathlib import Path
from typing import Dict, Optional, Union

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class TranscriptionModule:
    """음성 → 텍스트 (Whisper 기반)."""
    
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'cuda',
        use_faster_whisper: bool = True
    ):
        self.model_size = model_size
        self.device = device
        self.backend = None
        self.model = None
        
        if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
            try:
                compute_type = 'float16' if device == 'cuda' else 'int8'
                print(f"  Faster-Whisper 로딩 ({model_size}, {device})...")
                self.model = WhisperModel(
                    model_size,
                    device=device if device == 'cuda' else 'cpu',
                    compute_type=compute_type
                )
                self.backend = 'faster_whisper'
                return
            except Exception as e:
                print(f"  Faster-Whisper 실패: {e}, fallback 시도")
        
        if WHISPER_AVAILABLE:
            try:
                print(f"  OpenAI Whisper 로딩 ({model_size})...")
                self.model = whisper.load_model(model_size, device=device)
                self.backend = 'whisper'
                return
            except Exception as e:
                print(f"  Whisper 로딩 실패: {e}")
        
        if self.model is None:
            raise ImportError(
                "Whisper 라이브러리 필요: pip install openai-whisper 또는 faster-whisper"
            )
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        try:
            if self.backend == 'faster_whisper':
                return self._transcribe_faster(audio_path, language)
            else:
                return self._transcribe_openai(audio_path, language)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _transcribe_faster(self, audio_path, language):
        segments, info = self.model.transcribe(
            str(audio_path), language=language, beam_size=5
        )
        segments_list = []
        full_text_parts = []
        for seg in segments:
            segments_list.append({
                'start': float(seg.start),
                'end': float(seg.end),
                'text': seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())
        
        return {
            'success': True,
            'backend': 'faster_whisper',
            'language': info.language,
            'language_probability': float(info.language_probability),
            'duration': float(info.duration),
            'full_text': ' '.join(full_text_parts),
            'segments': segments_list,
        }
    
    def _transcribe_openai(self, audio_path, language):
        result = self.model.transcribe(str(audio_path), language=language)
        segments_list = []
        for seg in result.get('segments', []):
            segments_list.append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
            })
        return {
            'success': True,
            'backend': 'whisper',
            'language': result.get('language'),
            'full_text': result.get('text', ''),
            'segments': segments_list,
        }
