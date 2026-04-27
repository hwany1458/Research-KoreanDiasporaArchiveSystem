"""
src/modules/diarization.py
화자 분리 모듈 (pyannote.audio 기반)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


class DiarizationModule:
    """화자 분리 (pyannote.audio 기반)."""
    
    DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
    
    def __init__(
        self,
        device: str = 'cuda',
        hf_token: Optional[str] = None
    ):
        self.pipeline = None
        self.device = device
        
        if not PYANNOTE_AVAILABLE:
            print("  ⚠ pyannote.audio 미설치, 화자 분리 비활성화")
            return
        
        if not hf_token:
            print("  ⚠ HuggingFace 토큰 없음 — pyannote.audio 사용 불가")
            print("     (https://huggingface.co/pyannote/speaker-diarization-3.1 동의 후 토큰 필요)")
            return
        
        try:
            print(f"  pyannote.audio 모델 로딩 중...")
            self.pipeline = PyannotePipeline.from_pretrained(
                self.DEFAULT_MODEL,
                use_auth_token=hf_token
            )
            if device == 'cuda' and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device('cuda'))
        except Exception as e:
            print(f"  ⚠ pyannote.audio 초기화 실패: {e}")
            self.pipeline = None
    
    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        화자 분리 실행.
        
        Returns:
            {
                'success': bool,
                'num_speakers': int,
                'segments': [{'start', 'end', 'speaker'}, ...]
            }
        """
        if self.pipeline is None:
            return {'success': False, 'error': 'pyannote 미사용 가능'}
        
        try:
            kwargs = {}
            if num_speakers is not None:
                kwargs['num_speakers'] = num_speakers
            
            diarization = self.pipeline(str(audio_path), **kwargs)
            
            segments = []
            speakers_set = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'speaker': speaker,
                })
                speakers_set.add(speaker)
            
            return {
                'success': True,
                'num_speakers': len(speakers_set),
                'speakers': sorted(speakers_set),
                'num_segments': len(segments),
                'segments': segments,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def merge_with_transcription(
        self,
        diarization_segments: List[Dict],
        transcription_segments: List[Dict]
    ) -> List[Dict]:
        """
        화자 분리 결과와 음성 전사 결과 병합.
        
        각 전사 세그먼트에 가장 많이 겹치는 화자를 할당.
        """
        merged = []
        
        for trans_seg in transcription_segments:
            t_start = trans_seg['start']
            t_end = trans_seg['end']
            
            # 가장 많이 겹치는 화자 찾기
            best_speaker = 'UNKNOWN'
            best_overlap = 0.0
            
            for diar_seg in diarization_segments:
                overlap_start = max(t_start, diar_seg['start'])
                overlap_end = min(t_end, diar_seg['end'])
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg['speaker']
            
            merged.append({
                **trans_seg,
                'speaker': best_speaker,
            })
        
        return merged
