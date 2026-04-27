"""
src/pipeline.py
구술 처리 파이프라인 (Whisper + 화자 분리 + 병합)
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.modules.diarization import DiarizationModule


class OralProcessingPipeline:
    """구술 처리 통합 파이프라인."""
    
    def __init__(
        self,
        whisper_model: str = 'base',
        device: str = 'cuda',
        hf_token: Optional[str] = None,
        verbose: bool = False
    ):
        self.whisper_model_size = whisper_model
        self.device = device
        self.hf_token = hf_token
        self.verbose = verbose
        
        self._whisper = None
        self._diarizer: Optional[DiarizationModule] = None
    
    @property
    def whisper(self):
        """Whisper 모듈 lazy 로딩."""
        if self._whisper is None:
            try:
                import whisper
                if self.verbose:
                    print(f"  Whisper 모델 로딩 ({self.whisper_model_size})...")
                self._whisper = whisper.load_model(self.whisper_model_size)
            except ImportError:
                raise ImportError(
                    "openai-whisper가 필요합니다: pip install openai-whisper"
                )
        return self._whisper
    
    @property
    def diarizer(self) -> DiarizationModule:
        if self._diarizer is None:
            self._diarizer = DiarizationModule(
                device=self.device,
                hf_token=self.hf_token
            )
        return self._diarizer
    
    def process_interview(
        self,
        input_path: Path,
        output_dir: Path,
        num_speakers: Optional[int] = None,
        language: str = 'ko',
        do_diarization: bool = True,
    ) -> Dict:
        """구술 인터뷰 통합 처리."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(input_path),
            'success': False,
        }
        
        if not input_path.exists():
            result['error'] = '파일 없음'
            return result
        
        # 1. Whisper 전사
        if self.verbose:
            print("\n[1/3] 음성 전사 (Whisper)...")
        try:
            transcribe_result = self.whisper.transcribe(
                str(input_path), language=language
            )
            result['transcription'] = {
                'success': True,
                'language': transcribe_result.get('language'),
                'full_text': transcribe_result.get('text', ''),
                'segments': [
                    {
                        'start': float(s['start']),
                        'end': float(s['end']),
                        'text': s['text'].strip()
                    }
                    for s in transcribe_result.get('segments', [])
                ]
            }
        except Exception as e:
            result['transcription'] = {'success': False, 'error': str(e)}
            return result
        
        # 2. 화자 분리
        if do_diarization:
            if self.verbose:
                print("[2/3] 화자 분리 (pyannote.audio)...")
            diarization_result = self.diarizer.diarize(
                input_path, num_speakers=num_speakers
            )
            result['diarization'] = diarization_result
            
            # 3. 전사 + 화자 병합
            if diarization_result.get('success'):
                if self.verbose:
                    print("[3/3] 전사 + 화자 정보 병합...")
                merged = self.diarizer.merge_with_transcription(
                    diarization_result['segments'],
                    result['transcription']['segments']
                )
                result['merged_segments'] = merged
                
                # 화자별 텍스트 저장
                self._save_by_speaker(merged, input_path, output_dir)
                self._save_dialogue_format(merged, input_path, output_dir)
        
        # 일반 SRT
        srt_path = output_dir / f"{input_path.stem}.srt"
        self._save_srt(result['transcription']['segments'], srt_path)
        result['srt_path'] = str(srt_path)
        
        result['success'] = True
        result['finished_at'] = datetime.now().isoformat()
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _save_by_speaker(self, merged: List[Dict], input_path: Path, output_dir: Path):
        """화자별로 텍스트 정리 저장."""
        by_speaker: Dict[str, List[str]] = {}
        for seg in merged:
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            if text:
                by_speaker.setdefault(speaker, []).append(text)
        
        out_path = output_dir / f"{input_path.stem}_by_speaker.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            for speaker, texts in sorted(by_speaker.items()):
                f.write(f"\n=== {speaker} ===\n")
                for text in texts:
                    f.write(f"{text}\n")
    
    def _save_dialogue_format(self, merged: List[Dict], input_path: Path, output_dir: Path):
        """대화 형식 저장 (시간 순서)."""
        out_path = output_dir / f"{input_path.stem}_dialogue.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            for seg in merged:
                speaker = seg.get('speaker', 'UNKNOWN')
                start = seg['start']
                text = seg['text'].strip()
                f.write(f"[{self._format_time(start)}] {speaker}: {text}\n")
    
    def _save_srt(self, segments: List[Dict], srt_path: Path):
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = self._format_srt_time(seg['start'])
                end = self._format_srt_time(seg['end'])
                f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    def _print_summary(self, result: Dict):
        print("\n=== 구술 처리 완료 ===")
        trans = result.get('transcription', {})
        if trans.get('success'):
            print(f"  세그먼트: {len(trans.get('segments', []))}개")
        diar = result.get('diarization', {})
        if diar.get('success'):
            print(f"  화자 수: {diar.get('num_speakers', 0)}명")
            print(f"  화자: {', '.join(diar.get('speakers', []))}")
