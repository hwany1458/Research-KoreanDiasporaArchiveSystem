"""
src/pipeline.py
음성 처리 파이프라인
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.modules.transcription import TranscriptionModule


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}


class AudioProcessingPipeline:
    """음성 처리 통합 파이프라인."""
    
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'cuda',
        verbose: bool = False
    ):
        self.device = device
        self.model_size = model_size
        self.verbose = verbose
        self._transcriber: Optional[TranscriptionModule] = None
    
    @property
    def transcriber(self) -> TranscriptionModule:
        if self._transcriber is None:
            self._transcriber = TranscriptionModule(
                model_size=self.model_size,
                device=self.device
            )
        return self._transcriber
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        language: Optional[str] = None
    ) -> Dict:
        """단일 음성 파일 처리."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(input_path),
            'success': False,
        }
        
        if not input_path.exists():
            result['error'] = '파일 없음'
            return result
        
        if self.verbose:
            print(f"\n[전사] {input_path.name}")
        
        transcription = self.transcriber.transcribe(input_path, language=language)
        result['transcription'] = transcription
        
        if not transcription.get('success'):
            return result
        
        srt_path = output_dir / f"{input_path.stem}.srt"
        self._save_srt(transcription.get('segments', []), srt_path)
        result['srt_path'] = str(srt_path)
        
        txt_path = output_dir / f"{input_path.stem}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription.get('full_text', ''))
        result['txt_path'] = str(txt_path)
        
        result['success'] = True
        result['finished_at'] = datetime.now().isoformat()
        
        if self.verbose:
            print(f"  언어: {transcription.get('language', 'N/A')}")
            print(f"  세그먼트: {len(transcription.get('segments', []))}개")
        
        return result
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        language: Optional[str] = None
    ) -> List[Dict]:
        """디렉토리 일괄 처리."""
        files = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ])
        
        results = []
        for i, file_path in enumerate(files, 1):
            if self.verbose:
                print(f"\n[{i}/{len(files)}] {file_path.name}")
            try:
                result = self.process_file(file_path, output_dir, language)
                results.append(result)
            except Exception as e:
                results.append({
                    'input_path': str(file_path),
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r.get('success'))
        print(f"\n총 {len(files)}개 중 {success_count}개 성공")
        
        return results
    
    @staticmethod
    def _save_srt(segments: List[Dict], srt_path: Path):
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = AudioProcessingPipeline._format_srt_time(seg['start'])
                end = AudioProcessingPipeline._format_srt_time(seg['end'])
                f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
