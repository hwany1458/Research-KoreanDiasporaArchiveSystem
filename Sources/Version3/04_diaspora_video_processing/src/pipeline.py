"""
src/pipeline.py
영상 처리 파이프라인
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.modules.video_info import VideoInfoModule
from src.modules.scene_detection import SceneDetectionModule


class VideoProcessingPipeline:
    """영상 처리 통합 파이프라인."""
    
    def __init__(self, device: str = 'cuda', verbose: bool = False):
        self.device = device
        self.verbose = verbose
        
        self.video_info = VideoInfoModule()
        self.scene_detector = SceneDetectionModule()
        # Whisper 모듈은 필요 시 lazy 로딩
        self._whisper = None
    
    @property
    def whisper(self):
        """Whisper 모듈 lazy 로딩."""
        if self._whisper is None:
            try:
                import whisper
                if self.verbose:
                    print("  Whisper 모델 로딩 중 (base)...")
                self._whisper = whisper.load_model("base")
            except ImportError:
                print("  ⚠ openai-whisper 미설치, 자막 생성 비활성화")
                self._whisper = False
        return self._whisper
    
    def process_video(
        self,
        input_path: Path,
        output_dir: Path,
        num_keyframes: int = 10,
        do_scene_detection: bool = True,
        do_audio_extraction: bool = True,
        do_subtitle: bool = True,
    ) -> Dict:
        """영상 통합 처리."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(input_path),
            'success': False,
        }
        
        if not input_path.exists():
            result['error'] = '파일 없음'
            return result
        
        # 출력 하위 디렉토리
        keyframe_dir = output_dir / f"{input_path.stem}_keyframes"
        audio_path = output_dir / f"{input_path.stem}_audio.wav"
        subtitle_path = output_dir / f"{input_path.stem}_subtitle.srt"
        
        # 1. 메타데이터
        if self.verbose:
            print("\n[1/5] 메타데이터 추출...")
        result['metadata'] = self.video_info.get_metadata(input_path)
        
        # 2. 키프레임 추출
        if self.verbose:
            print(f"[2/5] 키프레임 {num_keyframes}장 추출...")
        result['keyframes'] = self.video_info.extract_keyframes(
            input_path, keyframe_dir, num_keyframes
        )
        
        # 3. 장면 분할
        if do_scene_detection:
            if self.verbose:
                print("[3/5] 장면 분할...")
            result['scenes'] = self.scene_detector.detect_scenes(input_path)
        
        # 4. 음성 추출
        if do_audio_extraction:
            if self.verbose:
                print("[4/5] 음성 추출 (ffmpeg)...")
            result['audio'] = self.video_info.extract_audio(input_path, audio_path)
        
        # 5. 자막 생성 (Whisper)
        if do_subtitle and result.get('audio', {}).get('success'):
            if self.verbose:
                print("[5/5] 자막 생성 (Whisper)...")
            result['subtitle'] = self._generate_subtitle(audio_path, subtitle_path)
        
        result['success'] = True
        result['finished_at'] = datetime.now().isoformat()
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _generate_subtitle(self, audio_path: Path, subtitle_path: Path) -> Dict:
        """Whisper로 자막 생성."""
        if not self.whisper:
            return {'success': False, 'error': 'Whisper 미사용 가능'}
        
        try:
            transcribe_result = self.whisper.transcribe(
                str(audio_path), language='ko'
            )
            
            # SRT 형식 저장
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(transcribe_result.get('segments', []), 1):
                    start = self._format_srt_time(seg['start'])
                    end = self._format_srt_time(seg['end'])
                    f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
            
            return {
                'success': True,
                'subtitle_path': str(subtitle_path),
                'num_segments': len(transcribe_result.get('segments', [])),
                'detected_language': transcribe_result.get('language'),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    def _print_summary(self, result: Dict):
        print("\n=== 영상 처리 완료 ===")
        meta = result.get('metadata', {})
        if meta.get('success'):
            print(f"  길이: {meta.get('duration_formatted', 'N/A')}")
            print(f"  해상도: {meta.get('width')}×{meta.get('height')}")
        if result.get('keyframes', {}).get('success'):
            print(f"  키프레임: {result['keyframes']['num_extracted']}장")
        if result.get('scenes', {}).get('success'):
            print(f"  장면: {result['scenes']['num_scenes']}개 ({result['scenes']['method']})")
        if result.get('audio', {}).get('success'):
            print(f"  음성 추출: {result['audio']['output_path']}")
        if result.get('subtitle', {}).get('success'):
            print(f"  자막: {result['subtitle']['num_segments']}개 세그먼트")
