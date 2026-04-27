"""
src/modules/video_info.py
영상 메타데이터 추출 및 키프레임 추출
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np


class VideoInfoModule:
    """영상 메타데이터 및 키프레임 추출."""
    
    def get_metadata(self, video_path: Union[str, Path]) -> Dict:
        """OpenCV로 영상 메타데이터 추출."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'success': False, 'error': '영상 열기 실패'}
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                'success': True,
                'fps': float(fps),
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': float(duration),
                'duration_formatted': self._format_duration(duration),
            }
        finally:
            cap.release()
    
    def _format_duration(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def extract_keyframes(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_frames: int = 10
    ) -> Dict:
        """uniform sampling으로 키프레임 추출."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'success': False, 'error': '영상 열기 실패'}
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            if frame_count <= 0:
                return {'success': False, 'error': '프레임 없음'}
            
            # uniform sampling 인덱스
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
            
            extracted = []
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp_sec = idx / fps
                output_path = output_dir / f"keyframe_{i:03d}_t{timestamp_sec:.2f}s.jpg"
                
                # 한글 경로 안전 저장
                success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if success:
                    encoded.tofile(str(output_path))
                    extracted.append({
                        'index': i,
                        'frame_number': int(idx),
                        'timestamp_seconds': float(timestamp_sec),
                        'path': str(output_path),
                    })
            
            return {
                'success': True,
                'num_extracted': len(extracted),
                'keyframes': extracted,
            }
        finally:
            cap.release()
    
    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Dict:
        """ffmpeg로 음성 추출 (16kHz mono WAV)."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',  # 영상 트랙 제거
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz
                '-ac', '1',  # mono
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0 and output_path.exists():
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'sample_rate': 16000,
                    'channels': 1,
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr[-500:] if result.stderr else 'ffmpeg 실패'
                }
        except FileNotFoundError:
            return {'success': False, 'error': 'ffmpeg가 설치되지 않음'}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'ffmpeg 타임아웃'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
