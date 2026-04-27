"""
src/modules/scene_detection.py
장면 분할 모듈 (PySceneDetect 기반, fallback 포함)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

try:
    from scenedetect import detect, ContentDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


class SceneDetectionModule:
    """장면 분할 (PySceneDetect 우선, frame-diff fallback)."""
    
    def detect_scenes(
        self,
        video_path: Union[str, Path],
        threshold: float = 27.0
    ) -> Dict:
        """
        장면 분할 실행.
        
        Returns:
            {
                'success': bool,
                'method': 'pyscenedetect' or 'frame_diff',
                'num_scenes': int,
                'scenes': [{'start': float, 'end': float, 'duration': float}, ...]
            }
        """
        # 1차: PySceneDetect
        if PYSCENEDETECT_AVAILABLE:
            try:
                return self._detect_pyscenedetect(video_path, threshold)
            except Exception as e:
                print(f"  PySceneDetect 실패: {e}, fallback 시도")
        
        # 2차: frame-diff fallback
        return self._detect_frame_diff(video_path)
    
    def _detect_pyscenedetect(
        self,
        video_path: Union[str, Path],
        threshold: float
    ) -> Dict:
        """PySceneDetect로 장면 분할."""
        scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
        
        scenes = []
        for start, end in scene_list:
            scenes.append({
                'start_seconds': start.get_seconds(),
                'end_seconds': end.get_seconds(),
                'duration_seconds': (end - start).get_seconds(),
            })
        
        return {
            'success': True,
            'method': 'pyscenedetect',
            'num_scenes': len(scenes),
            'scenes': scenes,
        }
    
    def _detect_frame_diff(
        self,
        video_path: Union[str, Path],
        threshold: float = 0.3
    ) -> Dict:
        """HSV 히스토그램 차이 기반 fallback."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'success': False, 'error': '영상 열기 실패'}
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            scene_starts = [0.0]
            prev_hist = None
            frame_idx = 0
            
            # 5프레임마다 샘플링 (속도)
            sample_step = 5
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_step == 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                    cv2.normalize(hist, hist)
                    
                    if prev_hist is not None:
                        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                        if diff > threshold:
                            scene_starts.append(frame_idx / fps)
                    
                    prev_hist = hist
                
                frame_idx += 1
            
            # 장면 구간 생성
            scenes = []
            for i, start in enumerate(scene_starts):
                end = scene_starts[i+1] if i+1 < len(scene_starts) else (frame_idx / fps)
                scenes.append({
                    'start_seconds': float(start),
                    'end_seconds': float(end),
                    'duration_seconds': float(end - start),
                })
            
            return {
                'success': True,
                'method': 'frame_diff',
                'num_scenes': len(scenes),
                'scenes': scenes,
            }
        finally:
            cap.release()
