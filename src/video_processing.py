"""
동영상 처리 모듈 (Video Processing Module)

한인 디아스포라 기록유산의 동영상 복원, 분석, 처리를 위한 AI 기반 모듈

주요 기능:
- 영상 품질 향상 (Video Enhancement): BasicVSR++, Real-ESRGAN Video
- 장면 분할 (Scene Detection): PySceneDetect, TransNetV2
- 핵심 프레임 추출 (Keyframe Extraction)
- 음성 트랙 분리 (Audio Extraction)
- 자막 생성 (Subtitle Generation)
- 영상 안정화 (Video Stabilization)
- 컬러 보정 (Color Correction)

Author: Diaspora Archive Project
License: MIT
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import json
import shutil

import numpy as np
import cv2
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """비디오 포맷"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    VHS = "vhs"
    DVD = "dvd"
    UNKNOWN = "unknown"


class VideoQuality(Enum):
    """비디오 품질"""
    SD = "sd"           # 480p 이하
    HD = "hd"           # 720p
    FULL_HD = "full_hd"  # 1080p
    QHD = "qhd"         # 1440p
    UHD = "uhd"         # 4K


@dataclass
class VideoMetadata:
    """비디오 메타데이터"""
    duration: float          # 초 단위
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    format: VideoFormat
    quality: VideoQuality
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_channels: int = 0
    audio_sample_rate: int = 0
    file_size: int = 0
    creation_date: Optional[str] = None
    frame_count: int = 0


@dataclass
class SceneInfo:
    """장면 정보"""
    scene_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration: float
    thumbnail_path: Optional[str] = None
    description: str = ""
    keyframes: List[int] = field(default_factory=list)


@dataclass
class SubtitleEntry:
    """자막 항목"""
    index: int
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 0.0


@dataclass
class VideoProcessingResult:
    """비디오 처리 결과"""
    input_path: str
    output_path: Optional[str]
    metadata: VideoMetadata
    scenes: List[SceneInfo]
    subtitles: List[SubtitleEntry]
    extracted_audio_path: Optional[str]
    keyframe_paths: List[str]
    processing_time: float
    enhancement_applied: bool = False
    stabilization_applied: bool = False


class VideoMetadataExtractor:
    """비디오 메타데이터 추출"""
    
    def __init__(self):
        self.ffprobe_path = self._find_ffprobe()
        logger.info(f"VideoMetadataExtractor initialized (ffprobe: {self.ffprobe_path})")
    
    def _find_ffprobe(self) -> Optional[str]:
        """ffprobe 경로 찾기"""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            return ffprobe
        
        # 일반적인 경로들
        common_paths = [
            "/usr/bin/ffprobe",
            "/usr/local/bin/ffprobe",
            "C:\\ffmpeg\\bin\\ffprobe.exe"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def extract(self, video_path: str) -> VideoMetadata:
        """메타데이터 추출"""
        if self.ffprobe_path:
            return self._extract_with_ffprobe(video_path)
        else:
            return self._extract_with_opencv(video_path)
    
    def _extract_with_ffprobe(self, video_path: str) -> VideoMetadata:
        """ffprobe로 메타데이터 추출"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            
            # 비디오 스트림 찾기
            video_stream = None
            audio_stream = None
            
            for stream in data.get("streams", []):
                if stream["codec_type"] == "video" and video_stream is None:
                    video_stream = stream
                elif stream["codec_type"] == "audio" and audio_stream is None:
                    audio_stream = stream
            
            if video_stream is None:
                raise ValueError("No video stream found")
            
            # FPS 파싱
            fps_str = video_stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            
            # 포맷 정보
            format_info = data.get("format", {})
            
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            duration = float(format_info.get("duration", 0))
            
            return VideoMetadata(
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                codec=video_stream.get("codec_name", "unknown"),
                bitrate=int(format_info.get("bit_rate", 0)),
                format=self._detect_format(video_path),
                quality=self._detect_quality(width, height),
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                audio_channels=int(audio_stream.get("channels", 0)) if audio_stream else 0,
                audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
                file_size=int(format_info.get("size", 0)),
                frame_count=int(video_stream.get("nb_frames", duration * fps))
            )
            
        except Exception as e:
            logger.error(f"ffprobe error: {e}")
            return self._extract_with_opencv(video_path)
    
    def _extract_with_opencv(self, video_path: str) -> VideoMetadata:
        """OpenCV로 메타데이터 추출"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoMetadata(
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            codec="unknown",
            bitrate=0,
            format=self._detect_format(video_path),
            quality=self._detect_quality(width, height),
            has_audio=True,  # 기본값
            frame_count=frame_count
        )
    
    def _detect_format(self, video_path: str) -> VideoFormat:
        """비디오 포맷 감지"""
        ext = os.path.splitext(video_path)[1].lower()
        format_map = {
            ".mp4": VideoFormat.MP4,
            ".avi": VideoFormat.AVI,
            ".mov": VideoFormat.MOV,
            ".mkv": VideoFormat.MKV,
            ".webm": VideoFormat.WEBM
        }
        return format_map.get(ext, VideoFormat.UNKNOWN)
    
    def _detect_quality(self, width: int, height: int) -> VideoQuality:
        """비디오 품질 감지"""
        max_dim = max(width, height)
        
        if max_dim >= 3840:
            return VideoQuality.UHD
        elif max_dim >= 2560:
            return VideoQuality.QHD
        elif max_dim >= 1920:
            return VideoQuality.FULL_HD
        elif max_dim >= 1280:
            return VideoQuality.HD
        else:
            return VideoQuality.SD


class SceneDetector:
    """장면 분할"""
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        logger.info(f"SceneDetector initialized (threshold: {threshold})")
    
    def detect_scenes(
        self,
        video_path: str,
        min_scene_length: float = 1.0
    ) -> List[SceneInfo]:
        """장면 감지"""
        try:
            # PySceneDetect 사용 시도
            return self._detect_with_pyscenedetect(video_path, min_scene_length)
        except ImportError:
            # 폴백: OpenCV 기반
            return self._detect_with_opencv(video_path, min_scene_length)
    
    def _detect_with_pyscenedetect(
        self,
        video_path: str,
        min_scene_length: float
    ) -> List[SceneInfo]:
        """PySceneDetect로 장면 감지"""
        from scenedetect import detect, ContentDetector, AdaptiveDetector
        
        # ContentDetector 사용
        scene_list = detect(
            video_path,
            ContentDetector(threshold=self.threshold, min_scene_len=int(min_scene_length * 30))
        )
        
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scene = SceneInfo(
                scene_id=i,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                start_frame=start.frame_num,
                end_frame=end.frame_num,
                duration=end.get_seconds() - start.get_seconds()
            )
            scenes.append(scene)
        
        return scenes
    
    def _detect_with_opencv(
        self,
        video_path: str,
        min_scene_length: float
    ) -> List[SceneInfo]:
        """OpenCV로 장면 감지 (히스토그램 비교)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frames = int(min_scene_length * fps)
        
        scenes = []
        prev_hist = None
        scene_start_frame = 0
        scene_start_time = 0.0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 히스토그램 계산
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                # 히스토그램 비교
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # 장면 전환 감지
                if diff < 0.5 and (frame_idx - scene_start_frame) >= min_frames:
                    current_time = frame_idx / fps
                    
                    scene = SceneInfo(
                        scene_id=len(scenes),
                        start_time=scene_start_time,
                        end_time=current_time,
                        start_frame=scene_start_frame,
                        end_frame=frame_idx,
                        duration=current_time - scene_start_time
                    )
                    scenes.append(scene)
                    
                    scene_start_frame = frame_idx
                    scene_start_time = current_time
            
            prev_hist = hist.copy()
            frame_idx += 1
        
        # 마지막 장면 추가
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time = total_frames / fps
        
        if frame_idx > scene_start_frame:
            scene = SceneInfo(
                scene_id=len(scenes),
                start_time=scene_start_time,
                end_time=total_time,
                start_frame=scene_start_frame,
                end_frame=total_frames,
                duration=total_time - scene_start_time
            )
            scenes.append(scene)
        
        cap.release()
        return scenes


class KeyframeExtractor:
    """핵심 프레임 추출"""
    
    def __init__(self):
        logger.info("KeyframeExtractor initialized")
    
    def extract_keyframes(
        self,
        video_path: str,
        scenes: List[SceneInfo],
        output_dir: str,
        frames_per_scene: int = 3
    ) -> List[str]:
        """장면별 핵심 프레임 추출"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframe_paths = []
        
        for scene in scenes:
            # 장면 내에서 균등하게 프레임 선택
            scene_frames = []
            duration = scene.end_frame - scene.start_frame
            
            for i in range(frames_per_scene):
                frame_idx = scene.start_frame + int(duration * (i + 0.5) / frames_per_scene)
                scene_frames.append(frame_idx)
            
            # 프레임 추출 및 저장
            for frame_idx in scene_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    filename = f"scene_{scene.scene_id:03d}_frame_{frame_idx:06d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
                    keyframe_paths.append(filepath)
            
            scene.keyframes = scene_frames
        
        cap.release()
        return keyframe_paths
    
    def extract_uniform_keyframes(
        self,
        video_path: str,
        interval_seconds: float = 5.0,
        output_dir: str = None
    ) -> List[Tuple[int, np.ndarray]]:
        """균일 간격으로 프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = int(interval_seconds * fps)
        
        keyframes = []
        
        for frame_idx in range(0, total_frames, interval_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                keyframes.append((frame_idx, frame))
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"frame_{frame_idx:06d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
        
        cap.release()
        return keyframes


class AudioExtractor:
    """오디오 추출"""
    
    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg")
        logger.info(f"AudioExtractor initialized (ffmpeg: {self.ffmpeg_path})")
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = "wav",
        sample_rate: int = 16000
    ) -> str:
        """비디오에서 오디오 추출"""
        if self.ffmpeg_path is None:
            raise RuntimeError("FFmpeg not found")
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vn",  # 비디오 제외
            "-acodec", "pcm_s16le" if format == "wav" else "libmp3lame",
            "-ar", str(sample_rate),
            "-ac", "1",  # 모노
            "-y",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f"Audio extracted to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e}")
            raise


class VideoEnhancer:
    """비디오 품질 향상"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        logger.info(f"VideoEnhancer initialized on {device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """품질 향상 모델 로드"""
        try:
            # Real-ESRGAN 비디오 모델 로드 시도
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2
            )
            
            self.model = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            logger.info("Video enhancement model loaded")
            
        except ImportError:
            logger.warning("Real-ESRGAN not available")
            self.model = None
    
    def enhance_video(
        self,
        input_path: str,
        output_path: str,
        scale: int = 2,
        denoise: bool = True
    ) -> str:
        """비디오 품질 향상"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # 비디오 속성
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 설정
        out_width = width * scale
        out_height = height * scale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 임시 파일
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))
        
        logger.info(f"Enhancing video: {width}x{height} -> {out_width}x{out_height}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 향상
            enhanced = self._enhance_frame(frame, scale, denoise)
            out.write(enhanced)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Progress: {frame_idx}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # 오디오 추가
        self._add_audio(input_path, temp_output, output_path)
        
        # 임시 파일 삭제
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        logger.info(f"Enhanced video saved to {output_path}")
        return output_path
    
    def _enhance_frame(
        self,
        frame: np.ndarray,
        scale: int,
        denoise: bool
    ) -> np.ndarray:
        """단일 프레임 향상"""
        # 노이즈 제거
        if denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
        
        # 업스케일링
        if self.model is not None:
            try:
                enhanced, _ = self.model.enhance(frame, outscale=scale)
                return enhanced
            except Exception as e:
                logger.error(f"Enhancement error: {e}")
        
        # 폴백: Lanczos 업스케일링
        height, width = frame.shape[:2]
        return cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)
    
    def _add_audio(self, original_video: str, enhanced_video: str, output_path: str):
        """향상된 비디오에 원본 오디오 추가"""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            shutil.move(enhanced_video, output_path)
            return
        
        cmd = [
            ffmpeg_path,
            "-i", enhanced_video,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-y",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError:
            shutil.move(enhanced_video, output_path)


class VideoStabilizer:
    """비디오 안정화"""
    
    def __init__(self):
        logger.info("VideoStabilizer initialized")
    
    def stabilize(
        self,
        input_path: str,
        output_path: str,
        smoothing_radius: int = 30
    ) -> str:
        """비디오 안정화"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 첫 번째 패스: 변환 행렬 수집
        transforms = []
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # 특징점 추출 및 추적
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_gray, maxCorners=200, qualityLevel=0.01,
                    minDistance=30, blockSize=3
                )
                
                if prev_pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_pts, None
                    )
                    
                    # 유효한 포인트만 선택
                    idx = np.where(status == 1)[0]
                    prev_pts = prev_pts[idx]
                    curr_pts = curr_pts[idx]
                    
                    # 변환 행렬 추정
                    if len(prev_pts) >= 4:
                        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                        if m is not None:
                            dx = m[0, 2]
                            dy = m[1, 2]
                            da = np.arctan2(m[1, 0], m[0, 0])
                            transforms.append([dx, dy, da])
                        else:
                            transforms.append([0, 0, 0])
                    else:
                        transforms.append([0, 0, 0])
                else:
                    transforms.append([0, 0, 0])
            
            prev_gray = gray
        
        cap.release()
        
        if not transforms:
            shutil.copy(input_path, output_path)
            return output_path
        
        transforms = np.array(transforms)
        
        # 궤적 계산
        trajectory = np.cumsum(transforms, axis=0)
        
        # 궤적 스무딩
        smoothed_trajectory = self._smooth_trajectory(trajectory, smoothing_radius)
        
        # 차이 계산
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference
        
        # 두 번째 패스: 안정화 적용
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, transform in enumerate(transforms_smooth):
            ret, frame = cap.read()
            if not ret:
                break
            
            dx, dy, da = transform
            
            # 변환 행렬 생성
            m = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy]
            ], dtype=np.float32)
            
            # 변환 적용
            stabilized = cv2.warpAffine(frame, m, (width, height))
            out.write(stabilized)
        
        cap.release()
        out.release()
        
        logger.info(f"Stabilized video saved to {output_path}")
        return output_path
    
    def _smooth_trajectory(
        self,
        trajectory: np.ndarray,
        radius: int
    ) -> np.ndarray:
        """궤적 스무딩 (이동 평균)"""
        smoothed = np.copy(trajectory)
        
        for i in range(3):  # dx, dy, da
            smoothed[:, i] = self._moving_average(trajectory[:, i], radius)
        
        return smoothed
    
    def _moving_average(self, data: np.ndarray, radius: int) -> np.ndarray:
        """이동 평균 필터"""
        window_size = 2 * radius + 1
        kernel = np.ones(window_size) / window_size
        
        # 패딩
        padded = np.pad(data, (radius, radius), mode='edge')
        
        # 컨볼루션
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed


class VideoColorCorrector:
    """비디오 컬러 보정"""
    
    def __init__(self):
        logger.info("VideoColorCorrector initialized")
    
    def correct_colors(
        self,
        input_path: str,
        output_path: str,
        auto_white_balance: bool = True,
        enhance_contrast: bool = True,
        restore_faded: bool = True
    ) -> str:
        """컬러 보정"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            corrected = frame.copy()
            
            # 화이트 밸런스
            if auto_white_balance:
                corrected = self._auto_white_balance(corrected)
            
            # 색바램 복원
            if restore_faded:
                corrected = self._restore_faded_colors(corrected)
            
            # 콘트라스트 향상
            if enhance_contrast:
                corrected = self._enhance_contrast(corrected)
            
            out.write(corrected)
        
        cap.release()
        out.release()
        
        logger.info(f"Color corrected video saved to {output_path}")
        return output_path
    
    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """자동 화이트 밸런스"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    def _restore_faded_colors(self, image: np.ndarray) -> np.ndarray:
        """색바램 복원"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 채도 증가
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """콘트라스트 향상"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class SubtitleGenerator:
    """자막 생성"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.whisper_model = None
        logger.info(f"SubtitleGenerator initialized on {device}")
    
    def load_model(self, model_size: str = "base"):
        """Whisper 모델 로드"""
        try:
            import whisper
            self.whisper_model = whisper.load_model(model_size, device=self.device)
            logger.info(f"Whisper {model_size} model loaded")
        except ImportError:
            logger.warning("Whisper not available")
    
    def generate_subtitles(
        self,
        audio_path: str,
        language: str = "ko",
        output_format: str = "srt"
    ) -> Tuple[List[SubtitleEntry], str]:
        """자막 생성"""
        if self.whisper_model is None:
            logger.error("Whisper model not loaded")
            return [], ""
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                task="transcribe"
            )
            
            subtitles = []
            for i, segment in enumerate(result["segments"]):
                entry = SubtitleEntry(
                    index=i + 1,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob", 0)
                )
                subtitles.append(entry)
            
            # 자막 파일 생성
            srt_content = self._format_srt(subtitles)
            
            return subtitles, srt_content
            
        except Exception as e:
            logger.error(f"Subtitle generation error: {e}")
            return [], ""
    
    def _format_srt(self, subtitles: List[SubtitleEntry]) -> str:
        """SRT 포맷 생성"""
        lines = []
        
        for entry in subtitles:
            start_str = self._format_timestamp(entry.start_time)
            end_str = self._format_timestamp(entry.end_time)
            
            lines.append(str(entry.index))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(entry.text)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """타임스탬프 포맷 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def save_subtitles(self, subtitles: List[SubtitleEntry], output_path: str):
        """자막 파일 저장"""
        srt_content = self._format_srt(subtitles)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"Subtitles saved to {output_path}")


class VideoProcessor:
    """통합 비디오 처리 파이프라인"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        
        # 모듈 초기화
        self.metadata_extractor = VideoMetadataExtractor()
        self.scene_detector = SceneDetector()
        self.keyframe_extractor = KeyframeExtractor()
        self.audio_extractor = AudioExtractor()
        self.enhancer = VideoEnhancer(self.device)
        self.stabilizer = VideoStabilizer()
        self.color_corrector = VideoColorCorrector()
        self.subtitle_generator = SubtitleGenerator(self.device)
        
        self.models_loaded = False
        logger.info(f"VideoProcessor initialized on {self.device}")
    
    def load_models(self):
        """모든 모델 로드"""
        try:
            self.enhancer.load_model()
            self.subtitle_generator.load_model()
            self.models_loaded = True
            logger.info("All video models loaded")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def process(
        self,
        video_path: str,
        output_dir: str,
        options: Optional[Dict] = None
    ) -> VideoProcessingResult:
        """비디오 처리"""
        import time
        start_time = time.time()
        
        options = options or {}
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 1. 메타데이터 추출
        metadata = self.metadata_extractor.extract(video_path)
        logger.info(f"Video: {metadata.width}x{metadata.height}, {metadata.duration:.1f}s, {metadata.fps}fps")
        
        # 2. 장면 분할
        scenes = []
        if options.get("detect_scenes", True):
            scenes = self.scene_detector.detect_scenes(video_path)
            logger.info(f"Detected {len(scenes)} scenes")
        
        # 3. 핵심 프레임 추출
        keyframe_paths = []
        if options.get("extract_keyframes", True):
            keyframe_dir = os.path.join(output_dir, "keyframes")
            if scenes:
                keyframe_paths = self.keyframe_extractor.extract_keyframes(
                    video_path, scenes, keyframe_dir
                )
            else:
                keyframe_paths = [
                    path for _, _ in self.keyframe_extractor.extract_uniform_keyframes(
                        video_path, 5.0, keyframe_dir
                    )
                ]
            logger.info(f"Extracted {len(keyframe_paths)} keyframes")
        
        # 4. 오디오 추출
        audio_path = None
        if options.get("extract_audio", True) and metadata.has_audio:
            audio_path = os.path.join(output_dir, f"{base_name}.wav")
            try:
                self.audio_extractor.extract_audio(video_path, audio_path)
            except Exception as e:
                logger.error(f"Audio extraction failed: {e}")
                audio_path = None
        
        # 5. 자막 생성
        subtitles = []
        if options.get("generate_subtitles", True) and audio_path:
            subtitles, srt_content = self.subtitle_generator.generate_subtitles(audio_path)
            if srt_content:
                srt_path = os.path.join(output_dir, f"{base_name}.srt")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                logger.info(f"Generated {len(subtitles)} subtitle entries")
        
        # 6. 비디오 향상 (선택적)
        output_video_path = None
        enhancement_applied = False
        stabilization_applied = False
        
        if options.get("enhance", False):
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.mp4")
            self.enhancer.enhance_video(video_path, enhanced_path)
            output_video_path = enhanced_path
            enhancement_applied = True
        
        if options.get("stabilize", False):
            input_for_stabilize = output_video_path or video_path
            stabilized_path = os.path.join(output_dir, f"{base_name}_stabilized.mp4")
            self.stabilizer.stabilize(input_for_stabilize, stabilized_path)
            output_video_path = stabilized_path
            stabilization_applied = True
        
        if options.get("color_correct", False):
            input_for_color = output_video_path or video_path
            corrected_path = os.path.join(output_dir, f"{base_name}_corrected.mp4")
            self.color_corrector.correct_colors(input_for_color, corrected_path)
            output_video_path = corrected_path
        
        processing_time = time.time() - start_time
        
        return VideoProcessingResult(
            input_path=video_path,
            output_path=output_video_path,
            metadata=metadata,
            scenes=scenes,
            subtitles=subtitles,
            extracted_audio_path=audio_path,
            keyframe_paths=keyframe_paths,
            processing_time=processing_time,
            enhancement_applied=enhancement_applied,
            stabilization_applied=stabilization_applied
        )
    
    def batch_process(
        self,
        video_paths: List[str],
        output_dir: str,
        options: Optional[Dict] = None
    ) -> List[VideoProcessingResult]:
        """배치 처리"""
        results = []
        
        for i, path in enumerate(video_paths):
            logger.info(f"Processing {i+1}/{len(video_paths)}: {path}")
            
            try:
                video_output_dir = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(path))[0]
                )
                result = self.process(path, video_output_dir, options)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        return results


# CLI 인터페이스
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diaspora Archive Video Processing")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--enhance", action="store_true", help="Enhance video quality")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize video")
    parser.add_argument("--color-correct", action="store_true", help="Color correction")
    parser.add_argument("--no-scenes", action="store_true", help="Skip scene detection")
    parser.add_argument("--no-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio extraction")
    parser.add_argument("--no-subtitles", action="store_true", help="Skip subtitle generation")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    processor = VideoProcessor({"device": args.device})
    processor.load_models()
    
    options = {
        "detect_scenes": not args.no_scenes,
        "extract_keyframes": not args.no_keyframes,
        "extract_audio": not args.no_audio,
        "generate_subtitles": not args.no_subtitles,
        "enhance": args.enhance,
        "stabilize": args.stabilize,
        "color_correct": args.color_correct
    }
    
    if os.path.isdir(args.input):
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in video_extensions
        ]
        results = processor.batch_process(video_paths, args.output, options)
        print(f"Processed {len(results)} videos")
    else:
        result = processor.process(args.input, args.output, options)
        
        print(f"\n=== Video Processing Result ===")
        print(f"Duration: {result.metadata.duration:.1f}s")
        print(f"Resolution: {result.metadata.width}x{result.metadata.height}")
        print(f"Scenes: {len(result.scenes)}")
        print(f"Keyframes: {len(result.keyframe_paths)}")
        print(f"Subtitles: {len(result.subtitles)}")
        print(f"Processing time: {result.processing_time:.1f}s")
        
        if result.output_path:
            print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()
