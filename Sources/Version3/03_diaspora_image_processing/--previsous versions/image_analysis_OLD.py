"""
image_analysis.py
이미지 분석 및 메타데이터 생성 모듈

BLIP, CLIP, face_recognition을 활용하여 이미지를 분석하고
Dublin Core 표준에 맞는 메타데이터를 자동 생성합니다.

주요 기능:
1. 이미지 캡션 생성 (BLIP)
2. 장면/이벤트 분류 (CLIP zero-shot)
3. 얼굴 감지 및 인코딩 (face_recognition)
4. Dublin Core 메타데이터 생성

References:
    - BLIP: Li et al. (2022). BLIP: Bootstrapping Language-Image Pre-training
    - CLIP: Radford et al. (2021). Learning Transferable Visual Models
    - face_recognition: Geitgey (2017)
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

# Transformers imports
try:
    from transformers import (
        BlipProcessor, 
        BlipForConditionalGeneration,
        CLIPProcessor,
        CLIPModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers")

# Face recognition imports
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not installed. Run: pip install face_recognition")


class ImageCaptioner:
    """
    BLIP 기반 이미지 캡션 생성기
    
    Salesforce의 BLIP 모델을 사용하여 이미지 내용을 
    자연어로 설명하는 캡션을 생성합니다.
    
    Example:
        >>> captioner = ImageCaptioner()
        >>> caption = captioner.generate("family_photo.jpg")
        >>> print(caption)  # "a family gathering outdoors with smiling people"
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = 'cuda',
        max_length: int = 50
    ):
        """
        캡션 생성기 초기화
        
        Args:
            model_name: HuggingFace 모델 이름
            device: 연산 장치
            max_length: 최대 캡션 길이
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 라이브러리가 필요합니다.")
        
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.model_name = model_name
        
        # 모델 및 프로세서 로드
        print(f"BLIP 모델 로딩 중: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ ImageCaptioner 초기화 완료 (장치: {self.device})")
    
    def _setup_device(self, device: str) -> str:
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    
    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        num_beams: int = 5,
        max_length: Optional[int] = None
    ) -> str:
        """
        이미지 캡션 생성
        
        Args:
            image: 입력 이미지
            num_beams: 빔 서치 크기
            max_length: 최대 토큰 길이
            
        Returns:
            생성된 캡션 문자열
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        
        # 전처리
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        # 캡션 생성
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length or self.max_length,
                num_beams=num_beams
            )
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    
    def generate_with_prompt(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str = "a photograph of"
    ) -> str:
        """
        프롬프트 기반 조건부 캡션 생성
        
        Args:
            image: 입력 이미지
            prompt: 시작 프롬프트
            
        Returns:
            생성된 캡션
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        inputs = self.processor(
            images=img, 
            text=prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length
            )
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()


class SceneClassifier:
    """
    CLIP 기반 장면/이벤트 분류기
    
    CLIP의 zero-shot 분류 기능을 활용하여
    이미지의 장면이나 이벤트 유형을 분류합니다.
    
    Example:
        >>> classifier = SceneClassifier()
        >>> results = classifier.classify("wedding.jpg")
        >>> print(results)  # [("wedding ceremony", 0.85), ("family gathering", 0.10)]
    """
    
    # 디아스포라 아카이브에 특화된 분류 카테고리
    DEFAULT_CATEGORIES = [
        # 가족 행사
        "wedding ceremony",
        "birthday party",
        "baby's first birthday (doljanchi)",
        "graduation ceremony",
        "family memorial service (jesa)",
        
        # 일상
        "family gathering",
        "family meal",
        "family picnic",
        "family trip",
        
        # 장소
        "indoor scene",
        "outdoor scene",
        "school",
        "workplace",
        "church",
        "restaurant",
        
        # 구성
        "portrait photo",
        "group photo",
        "landscape",
        "street scene",
        "building"
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = 'cuda',
        categories: Optional[List[str]] = None
    ):
        """
        분류기 초기화
        
        Args:
            model_name: CLIP 모델 이름
            device: 연산 장치
            categories: 분류 카테고리 리스트
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 라이브러리가 필요합니다.")
        
        self.device = self._setup_device(device)
        self.categories = categories or self.DEFAULT_CATEGORIES
        
        # 모델 로드
        print(f"CLIP 모델 로딩 중: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ SceneClassifier 초기화 완료 (카테고리: {len(self.categories)}개)")
    
    def _setup_device(self, device: str) -> str:
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    
    def classify(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
        threshold: float = 0.1,
        categories: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        이미지 분류 수행
        
        Args:
            image: 입력 이미지
            top_k: 상위 k개 결과 반환
            threshold: 최소 확률 임계값
            categories: 분류 카테고리 (None이면 기본값 사용)
            
        Returns:
            [(카테고리, 확률), ...] 리스트
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        
        categories = categories or self.categories
        
        # 텍스트 프롬프트 생성
        text_prompts = [f"a photo of {cat}" for cat in categories]
        
        # 전처리
        inputs = self.processor(
            text=text_prompts,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # 결과 정렬
        results = list(zip(categories, probs))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 필터링
        filtered = [(cat, float(prob)) for cat, prob in results[:top_k] if prob >= threshold]
        
        return filtered
    
    def get_category_scores(
        self,
        image: Union[str, Path, Image.Image],
        categories: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        모든 카테고리에 대한 점수 반환
        
        Args:
            image: 입력 이미지
            categories: 분류 카테고리
            
        Returns:
            {카테고리: 확률} 딕셔너리
        """
        results = self.classify(image, top_k=len(self.categories), threshold=0.0, categories=categories)
        return {cat: prob for cat, prob in results}


class FaceAnalyzer:
    """
    얼굴 감지 및 분석기
    
    face_recognition 라이브러리를 사용하여
    이미지 내 얼굴을 감지하고 인코딩합니다.
    동일 인물 매칭에 활용할 수 있습니다.
    
    Example:
        >>> analyzer = FaceAnalyzer()
        >>> faces = analyzer.detect("group_photo.jpg")
        >>> print(f"감지된 얼굴: {len(faces)}개")
    """
    
    def __init__(
        self,
        model: str = "hog",
        encoding_model: str = "large",
        tolerance: float = 0.6
    ):
        """
        얼굴 분석기 초기화
        
        Args:
            model: 감지 모델 ('hog' 또는 'cnn')
            encoding_model: 인코딩 모델 ('small' 또는 'large')
            tolerance: 동일 인물 판정 임계값 (낮을수록 엄격)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError(
                "face_recognition 라이브러리가 필요합니다. "
                "설치: pip install face_recognition"
            )
        
        self.model = model
        self.encoding_model = encoding_model
        self.tolerance = tolerance
        
        print(f"✓ FaceAnalyzer 초기화 완료")
        print(f"  - 감지 모델: {model}")
        print(f"  - 인코딩 모델: {encoding_model}")
    
    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        이미지에서 얼굴 감지
        
        Args:
            image: 입력 이미지
            
        Returns:
            감지된 얼굴 정보 리스트
            [{
                'location': (top, right, bottom, left),
                'encoding': numpy array (128,),
                'area': int
            }, ...]
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = face_recognition.load_image_file(str(image))
        elif isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            # BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img = image
        
        # 얼굴 감지
        face_locations = face_recognition.face_locations(img, model=self.model)
        
        if not face_locations:
            return []
        
        # 얼굴 인코딩
        face_encodings = face_recognition.face_encodings(
            img, 
            face_locations,
            model=self.encoding_model
        )
        
        # 결과 구성
        faces = []
        for i, (loc, enc) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = loc
            area = (right - left) * (bottom - top)
            
            faces.append({
                'index': i,
                'location': loc,
                'bbox': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left
                },
                'encoding': enc,
                'area': area,
                'center': ((left + right) // 2, (top + bottom) // 2)
            })
        
        # 얼굴 크기 순으로 정렬
        faces.sort(key=lambda x: x['area'], reverse=True)
        
        return faces
    
    def compare_faces(
        self,
        face1_encoding: np.ndarray,
        face2_encoding: np.ndarray,
        tolerance: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        두 얼굴이 동일 인물인지 비교
        
        Args:
            face1_encoding: 첫 번째 얼굴 인코딩
            face2_encoding: 두 번째 얼굴 인코딩
            tolerance: 임계값 (None이면 기본값 사용)
            
        Returns:
            (동일 인물 여부, 거리)
        """
        tolerance = tolerance or self.tolerance
        
        distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
        is_match = distance < tolerance
        
        return is_match, float(distance)
    
    def find_matches(
        self,
        target_encoding: np.ndarray,
        face_encodings: List[np.ndarray],
        tolerance: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        대상 얼굴과 매칭되는 얼굴 찾기
        
        Args:
            target_encoding: 대상 얼굴 인코딩
            face_encodings: 비교할 얼굴 인코딩 리스트
            tolerance: 임계값
            
        Returns:
            [(인덱스, 거리), ...] - 매칭된 얼굴들
        """
        tolerance = tolerance or self.tolerance
        
        distances = face_recognition.face_distance(face_encodings, target_encoding)
        matches = []
        
        for i, dist in enumerate(distances):
            if dist < tolerance:
                matches.append((i, float(dist)))
        
        matches.sort(key=lambda x: x[1])
        return matches
    
    def extract_face_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        face_location: tuple,
        padding: float = 0.2
    ) -> Image.Image:
        """
        이미지에서 얼굴 영역 추출
        
        Args:
            image: 원본 이미지
            face_location: (top, right, bottom, left)
            padding: 여백 비율
            
        Returns:
            추출된 얼굴 이미지
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        top, right, bottom, left = face_location
        width = right - left
        height = bottom - top
        
        # 패딩 적용
        pad_w = int(width * padding)
        pad_h = int(height * padding)
        
        left = max(0, left - pad_w)
        top = max(0, top - pad_h)
        right = min(img.width, right + pad_w)
        bottom = min(img.height, bottom + pad_h)
        
        return img.crop((left, top, right, bottom))


class ImageAnalysisModule:
    """
    통합 이미지 분석 모듈
    
    캡션 생성, 장면 분류, 얼굴 분석을 통합하여
    Dublin Core 표준 메타데이터를 생성합니다.
    
    Example:
        >>> analyzer = ImageAnalysisModule()
        >>> result = analyzer.analyze("family_photo.jpg")
        >>> metadata = result['metadata']
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        enable_captioning: bool = True,
        enable_classification: bool = True,
        enable_face_analysis: bool = True
    ):
        """
        분석 모듈 초기화
        
        Args:
            device: 연산 장치
            enable_captioning: 캡션 생성 활성화
            enable_classification: 분류 활성화
            enable_face_analysis: 얼굴 분석 활성화
        """
        self.device = device
        
        # 서브 모듈 초기화
        self.captioner = None
        self.classifier = None
        self.face_analyzer = None
        
        if enable_captioning and TRANSFORMERS_AVAILABLE:
            try:
                self.captioner = ImageCaptioner(device=device)
            except Exception as e:
                print(f"Warning: 캡션 모듈 초기화 실패: {e}")
        
        if enable_classification and TRANSFORMERS_AVAILABLE:
            try:
                self.classifier = SceneClassifier(device=device)
            except Exception as e:
                print(f"Warning: 분류 모듈 초기화 실패: {e}")
        
        if enable_face_analysis and FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_analyzer = FaceAnalyzer()
            except Exception as e:
                print(f"Warning: 얼굴 분석 모듈 초기화 실패: {e}")
        
        print(f"✓ ImageAnalysisModule 초기화 완료")
    
    def analyze(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        generate_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        이미지 종합 분석 수행
        
        Args:
            image: 입력 이미지
            generate_metadata: Dublin Core 메타데이터 생성 여부
            
        Returns:
            {
                'caption': str,
                'scenes': [(category, probability), ...],
                'faces': [{face_info}, ...],
                'metadata': {dublin_core_fields}
            }
        """
        result = {
            'caption': None,
            'scenes': [],
            'faces': [],
            'metadata': None
        }
        
        # 캡션 생성
        if self.captioner:
            try:
                result['caption'] = self.captioner.generate(image)
            except Exception as e:
                print(f"Warning: 캡션 생성 실패: {e}")
        
        # 장면 분류
        if self.classifier:
            try:
                result['scenes'] = self.classifier.classify(image)
            except Exception as e:
                print(f"Warning: 장면 분류 실패: {e}")
        
        # 얼굴 분석
        if self.face_analyzer:
            try:
                faces = self.face_analyzer.detect(image)
                # 인코딩은 직렬화 불가하므로 제외
                result['faces'] = [
                    {k: v for k, v in f.items() if k != 'encoding'}
                    for f in faces
                ]
                result['face_encodings'] = [f['encoding'] for f in faces]
            except Exception as e:
                print(f"Warning: 얼굴 분석 실패: {e}")
        
        # Dublin Core 메타데이터 생성
        if generate_metadata:
            result['metadata'] = self._generate_dublin_core(result, image)
        
        return result
    
    def _generate_dublin_core(
        self,
        analysis: Dict[str, Any],
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Dublin Core 메타데이터 생성
        
        Args:
            analysis: 분석 결과
            image: 원본 이미지
            
        Returns:
            Dublin Core 필드 딕셔너리
        """
        metadata = {
            'dc:type': 'Image',
            'dc:format': 'image/jpeg',
            'dc:created': datetime.now().isoformat()
        }
        
        # 캡션 → description
        if analysis.get('caption'):
            metadata['dc:description'] = analysis['caption']
        
        # 장면 분류 → subject
        if analysis.get('scenes'):
            subjects = [cat for cat, prob in analysis['scenes'] if prob > 0.3]
            if subjects:
                metadata['dc:subject'] = subjects
        
        # 얼굴 수 → contributor 또는 커스텀 필드
        if analysis.get('faces'):
            metadata['x:faceCount'] = len(analysis['faces'])
        
        # EXIF 정보 추출 시도
        if isinstance(image, (str, Path)):
            exif_data = self._extract_exif(str(image))
            if exif_data:
                if exif_data.get('date'):
                    metadata['dc:date'] = exif_data['date']
                if exif_data.get('camera'):
                    metadata['x:camera'] = exif_data['camera']
        
        return metadata
    
    def _extract_exif(self, image_path: str) -> Optional[Dict[str, Any]]:
        """EXIF 메타데이터 추출"""
        try:
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if not exif_data:
                return None
            
            result = {}
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    result['date'] = str(value)
                elif tag == 'Make':
                    result['camera'] = str(value)
            
            return result if result else None
            
        except Exception:
            return None
    
    def save_analysis(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """분석 결과를 JSON으로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # numpy array 제거 (직렬화 불가)
        save_result = {k: v for k, v in result.items() if k != 'face_encodings'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)


# ============================================
# 테스트 코드
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Analysis Module Test")
    parser.add_argument("--input", "-i", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output", "-o", type=str, default="analysis.json", help="출력 JSON 경로")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="장치")
    
    args = parser.parse_args()
    
    # 모듈 초기화
    analyzer = ImageAnalysisModule(device=args.device)
    
    # 분석 수행
    print(f"이미지 분석 중: {args.input}")
    result = analyzer.analyze(args.input)
    
    # 결과 출력
    print("\n=== 분석 결과 ===")
    print(f"캡션: {result['caption']}")
    print(f"장면: {result['scenes']}")
    print(f"얼굴 수: {len(result['faces'])}")
    print(f"\nDublin Core 메타데이터:")
    for key, value in (result['metadata'] or {}).items():
        print(f"  {key}: {value}")
    
    # 결과 저장
    analyzer.save_analysis(result, args.output)
    print(f"\n결과 저장 완료: {args.output}")
