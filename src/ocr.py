"""
문서 처리 모듈 (Document Processing Module)

한인 디아스포라 기록유산의 문서 인식 및 분석을 위한 AI 기반 처리 모듈

주요 기능:
- 문서 이미지 전처리 (Preprocessing)
- 인쇄체 OCR (Printed Text Recognition): EasyOCR, Tesseract
- 손글씨 인식 (Handwriting Recognition): TrOCR
- 다국어 처리 (Multilingual Processing)
- 문서 레이아웃 분석 (Layout Analysis): LayoutLMv3
- 개체명 인식 (Named Entity Recognition): KoBERT
- 키워드 추출 (Keyword Extraction)

Author: Diaspora Archive Project
License: MIT
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import cv2
from PIL import Image
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """문서 유형"""
    LETTER = "letter"              # 편지
    POSTCARD = "postcard"          # 엽서
    DIARY = "diary"                # 일기
    OFFICIAL = "official"          # 공식 문서
    CERTIFICATE = "certificate"    # 증명서
    NEWSPAPER = "newspaper"        # 신문 스크랩
    RECIPE = "recipe"              # 레시피
    MEMO = "memo"                  # 메모
    ADDRESS_BOOK = "address_book"  # 주소록
    UNKNOWN = "unknown"


class ScriptType(Enum):
    """필체 유형"""
    PRINTED = "printed"            # 인쇄체
    HANDWRITTEN_FORMAL = "formal"  # 정자체
    HANDWRITTEN_SEMI = "semi"      # 반흘림체
    HANDWRITTEN_CURSIVE = "cursive"  # 흘림체
    MIXED = "mixed"                # 혼합


class Language(Enum):
    """언어"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    RUSSIAN = "ru"
    MIXED = "mixed"


@dataclass
class BoundingBox:
    """텍스트 영역 바운딩 박스"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    def to_list(self) -> List[int]:
        return [self.x, self.y, self.width, self.height]
    
    def to_points(self) -> List[List[int]]:
        """4개 꼭지점 반환"""
        return [
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height]
        ]


@dataclass
class TextBlock:
    """텍스트 블록"""
    text: str
    bbox: BoundingBox
    confidence: float
    language: Language = Language.KOREAN
    script_type: ScriptType = ScriptType.PRINTED
    block_type: str = "paragraph"  # paragraph, title, date, signature, etc.


@dataclass
class NamedEntity:
    """개체명"""
    text: str
    entity_type: str  # PER, LOC, ORG, DAT, TIM
    start_pos: int
    end_pos: int
    confidence: float = 0.0


@dataclass
class DocumentMetadata:
    """문서 메타데이터"""
    document_type: DocumentType
    script_type: ScriptType
    languages: List[Language]
    estimated_date: Optional[str] = None
    author: Optional[str] = None
    recipient: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[NamedEntity] = field(default_factory=list)


@dataclass
class OCRResult:
    """OCR 결과"""
    full_text: str
    text_blocks: List[TextBlock]
    metadata: DocumentMetadata
    confidence: float
    processing_time: float
    raw_output: Optional[Dict] = None


class DocumentPreprocessor:
    """문서 이미지 전처리"""
    
    def __init__(self):
        logger.info("DocumentPreprocessor initialized")
    
    def preprocess(
        self,
        image: np.ndarray,
        auto_rotate: bool = True,
        denoise: bool = True,
        binarize: bool = False
    ) -> np.ndarray:
        """문서 이미지 전처리"""
        result = image.copy()
        
        # 그레이스케일 변환 (필요시)
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result
        
        # 노이즈 제거
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 기울기 보정
        if auto_rotate:
            angle = self._detect_skew(gray)
            if abs(angle) > 0.5:
                gray = self._rotate_image(gray, angle)
        
        # 이진화 (선택적)
        if binarize:
            gray = self._adaptive_binarize(gray)
        
        # 컬러 이미지로 반환 (OCR 호환성)
        if len(image.shape) == 3:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            result = gray
        
        return result
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """기울기 감지"""
        # 에지 감지
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 허프 변환으로 선 감지
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return 0.0
        
        # 각도 계산
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        return np.median(angles)
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """적응적 이진화"""
        # Otsu's method 먼저 시도
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 적응적 이진화
        adaptive = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 두 결과의 AND 연산
        result = cv2.bitwise_and(binary, adaptive)
        
        return result
    
    def detect_text_regions(self, image: np.ndarray) -> List[BoundingBox]:
        """텍스트 영역 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # MSER (Maximally Stable Extremal Regions) 사용
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # 바운딩 박스 생성
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 10 and h > 10:  # 최소 크기 필터
                bboxes.append(BoundingBox(x, y, w, h))
        
        # 겹치는 박스 병합
        bboxes = self._merge_overlapping_boxes(bboxes)
        
        return bboxes
    
    def _merge_overlapping_boxes(
        self,
        boxes: List[BoundingBox],
        overlap_threshold: float = 0.5
    ) -> List[BoundingBox]:
        """겹치는 박스 병합"""
        if not boxes:
            return []
        
        # 면적순 정렬
        boxes = sorted(boxes, key=lambda b: b.width * b.height, reverse=True)
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            current = box1
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                
                # IoU 계산
                iou = self._calculate_iou(current, box2)
                if iou > overlap_threshold:
                    # 병합
                    x = min(current.x, box2.x)
                    y = min(current.y, box2.y)
                    x2 = max(current.x + current.width, box2.x + box2.width)
                    y2 = max(current.y + current.height, box2.y + box2.height)
                    current = BoundingBox(x, y, x2 - x, y2 - y)
                    used.add(j)
            
            merged.append(current)
        
        return merged
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """IoU 계산"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class PrintedTextOCR:
    """인쇄체 OCR (EasyOCR, Tesseract)"""
    
    def __init__(self, languages: List[str] = None, device: str = "cuda"):
        self.languages = languages or ["ko", "en"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.reader = None
        self.tesseract_available = False
        logger.info(f"PrintedTextOCR initialized for {self.languages} on {self.device}")
    
    def load_model(self):
        """OCR 모델 로드"""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=(self.device == "cuda")
            )
            logger.info("EasyOCR loaded successfully")
        except ImportError:
            logger.warning("EasyOCR not installed")
        
        # Tesseract 체크
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract available")
        except Exception:
            logger.warning("Tesseract not available")
    
    def recognize(
        self,
        image: np.ndarray,
        use_tesseract: bool = False
    ) -> List[TextBlock]:
        """텍스트 인식"""
        if use_tesseract and self.tesseract_available:
            return self._recognize_tesseract(image)
        elif self.reader is not None:
            return self._recognize_easyocr(image)
        else:
            logger.error("No OCR engine available")
            return []
    
    def _recognize_easyocr(self, image: np.ndarray) -> List[TextBlock]:
        """EasyOCR로 인식"""
        results = self.reader.readtext(image)
        
        text_blocks = []
        for bbox, text, confidence in results:
            # bbox는 4개 꼭지점 좌표
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            
            block = TextBlock(
                text=text,
                bbox=BoundingBox(x, y, w, h, confidence),
                confidence=confidence,
                language=self._detect_language(text),
                script_type=ScriptType.PRINTED
            )
            text_blocks.append(block)
        
        return text_blocks
    
    def _recognize_tesseract(self, image: np.ndarray) -> List[TextBlock]:
        """Tesseract로 인식"""
        import pytesseract
        
        # 언어 설정
        lang_map = {"ko": "kor", "en": "eng", "ja": "jpn", "zh": "chi_sim"}
        lang_str = "+".join([lang_map.get(l, l) for l in self.languages])
        
        # OCR 실행
        data = pytesseract.image_to_data(
            image,
            lang=lang_str,
            output_type=pytesseract.Output.DICT
        )
        
        text_blocks = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            conf = float(data['conf'][i]) / 100.0
            if conf < 0:
                conf = 0.0
            
            block = TextBlock(
                text=text,
                bbox=BoundingBox(
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i],
                    conf
                ),
                confidence=conf,
                language=self._detect_language(text),
                script_type=ScriptType.PRINTED
            )
            text_blocks.append(block)
        
        return text_blocks
    
    def _detect_language(self, text: str) -> Language:
        """텍스트 언어 감지"""
        # 간단한 문자 기반 감지
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
        chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        russian_chars = len(re.findall(r'[а-яА-Я]', text))
        
        total = korean_chars + english_chars + japanese_chars + chinese_chars + russian_chars
        
        if total == 0:
            return Language.KOREAN
        
        if korean_chars / total > 0.5:
            return Language.KOREAN
        elif english_chars / total > 0.5:
            return Language.ENGLISH
        elif japanese_chars / total > 0.3:
            return Language.JAPANESE
        elif chinese_chars / total > 0.3:
            return Language.CHINESE
        elif russian_chars / total > 0.3:
            return Language.RUSSIAN
        else:
            return Language.MIXED


class HandwritingRecognizer:
    """손글씨 인식 (TrOCR)"""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        logger.info(f"HandwritingRecognizer initialized on {self.device}")
    
    def load_model(self):
        """TrOCR 모델 로드"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            
            logger.info("TrOCR model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
    
    def recognize(self, image: np.ndarray, line_images: List[np.ndarray] = None) -> List[TextBlock]:
        """손글씨 인식"""
        if self.model is None:
            logger.error("TrOCR model not loaded")
            return []
        
        # 라인별 이미지가 없으면 전체 이미지 사용
        if line_images is None:
            line_images = self._segment_lines(image)
        
        text_blocks = []
        
        for i, line_img in enumerate(line_images):
            try:
                # BGR to RGB
                if len(line_img.shape) == 3:
                    rgb_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
                
                pil_img = Image.fromarray(rgb_img)
                
                # 추론
                pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 텍스트 블록 생성
                h, w = line_img.shape[:2]
                block = TextBlock(
                    text=text,
                    bbox=BoundingBox(0, i * h, w, h, 0.9),
                    confidence=0.9,
                    script_type=self._classify_script(line_img)
                )
                text_blocks.append(block)
                
            except Exception as e:
                logger.error(f"Line recognition error: {e}")
        
        return text_blocks
    
    def _segment_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """텍스트 라인 분할"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 수평 프로젝션
        h_proj = np.sum(binary, axis=1)
        
        # 라인 경계 찾기
        threshold = np.max(h_proj) * 0.1
        in_line = False
        lines = []
        start_y = 0
        
        for y, val in enumerate(h_proj):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                if y - start_y > 10:  # 최소 높이
                    lines.append((start_y, y))
        
        # 마지막 라인 처리
        if in_line:
            lines.append((start_y, len(h_proj)))
        
        # 라인 이미지 추출
        line_images = []
        for start_y, end_y in lines:
            line_img = image[max(0, start_y-5):min(image.shape[0], end_y+5), :]
            line_images.append(line_img)
        
        return line_images if line_images else [image]
    
    def _classify_script(self, line_image: np.ndarray) -> ScriptType:
        """필체 유형 분류"""
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if len(line_image.shape) == 3 else line_image
        
        # 에지 복잡도 분석
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (gray.shape[0] * gray.shape[1] * 255)
        
        # 수직/수평 선 비율
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        h_ratio = np.abs(sobel_x).sum() / (np.abs(sobel_x).sum() + np.abs(sobel_y).sum() + 1e-6)
        
        # 분류
        if edge_density < 0.05 and h_ratio > 0.6:
            return ScriptType.HANDWRITTEN_FORMAL
        elif edge_density < 0.1:
            return ScriptType.HANDWRITTEN_SEMI
        else:
            return ScriptType.HANDWRITTEN_CURSIVE


class LayoutAnalyzer:
    """문서 레이아웃 분석 (LayoutLMv3)"""
    
    BLOCK_TYPES = [
        "title", "paragraph", "date", "greeting", "body",
        "signature", "address", "header", "footer", "table"
    ]
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        logger.info(f"LayoutAnalyzer initialized on {self.device}")
    
    def load_model(self):
        """LayoutLMv3 모델 로드"""
        try:
            from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
            
            model_name = "microsoft/layoutlmv3-base"
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)
            
            logger.info("LayoutLMv3 model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
    
    def analyze(self, image: np.ndarray, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """레이아웃 분석"""
        if not text_blocks:
            return []
        
        # 규칙 기반 분석 (모델이 없는 경우)
        return self._rule_based_analysis(image, text_blocks)
    
    def _rule_based_analysis(self, image: np.ndarray, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """규칙 기반 레이아웃 분석"""
        height, width = image.shape[:2]
        
        for block in text_blocks:
            # 위치 기반 분류
            y_ratio = block.bbox.y / height
            x_ratio = block.bbox.x / width
            
            text_lower = block.text.lower()
            
            # 날짜 패턴
            date_pattern = r'\d{4}[년\.\-/]\s*\d{1,2}[월\.\-/]?\s*\d{0,2}[일]?'
            if re.search(date_pattern, block.text):
                block.block_type = "date"
            # 인사말
            elif any(greeting in text_lower for greeting in ["dear", "친애하는", "안녕", "안부"]):
                block.block_type = "greeting"
            # 서명
            elif y_ratio > 0.8 and len(block.text) < 20:
                block.block_type = "signature"
            # 주소
            elif any(addr in text_lower for addr in ["주소", "address", "street", "city"]):
                block.block_type = "address"
            # 제목 (상단, 짧은 텍스트)
            elif y_ratio < 0.15 and len(block.text) < 50:
                block.block_type = "title"
            # 본문
            else:
                block.block_type = "body"
        
        return text_blocks


class NamedEntityRecognizer:
    """개체명 인식 (KoBERT)"""
    
    ENTITY_TYPES = {
        "PER": "인명",
        "LOC": "지명",
        "ORG": "기관명",
        "DAT": "날짜",
        "TIM": "시간"
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        logger.info(f"NamedEntityRecognizer initialized on {self.device}")
    
    def load_model(self):
        """NER 모델 로드"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            # 한국어 NER 모델
            model_name = "monologg/koelectra-base-v3-discriminator"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                "monologg/koelectra-base-v3-discriminator"
            ).to(self.device)
            
            logger.info("NER model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
    
    def extract_entities(self, text: str) -> List[NamedEntity]:
        """개체명 추출"""
        entities = []
        
        # 규칙 기반 추출 (모델이 없는 경우도 동작)
        entities.extend(self._extract_dates(text))
        entities.extend(self._extract_names(text))
        entities.extend(self._extract_locations(text))
        entities.extend(self._extract_organizations(text))
        
        return entities
    
    def _extract_dates(self, text: str) -> List[NamedEntity]:
        """날짜 추출"""
        entities = []
        
        # 한국어 날짜 패턴
        patterns = [
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',
            r'\d{4}[\./-]\d{1,2}[\./-]\d{1,2}',
            r'\d{1,2}월\s*\d{1,2}일',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(NamedEntity(
                    text=match.group(),
                    entity_type="DAT",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95
                ))
        
        return entities
    
    def _extract_names(self, text: str) -> List[NamedEntity]:
        """인명 추출"""
        entities = []
        
        # 한국어 이름 패턴 (성+이름)
        korean_name_pattern = r'[가-힣]{2,4}(?=\s*(씨|님|선생|교수|사장|대표|회장|씨가|님이|에게))'
        
        for match in re.finditer(korean_name_pattern, text):
            entities.append(NamedEntity(
                text=match.group(),
                entity_type="PER",
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.8
            ))
        
        # 영어 이름 패턴
        english_name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        
        for match in re.finditer(english_name_pattern, text):
            entities.append(NamedEntity(
                text=match.group(),
                entity_type="PER",
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.7
            ))
        
        return entities
    
    def _extract_locations(self, text: str) -> List[NamedEntity]:
        """지명 추출"""
        entities = []
        
        # 한국 지명 키워드
        korean_locations = [
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
            "목포", "전주", "청주", "춘천", "원주", "강릉"
        ]
        
        # 미국 지명 키워드
        us_locations = [
            "Los Angeles", "LA", "New York", "Chicago", "Houston",
            "San Francisco", "Seattle", "Hawaii", "California", "Texas"
        ]
        
        all_locations = korean_locations + us_locations
        
        for loc in all_locations:
            for match in re.finditer(re.escape(loc), text, re.IGNORECASE):
                entities.append(NamedEntity(
                    text=match.group(),
                    entity_type="LOC",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_organizations(self, text: str) -> List[NamedEntity]:
        """기관명 추출"""
        entities = []
        
        # 기관 키워드
        org_keywords = [
            "한인회", "교회", "성당", "학교", "대학", "회사", "병원",
            "Church", "School", "University", "Company", "Association"
        ]
        
        # 패턴: 키워드 앞의 명사구
        for keyword in org_keywords:
            pattern = rf'[가-힣A-Za-z\s]{{2,20}}{keyword}'
            for match in re.finditer(pattern, text):
                entities.append(NamedEntity(
                    text=match.group(),
                    entity_type="ORG",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.75
                ))
        
        return entities


class KeywordExtractor:
    """키워드 추출"""
    
    # 불용어
    STOPWORDS_KO = {
        "이", "그", "저", "것", "수", "등", "및", "또는", "그리고",
        "하다", "되다", "있다", "없다", "이다", "아니다",
        "은", "는", "이", "가", "을", "를", "에", "의", "와", "과"
    }
    
    STOPWORDS_EN = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall"
    }
    
    def __init__(self):
        self.stopwords = self.STOPWORDS_KO | self.STOPWORDS_EN
        logger.info("KeywordExtractor initialized")
    
    def extract(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """키워드 추출 (TF-IDF 기반 간소화)"""
        # 토큰화
        tokens = self._tokenize(text)
        
        # 불용어 제거
        tokens = [t for t in tokens if t.lower() not in self.stopwords and len(t) > 1]
        
        # 빈도 계산
        from collections import Counter
        freq = Counter(tokens)
        
        # 정규화
        total = sum(freq.values())
        keywords = [(word, count/total) for word, count in freq.most_common(top_n)]
        
        return keywords
    
    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화"""
        # 한국어와 영어 분리
        korean = re.findall(r'[가-힣]+', text)
        english = re.findall(r'[a-zA-Z]+', text)
        
        return korean + english


class DocumentClassifier:
    """문서 유형 분류"""
    
    def __init__(self):
        logger.info("DocumentClassifier initialized")
    
    def classify(self, text: str, metadata: Dict = None) -> DocumentType:
        """문서 유형 분류"""
        text_lower = text.lower()
        
        # 키워드 기반 분류
        if any(kw in text_lower for kw in ["dear", "친애하는", "안부", "그리운"]):
            return DocumentType.LETTER
        
        if any(kw in text_lower for kw in ["엽서", "postcard"]):
            return DocumentType.POSTCARD
        
        if any(kw in text_lower for kw in ["일기", "diary", "오늘은", "today"]):
            return DocumentType.DIARY
        
        if any(kw in text_lower for kw in ["증명", "certificate", "hereby", "이에"]):
            return DocumentType.CERTIFICATE
        
        if any(kw in text_lower for kw in ["공문", "관인", "official"]):
            return DocumentType.OFFICIAL
        
        if any(kw in text_lower for kw in ["재료", "recipe", "컵", "큰술"]):
            return DocumentType.RECIPE
        
        if any(kw in text_lower for kw in ["신문", "newspaper", "기사"]):
            return DocumentType.NEWSPAPER
        
        return DocumentType.UNKNOWN


class DocumentProcessor:
    """통합 문서 처리 파이프라인"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 모듈 초기화
        self.preprocessor = DocumentPreprocessor()
        self.printed_ocr = PrintedTextOCR(device=self.device)
        self.handwriting_ocr = HandwritingRecognizer(device=self.device)
        self.layout_analyzer = LayoutAnalyzer(device=self.device)
        self.ner = NamedEntityRecognizer(device=self.device)
        self.keyword_extractor = KeywordExtractor()
        self.classifier = DocumentClassifier()
        
        self.models_loaded = False
        logger.info(f"DocumentProcessor initialized on {self.device}")
    
    def load_models(self):
        """모든 모델 로드"""
        try:
            self.printed_ocr.load_model()
            self.handwriting_ocr.load_model()
            self.layout_analyzer.load_model()
            self.ner.load_model()
            
            self.models_loaded = True
            logger.info("All document models loaded")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def process(
        self,
        image_path: str,
        script_type: Optional[ScriptType] = None,
        languages: Optional[List[str]] = None
    ) -> OCRResult:
        """문서 처리"""
        import time
        start_time = time.time()
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 1. 전처리
        processed = self.preprocessor.preprocess(image)
        
        # 2. 필체 유형 자동 감지 (지정되지 않은 경우)
        if script_type is None:
            script_type = self._detect_script_type(processed)
        
        # 3. OCR 실행
        if script_type == ScriptType.PRINTED:
            text_blocks = self.printed_ocr.recognize(processed)
        else:
            text_blocks = self.handwriting_ocr.recognize(processed)
        
        # 4. 레이아웃 분석
        text_blocks = self.layout_analyzer.analyze(processed, text_blocks)
        
        # 5. 전체 텍스트 조합
        full_text = self._combine_text_blocks(text_blocks)
        
        # 6. 개체명 인식
        entities = self.ner.extract_entities(full_text)
        
        # 7. 키워드 추출
        keywords = self.keyword_extractor.extract(full_text)
        
        # 8. 문서 유형 분류
        doc_type = self.classifier.classify(full_text)
        
        # 9. 언어 감지
        detected_languages = self._detect_languages(text_blocks)
        
        # 10. 날짜 추정
        estimated_date = self._extract_date(entities)
        
        # 메타데이터 생성
        metadata = DocumentMetadata(
            document_type=doc_type,
            script_type=script_type,
            languages=detected_languages,
            estimated_date=estimated_date,
            keywords=[kw for kw, _ in keywords],
            entities=entities
        )
        
        # 평균 신뢰도
        avg_confidence = np.mean([b.confidence for b in text_blocks]) if text_blocks else 0.0
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            full_text=full_text,
            text_blocks=text_blocks,
            metadata=metadata,
            confidence=avg_confidence,
            processing_time=processing_time
        )
    
    def _detect_script_type(self, image: np.ndarray) -> ScriptType:
        """필체 유형 자동 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 에지 복잡도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (gray.shape[0] * gray.shape[1] * 255)
        
        # 수직선 비율 (인쇄체는 수직선이 많음)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        v_ratio = np.abs(sobel_y).sum() / (np.abs(sobel_x).sum() + np.abs(sobel_y).sum() + 1e-6)
        
        # 분류
        if edge_density < 0.03 and v_ratio > 0.55:
            return ScriptType.PRINTED
        else:
            return ScriptType.HANDWRITTEN_SEMI
    
    def _combine_text_blocks(self, blocks: List[TextBlock]) -> str:
        """텍스트 블록 조합"""
        if not blocks:
            return ""
        
        # Y 좌표로 정렬 후 조합
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y, b.bbox.x))
        
        lines = []
        current_y = -100
        current_line = []
        
        for block in sorted_blocks:
            if block.bbox.y - current_y > 20:  # 새 줄
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [block.text]
                current_y = block.bbox.y
            else:
                current_line.append(block.text)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def _detect_languages(self, blocks: List[TextBlock]) -> List[Language]:
        """언어 감지"""
        languages = set()
        for block in blocks:
            languages.add(block.language)
        
        return list(languages) if languages else [Language.KOREAN]
    
    def _extract_date(self, entities: List[NamedEntity]) -> Optional[str]:
        """날짜 추출"""
        date_entities = [e for e in entities if e.entity_type == "DAT"]
        if date_entities:
            return date_entities[0].text
        return None
    
    def batch_process(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None
    ) -> List[OCRResult]:
        """배치 처리"""
        results = []
        
        for i, path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {path}")
            
            try:
                result = self.process(path)
                results.append(result)
                
                # 결과 저장
                if output_dir:
                    self._save_result(result, path, output_dir)
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        return results
    
    def _save_result(self, result: OCRResult, image_path: str, output_dir: str):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 텍스트 저장
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.full_text)
        
        # JSON 메타데이터 저장
        json_path = os.path.join(output_dir, f"{base_name}.json")
        metadata_dict = {
            "document_type": result.metadata.document_type.value,
            "script_type": result.metadata.script_type.value,
            "languages": [l.value for l in result.metadata.languages],
            "estimated_date": result.metadata.estimated_date,
            "keywords": result.metadata.keywords,
            "entities": [
                {"text": e.text, "type": e.entity_type, "confidence": e.confidence}
                for e in result.metadata.entities
            ],
            "confidence": result.confidence,
            "processing_time": result.processing_time
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)


# CLI 인터페이스
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diaspora Archive Document OCR")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--script", choices=["printed", "handwritten"], help="Script type")
    parser.add_argument("--languages", nargs="+", default=["ko", "en"], help="Languages")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = DocumentProcessor({"device": args.device})
    processor.load_models()
    
    # 스크립트 타입
    script_type = None
    if args.script == "printed":
        script_type = ScriptType.PRINTED
    elif args.script == "handwritten":
        script_type = ScriptType.HANDWRITTEN_SEMI
    
    if os.path.isdir(args.input):
        # 디렉토리 배치 처리
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        output_dir = args.output or os.path.join(args.input, "ocr_results")
        results = processor.batch_process(image_paths, output_dir)
        print(f"Processed {len(results)} documents")
        
    else:
        # 단일 이미지 처리
        result = processor.process(args.input, script_type)
        
        print(f"\n=== OCR Result ===")
        print(f"Document Type: {result.metadata.document_type.value}")
        print(f"Script Type: {result.metadata.script_type.value}")
        print(f"Languages: {[l.value for l in result.metadata.languages]}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"\n--- Text ---")
        print(result.full_text[:500] + "..." if len(result.full_text) > 500 else result.full_text)
        print(f"\n--- Keywords ---")
        print(", ".join(result.metadata.keywords[:10]))
        print(f"\n--- Entities ---")
        for entity in result.metadata.entities[:10]:
            print(f"  [{entity.entity_type}] {entity.text}")
        
        # 결과 저장
        if args.output:
            processor._save_result(result, args.input, args.output)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
