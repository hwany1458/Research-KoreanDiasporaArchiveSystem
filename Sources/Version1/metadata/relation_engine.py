"""
관계 추론 엔진 (Relation Inference Engine)

자료 간 관계를 자동으로 추론하고 연결하는 엔진

주요 기능:
- 인물 연결 (얼굴 인식 기반)
- 시간적 연결 (날짜/시기 기반)
- 공간적 연결 (장소 기반)
- 주제적 연결 (키워드/의미 기반)
- 행사 연결 (이벤트 기반)

Author: Diaspora Archive Project
"""

import os
import re
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

import numpy as np

from .schema import UnifiedMetadata, PersonInfo, LocationInfo, EventInfo

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """관계 유형"""
    # 인물 관계
    DEPICTS = "depicts"
    CREATED_BY = "created_by"
    
    # 시간 관계
    SAME_DATE = "same_date"
    SAME_PERIOD = "same_period"
    BEFORE = "before"
    AFTER = "after"
    
    # 장소 관계
    SAME_LOCATION = "same_location"
    SAME_REGION = "same_region"
    
    # 행사 관계
    SAME_EVENT = "same_event"
    
    # 주제 관계
    SAME_TOPIC = "same_topic"
    SIMILAR_CONTENT = "similar_content"
    
    # 자료 관계
    PART_OF = "part_of"
    RELATED_TO = "related_to"


@dataclass
class Relation:
    """관계 정보"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 0.0
    evidence: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
            "is_verified": self.is_verified,
            "metadata": self.metadata
        }


class FaceRelationEngine:
    """얼굴 인식 기반 인물 연결"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.face_lib = None
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        logger.info(f"FaceRelationEngine initialized on {device}")
    
    def load_models(self):
        """얼굴 인식 모델 로드"""
        try:
            import face_recognition
            self.face_lib = face_recognition
            logger.info("face_recognition loaded")
        except ImportError:
            logger.warning("face_recognition not available")
    
    def extract_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """이미지에서 얼굴 추출"""
        faces = []
        if self.face_lib is None:
            return faces
        
        try:
            image = self.face_lib.load_image_file(image_path)
            locations = self.face_lib.face_locations(image)
            encodings = self.face_lib.face_encodings(image, locations)
            
            for i, (loc, enc) in enumerate(zip(locations, encodings)):
                top, right, bottom, left = loc
                faces.append({
                    "face_id": i,
                    "bbox": [left, top, right - left, bottom - top],
                    "encoding": enc.tolist(),
                    "confidence": 0.9
                })
        except Exception as e:
            logger.error(f"Face extraction error: {e}")
        
        return faces
    
    def register_person(self, person_id: str, face_encoding: List[float]):
        """인물 얼굴 등록"""
        if person_id not in self.known_faces:
            self.known_faces[person_id] = []
        self.known_faces[person_id].append(np.array(face_encoding))
    
    def identify_face(self, face_encoding: List[float], threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """얼굴 식별"""
        if not self.known_faces:
            return None, 0.0
        
        encoding = np.array(face_encoding)
        best_match = None
        best_distance = float('inf')
        
        for person_id, known_encodings in self.known_faces.items():
            for known in known_encodings:
                distance = np.linalg.norm(encoding - known)
                if distance < best_distance:
                    best_distance = distance
                    best_match = person_id
        
        if best_distance < threshold:
            return best_match, 1.0 - (best_distance / threshold)
        return None, 0.0
    
    def find_person_relations(self, items: List[UnifiedMetadata]) -> List[Relation]:
        """같은 인물이 등장하는 자료 연결"""
        relations = []
        face_to_items: Dict[str, List[str]] = defaultdict(list)
        
        for item in items:
            if hasattr(item.technical, 'face_regions'):
                for face in item.technical.face_regions:
                    if 'encoding' in face:
                        person_id, conf = self.identify_face(face['encoding'])
                        if person_id:
                            face_to_items[person_id].append(item.item_id)
                        else:
                            new_id = f"unknown_{len(self.known_faces)}"
                            self.register_person(new_id, face['encoding'])
                            face_to_items[new_id].append(item.item_id)
        
        for person_id, item_ids in face_to_items.items():
            if len(item_ids) > 1:
                for i, src_id in enumerate(item_ids):
                    for tgt_id in item_ids[i+1:]:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=RelationType.DEPICTS,
                            confidence=0.8,
                            evidence=f"Same person: {person_id}",
                            metadata={"person_id": person_id}
                        ))
        
        return relations


class TemporalRelationEngine:
    """시간 기반 관계 추론"""
    
    DATE_PATTERNS = [
        r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})',
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
        r'(\d{4})년\s*(\d{1,2})월',
        r'(\d{4})년',
    ]
    
    def __init__(self):
        logger.info("TemporalRelationEngine initialized")
    
    def parse_date(self, date_str: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """날짜 파싱"""
        if not date_str:
            return None, None, None
        
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                year = int(groups[0]) if len(groups) > 0 else None
                month = int(groups[1]) if len(groups) > 1 else None
                day = int(groups[2]) if len(groups) > 2 else None
                return year, month, day
        
        return None, None, None
    
    def extract_year(self, item: UnifiedMetadata) -> Optional[int]:
        """자료에서 연도 추출"""
        for date_field in [item.dublin_core.date, item.dublin_core.date_created, 
                          item.dublin_core.coverage_temporal]:
            if date_field:
                year, _, _ = self.parse_date(date_field)
                if year:
                    return year
        return None
    
    def find_temporal_relations(self, items: List[UnifiedMetadata], 
                                year_tolerance: int = 2) -> List[Relation]:
        """시간 기반 관계 찾기"""
        relations = []
        year_groups: Dict[int, List[str]] = defaultdict(list)
        
        for item in items:
            year = self.extract_year(item)
            if year:
                year_groups[year].append(item.item_id)
        
        # 같은 연도
        for year, item_ids in year_groups.items():
            if len(item_ids) > 1:
                for i, src_id in enumerate(item_ids):
                    for tgt_id in item_ids[i+1:]:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=RelationType.SAME_PERIOD,
                            confidence=0.9,
                            evidence=f"Same year: {year}",
                            metadata={"year": year}
                        ))
        
        # 인접 연도
        years = sorted(year_groups.keys())
        for i, year1 in enumerate(years):
            for year2 in years[i+1:]:
                if year2 - year1 <= year_tolerance:
                    for src_id in year_groups[year1]:
                        for tgt_id in year_groups[year2]:
                            relations.append(Relation(
                                relation_id=str(uuid.uuid4()),
                                source_id=src_id,
                                target_id=tgt_id,
                                relation_type=RelationType.SAME_PERIOD,
                                confidence=0.7 - (year2 - year1) * 0.1,
                                evidence=f"Similar period: {year1}-{year2}"
                            ))
        
        return relations


class SpatialRelationEngine:
    """장소 기반 관계 추론"""
    
    LOCATION_ALIASES = {
        "los angeles": "la", "로스앤젤레스": "la",
        "new york": "ny", "뉴욕": "ny",
        "연변": "yanbian", "사할린": "sakhalin"
    }
    
    def __init__(self):
        logger.info("SpatialRelationEngine initialized")
    
    def extract_location(self, item: UnifiedMetadata) -> Optional[str]:
        """장소 추출"""
        if item.dublin_core.coverage_spatial:
            return item.dublin_core.coverage_spatial
        if item.diaspora.locations:
            loc = item.diaspora.locations[0]
            return ", ".join(filter(None, [loc.city, loc.region, loc.country]))
        if item.diaspora.diaspora_region:
            return str(item.diaspora.diaspora_region)
        return None
    
    def normalize_location(self, location: str) -> str:
        """장소명 정규화"""
        location = location.lower().strip()
        for full, abbr in self.LOCATION_ALIASES.items():
            location = location.replace(full, abbr)
        return location
    
    def find_spatial_relations(self, items: List[UnifiedMetadata]) -> List[Relation]:
        """장소 기반 관계 찾기"""
        relations = []
        location_groups: Dict[str, List[str]] = defaultdict(list)
        
        for item in items:
            location = self.extract_location(item)
            if location:
                normalized = self.normalize_location(location)
                location_groups[normalized].append(item.item_id)
        
        for location, item_ids in location_groups.items():
            if len(item_ids) > 1:
                for i, src_id in enumerate(item_ids):
                    for tgt_id in item_ids[i+1:]:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=RelationType.SAME_LOCATION,
                            confidence=0.9,
                            evidence=f"Same location: {location}",
                            metadata={"location": location}
                        ))
        
        return relations


class TopicRelationEngine:
    """주제/키워드 기반 관계 추론"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.embedding_model = None
        logger.info(f"TopicRelationEngine initialized on {device}")
    
    def load_models(self):
        """임베딩 모델 로드"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            logger.info("Sentence transformer loaded")
        except ImportError:
            logger.warning("sentence-transformers not available")
    
    def extract_keywords(self, item: UnifiedMetadata) -> Set[str]:
        """키워드 추출"""
        keywords = set()
        keywords.update(item.dublin_core.subject)
        keywords.update(item.tags)
        keywords.update(item.diaspora.topics)
        return keywords
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        if self.embedding_model:
            embeddings = self.embedding_model.encode([text1, text2])
            return float(np.dot(embeddings[0], embeddings[1]) / 
                        (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        else:
            # 키워드 오버랩
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / min(len(words1), len(words2))
    
    def find_topic_relations(self, items: List[UnifiedMetadata], 
                            similarity_threshold: float = 0.5) -> List[Relation]:
        """주제 기반 관계 찾기"""
        relations = []
        
        # 키워드 기반
        keyword_map: Dict[str, List[str]] = defaultdict(list)
        for item in items:
            for kw in self.extract_keywords(item):
                keyword_map[kw.lower()].append(item.item_id)
        
        for keyword, item_ids in keyword_map.items():
            if len(item_ids) > 1 and len(keyword) > 2:
                for i, src_id in enumerate(item_ids):
                    for tgt_id in item_ids[i+1:]:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=RelationType.SAME_TOPIC,
                            confidence=0.7,
                            evidence=f"Shared keyword: {keyword}",
                            metadata={"keyword": keyword}
                        ))
        
        # 텍스트 유사도 기반
        items_with_text = [
            (item, item.diaspora.transcription or item.dublin_core.description or "")
            for item in items
            if item.diaspora.transcription or item.dublin_core.description
        ]
        
        for i, (item1, text1) in enumerate(items_with_text):
            for item2, text2 in items_with_text[i+1:]:
                if len(text1) > 50 and len(text2) > 50:
                    similarity = self.compute_similarity(text1[:500], text2[:500])
                    if similarity >= similarity_threshold:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=item1.item_id,
                            target_id=item2.item_id,
                            relation_type=RelationType.SIMILAR_CONTENT,
                            confidence=similarity,
                            evidence=f"Text similarity: {similarity:.2f}"
                        ))
        
        return relations


class EventRelationEngine:
    """행사 기반 관계 추론"""
    
    EVENT_KEYWORDS = {
        "wedding": ["결혼", "wedding", "혼례"],
        "birthday": ["생일", "birthday", "돌잔치"],
        "graduation": ["졸업", "graduation"],
        "seollal": ["설", "seollal", "새해"],
        "chuseok": ["추석", "chuseok", "한가위"]
    }
    
    def __init__(self):
        logger.info("EventRelationEngine initialized")
    
    def extract_events(self, item: UnifiedMetadata) -> List[EventInfo]:
        """행사 추출"""
        events = list(item.diaspora.events)
        
        # 제목에서 이벤트 추론
        title = item.dublin_core.title.lower()
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            if any(kw in title for kw in keywords):
                events.append(EventInfo(
                    name=item.dublin_core.title,
                    event_type=event_type,
                    date=item.dublin_core.date
                ))
                break
        
        return events
    
    def find_event_relations(self, items: List[UnifiedMetadata]) -> List[Relation]:
        """행사 기반 관계 찾기"""
        relations = []
        event_groups: Dict[str, List[Tuple[str, EventInfo]]] = defaultdict(list)
        
        for item in items:
            for event in self.extract_events(item):
                key = f"{event.event_type}_{event.date or 'unknown'}"
                event_groups[key].append((item.item_id, event))
        
        for event_key, items_events in event_groups.items():
            if len(items_events) > 1:
                for i, (src_id, src_event) in enumerate(items_events):
                    for tgt_id, _ in items_events[i+1:]:
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=RelationType.SAME_EVENT,
                            confidence=0.8,
                            evidence=f"Same event: {src_event.event_type}",
                            metadata={"event_type": str(src_event.event_type)}
                        ))
        
        return relations


class RelationInferenceEngine:
    """통합 관계 추론 엔진"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        
        self.face_engine = FaceRelationEngine(self.device)
        self.temporal_engine = TemporalRelationEngine()
        self.spatial_engine = SpatialRelationEngine()
        self.topic_engine = TopicRelationEngine(self.device)
        self.event_engine = EventRelationEngine()
        
        self.relations: List[Relation] = []
        self.relation_index: Dict[str, List[Relation]] = defaultdict(list)
        
        logger.info(f"RelationInferenceEngine initialized on {self.device}")
    
    def load_models(self):
        """모든 모델 로드"""
        self.face_engine.load_models()
        self.topic_engine.load_models()
        logger.info("All relation inference models loaded")
    
    def infer_relations(self, items: List[UnifiedMetadata],
                       relation_types: Optional[List[str]] = None) -> List[Relation]:
        """관계 추론 실행"""
        all_relations = []
        types = relation_types or ["face", "temporal", "spatial", "topic", "event"]
        
        if "face" in types:
            logger.info("Inferring face-based relations...")
            all_relations.extend(self.face_engine.find_person_relations(items))
        
        if "temporal" in types:
            logger.info("Inferring temporal relations...")
            all_relations.extend(self.temporal_engine.find_temporal_relations(items))
        
        if "spatial" in types:
            logger.info("Inferring spatial relations...")
            all_relations.extend(self.spatial_engine.find_spatial_relations(items))
        
        if "topic" in types:
            logger.info("Inferring topic-based relations...")
            all_relations.extend(self.topic_engine.find_topic_relations(items))
        
        if "event" in types:
            logger.info("Inferring event-based relations...")
            all_relations.extend(self.event_engine.find_event_relations(items))
        
        # 중복 제거
        unique = self._deduplicate(all_relations)
        
        # 인덱스 업데이트
        self.relations.extend(unique)
        for rel in unique:
            self.relation_index[rel.source_id].append(rel)
            self.relation_index[rel.target_id].append(rel)
        
        logger.info(f"Total relations found: {len(unique)}")
        return unique
    
    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        """중복 제거"""
        seen = set()
        unique = []
        for rel in relations:
            key = tuple(sorted([rel.source_id, rel.target_id])) + (rel.relation_type.value,)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique
    
    def get_relations_for_item(self, item_id: str) -> List[Relation]:
        """특정 자료의 관계 조회"""
        return self.relation_index.get(item_id, [])
    
    def get_related_items(self, item_id: str, 
                         relation_types: Optional[List[RelationType]] = None,
                         min_confidence: float = 0.0) -> List[Tuple[str, Relation]]:
        """관련 자료 조회"""
        related = []
        for rel in self.get_relations_for_item(item_id):
            if relation_types and rel.relation_type not in relation_types:
                continue
            if rel.confidence < min_confidence:
                continue
            other_id = rel.target_id if rel.source_id == item_id else rel.source_id
            related.append((other_id, rel))
        return sorted(related, key=lambda x: x[1].confidence, reverse=True)
    
    def export_relations(self, output_path: str):
        """관계 내보내기"""
        data = [rel.to_dict() for rel in self.relations]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(data)} relations to {output_path}")
    
    def import_relations(self, input_path: str):
        """관계 가져오기"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            rel = Relation(
                relation_id=item["relation_id"],
                source_id=item["source_id"],
                target_id=item["target_id"],
                relation_type=RelationType(item["relation_type"]),
                confidence=item["confidence"],
                evidence=item["evidence"],
                is_verified=item.get("is_verified", False),
                metadata=item.get("metadata", {})
            )
            self.relations.append(rel)
            self.relation_index[rel.source_id].append(rel)
            self.relation_index[rel.target_id].append(rel)
        
        logger.info(f"Imported {len(data)} relations")
