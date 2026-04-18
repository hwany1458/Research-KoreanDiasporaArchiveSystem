"""
메타데이터 스키마 정의 (Metadata Schema)

Dublin Core 기반 + 디아스포라 확장 메타데이터 스키마

Author: Diaspora Archive Project
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# ============ 열거형 정의 ============

class ResourceType(str, Enum):
    """자원 유형 (Dublin Core)"""
    IMAGE = "Image"
    TEXT = "Text"
    MOVING_IMAGE = "MovingImage"
    SOUND = "Sound"
    COLLECTION = "Collection"


class MaterialType(str, Enum):
    """자료 유형"""
    PHOTOGRAPH = "photograph"
    PORTRAIT = "portrait"
    GROUP_PHOTO = "group_photo"
    EVENT_PHOTO = "event_photo"
    LETTER = "letter"
    POSTCARD = "postcard"
    DIARY = "diary"
    CERTIFICATE = "certificate"
    OFFICIAL_DOCUMENT = "official_document"
    HOME_VIDEO = "home_video"
    EVENT_VIDEO = "event_video"
    RECORDING = "recording"
    ORAL_HISTORY = "oral_history"
    OTHER = "other"


class DiasporaGeneration(int, Enum):
    """이민 세대"""
    UNKNOWN = 0
    FIRST = 1
    ONE_POINT_FIVE = 15
    SECOND = 2
    THIRD = 3


class DiasporaRegion(str, Enum):
    """디아스포라 지역"""
    USA = "usa"
    USA_LA = "usa_la"
    USA_NY = "usa_ny"
    CANADA = "canada"
    JAPAN = "japan"
    CHINA = "china"
    CHINA_YANBIAN = "china_yanbian"
    RUSSIA = "russia"
    RUSSIA_SAKHALIN = "russia_sakhalin"
    KAZAKHSTAN = "kazakhstan"
    UZBEKISTAN = "uzbekistan"
    GERMANY = "germany"
    OTHER = "other"


class Language(str, Enum):
    """언어"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    RUSSIAN = "ru"
    MIXED = "mixed"


class RightsStatus(str, Enum):
    """권리 상태"""
    FAMILY_OWNED = "family_owned"
    DONATED = "donated"
    PUBLIC_DOMAIN = "public_domain"
    RESTRICTED = "restricted"


class ProcessingStatus(str, Enum):
    """처리 상태"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    VERIFIED = "verified"
    PUBLISHED = "published"


class EmotionType(str, Enum):
    """감정/분위기"""
    HAPPY = "happy"
    SAD = "sad"
    NOSTALGIC = "nostalgic"
    PROUD = "proud"
    NEUTRAL = "neutral"


class EventType(str, Enum):
    """행사 유형"""
    WEDDING = "wedding"
    BIRTHDAY = "birthday"
    FUNERAL = "funeral"
    GRADUATION = "graduation"
    DEPARTURE = "departure"
    ARRIVAL = "arrival"
    COMMUNITY_EVENT = "community_event"
    CHURCH_EVENT = "church_event"
    SEOLLAL = "seollal"
    CHUSEOK = "chuseok"
    OTHER = "other"


# ============ 하위 모델 ============

class PersonInfo(BaseModel):
    """인물 정보"""
    person_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name_korean: Optional[str] = None
    name_romanized: Optional[str] = None
    name_local: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    gender: Optional[str] = None
    generation: DiasporaGeneration = DiasporaGeneration.UNKNOWN
    birth_place: Optional[str] = None
    migration_year: Optional[int] = None
    migration_destination: Optional[DiasporaRegion] = None
    family_role: Optional[str] = None
    relation_to_collector: Optional[str] = None
    occupation: Optional[str] = None
    face_encoding: Optional[List[float]] = None
    face_image_path: Optional[str] = None

    class Config:
        use_enum_values = True


class LocationInfo(BaseModel):
    """장소 정보"""
    location_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name_korean: Optional[str] = None
    name_english: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_type: Optional[str] = None
    diaspora_region: Optional[DiasporaRegion] = None

    class Config:
        use_enum_values = True


class EventInfo(BaseModel):
    """행사/사건 정보"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    event_type: EventType = EventType.OTHER
    date: Optional[str] = None
    year: Optional[int] = None
    location_id: Optional[str] = None
    location_name: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    class Config:
        use_enum_values = True


class MigrationInfo(BaseModel):
    """이주 정보"""
    origin_country: str = "Korea"
    origin_region: Optional[str] = None
    destination_country: Optional[str] = None
    destination_region: Optional[DiasporaRegion] = None
    migration_year: Optional[int] = None
    migration_reason: Optional[str] = None
    generation: DiasporaGeneration = DiasporaGeneration.FIRST

    class Config:
        use_enum_values = True


# ============ Dublin Core 메타데이터 ============

class DublinCoreMetadata(BaseModel):
    """Dublin Core 15개 핵심 요소"""
    # 1. Title
    title: str = Field(..., description="자료의 제목")
    title_alternative: Optional[str] = None
    
    # 2. Creator
    creator: Optional[str] = None
    creator_id: Optional[str] = None
    
    # 3. Subject
    subject: List[str] = Field(default_factory=list)
    
    # 4. Description
    description: Optional[str] = None
    
    # 5. Publisher
    publisher: Optional[str] = None
    
    # 6. Contributor
    contributor: List[str] = Field(default_factory=list)
    
    # 7. Date
    date: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    
    # 8. Type
    type: ResourceType = ResourceType.IMAGE
    
    # 9. Format
    format: Optional[str] = None
    extent: Optional[str] = None
    
    # 10. Identifier
    identifier: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # 11. Source
    source: Optional[str] = None
    
    # 12. Language
    language: List[Language] = Field(default_factory=lambda: [Language.KOREAN])
    
    # 13. Relation
    relation: List[str] = Field(default_factory=list)
    is_part_of: Optional[str] = None
    has_part: List[str] = Field(default_factory=list)
    
    # 14. Coverage
    coverage_spatial: Optional[str] = None
    coverage_temporal: Optional[str] = None
    
    # 15. Rights
    rights: RightsStatus = RightsStatus.FAMILY_OWNED
    rights_holder: Optional[str] = None

    class Config:
        use_enum_values = True


# ============ 디아스포라 확장 ============

class DiasporaExtension(BaseModel):
    """디아스포라 확장 필드"""
    migration: Optional[MigrationInfo] = None
    diaspora_region: Optional[DiasporaRegion] = None
    
    persons: List[PersonInfo] = Field(default_factory=list)
    person_ids: List[str] = Field(default_factory=list)
    person_count: Optional[int] = None
    
    locations: List[LocationInfo] = Field(default_factory=list)
    location_ids: List[str] = Field(default_factory=list)
    
    events: List[EventInfo] = Field(default_factory=list)
    event_ids: List[str] = Field(default_factory=list)
    
    topics: List[str] = Field(default_factory=list)
    emotion: Optional[EmotionType] = None
    
    transcription: Optional[str] = None
    transcription_language: Optional[Language] = None
    key_quotes: List[Dict[str, Any]] = Field(default_factory=list)
    
    ai_generated: Dict[str, Any] = Field(default_factory=dict)
    user_verified: Dict[str, bool] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# ============ 기술 메타데이터 ============

class ImageTechnicalMetadata(BaseModel):
    """이미지 기술 메타데이터"""
    width: int
    height: int
    color_mode: str = "RGB"
    dpi: Optional[int] = None
    file_size: int = 0
    file_format: str = "JPEG"
    is_color: bool = True
    is_restored: bool = False
    is_colorized: bool = False
    quality_score: Optional[float] = None
    faces_detected: int = 0
    face_regions: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentTechnicalMetadata(BaseModel):
    """문서 기술 메타데이터"""
    width: int
    height: int
    page_count: int = 1
    dpi: Optional[int] = None
    file_size: int = 0
    file_format: str = "JPEG"
    script_type: Optional[str] = None
    ocr_performed: bool = False
    ocr_engine: Optional[str] = None
    ocr_confidence: Optional[float] = None
    word_count: Optional[int] = None


class AudioTechnicalMetadata(BaseModel):
    """오디오 기술 메타데이터"""
    duration: float
    duration_formatted: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1
    file_size: int = 0
    file_format: str = "WAV"
    is_enhanced: bool = False
    transcription_performed: bool = False
    transcription_engine: Optional[str] = None
    word_count: Optional[int] = None
    speaker_count: Optional[int] = None
    speaker_labels: List[str] = Field(default_factory=list)


class VideoTechnicalMetadata(BaseModel):
    """비디오 기술 메타데이터"""
    duration: float
    duration_formatted: Optional[str] = None
    width: int
    height: int
    fps: float
    file_size: int = 0
    file_format: str = "MP4"
    video_codec: str = "h264"
    has_audio: bool = True
    is_enhanced: bool = False
    is_stabilized: bool = False
    scene_count: Optional[int] = None
    keyframe_count: Optional[int] = None
    has_subtitles: bool = False


# ============ 통합 메타데이터 ============

class UnifiedMetadata(BaseModel):
    """통합 메타데이터 스키마"""
    schema_version: str = "1.0.0"
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    
    dublin_core: DublinCoreMetadata
    material_type: MaterialType = MaterialType.OTHER
    diaspora: DiasporaExtension = Field(default_factory=DiasporaExtension)
    
    technical: Union[
        ImageTechnicalMetadata,
        DocumentTechnicalMetadata,
        AudioTechnicalMetadata,
        VideoTechnicalMetadata,
        Dict[str, Any]
    ] = Field(default_factory=dict)
    
    file_path: Optional[str] = None
    original_filename: Optional[str] = None
    file_hash: Optional[str] = None
    storage_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    
    collection_id: Optional[str] = None
    collection_name: Optional[str] = None
    
    related_items: List[str] = Field(default_factory=list)
    related_persons: List[str] = Field(default_factory=list)
    related_locations: List[str] = Field(default_factory=list)
    related_events: List[str] = Field(default_factory=list)
    
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    def to_json(self, indent: int = 2) -> str:
        return self.json(indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "UnifiedMetadata":
        return cls.parse_raw(json_str)

    def update_timestamp(self):
        self.updated_at = datetime.now()

    def compute_file_hash(self) -> str:
        if self.file_path and os.path.exists(self.file_path):
            hash_func = hashlib.md5()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            self.file_hash = hash_func.hexdigest()
            return self.file_hash
        return ""

    def get_search_text(self) -> str:
        parts = [
            self.dublin_core.title,
            self.dublin_core.description or "",
            " ".join(self.dublin_core.subject),
            self.diaspora.transcription or "",
            " ".join(self.tags),
        ]
        for person in self.diaspora.persons:
            if person.name_korean:
                parts.append(person.name_korean)
        return " ".join(filter(None, parts))


# ============ 팩토리 ============

class MetadataFactory:
    """메타데이터 생성 팩토리"""

    @staticmethod
    def create_for_image(title: str, file_path: str, width: int, height: int, **kwargs) -> UnifiedMetadata:
        dublin_core = DublinCoreMetadata(title=title, type=ResourceType.IMAGE, format="image/jpeg")
        technical = ImageTechnicalMetadata(width=width, height=height)
        return UnifiedMetadata(
            dublin_core=dublin_core,
            material_type=kwargs.get("material_type", MaterialType.PHOTOGRAPH),
            technical=technical,
            file_path=file_path,
            original_filename=os.path.basename(file_path) if file_path else None
        )

    @staticmethod
    def create_for_document(title: str, file_path: str, width: int, height: int, 
                           transcription: str = None, **kwargs) -> UnifiedMetadata:
        dublin_core = DublinCoreMetadata(title=title, type=ResourceType.TEXT)
        technical = DocumentTechnicalMetadata(width=width, height=height, 
                                              ocr_performed=transcription is not None)
        diaspora = DiasporaExtension(transcription=transcription)
        return UnifiedMetadata(
            dublin_core=dublin_core,
            material_type=kwargs.get("material_type", MaterialType.LETTER),
            technical=technical,
            diaspora=diaspora,
            file_path=file_path
        )

    @staticmethod
    def create_for_audio(title: str, file_path: str, duration: float, 
                        transcription: str = None, **kwargs) -> UnifiedMetadata:
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
        formatted = f"{h:02d}:{m:02d}:{s:02d}"
        dublin_core = DublinCoreMetadata(title=title, type=ResourceType.SOUND, extent=formatted)
        technical = AudioTechnicalMetadata(duration=duration, duration_formatted=formatted,
                                           transcription_performed=transcription is not None)
        diaspora = DiasporaExtension(transcription=transcription)
        return UnifiedMetadata(
            dublin_core=dublin_core,
            material_type=kwargs.get("material_type", MaterialType.RECORDING),
            technical=technical,
            diaspora=diaspora,
            file_path=file_path
        )

    @staticmethod
    def create_for_video(title: str, file_path: str, duration: float, 
                        width: int, height: int, fps: float, **kwargs) -> UnifiedMetadata:
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
        formatted = f"{h:02d}:{m:02d}:{s:02d}"
        dublin_core = DublinCoreMetadata(title=title, type=ResourceType.MOVING_IMAGE, extent=formatted)
        technical = VideoTechnicalMetadata(duration=duration, duration_formatted=formatted,
                                           width=width, height=height, fps=fps)
        return UnifiedMetadata(
            dublin_core=dublin_core,
            material_type=kwargs.get("material_type", MaterialType.HOME_VIDEO),
            technical=technical,
            file_path=file_path
        )
