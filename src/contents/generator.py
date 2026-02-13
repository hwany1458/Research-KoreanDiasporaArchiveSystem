"""
인터랙티브 스토리 생성기 (Interactive Story Generator)

AI 기반 디아스포라 서사 자동 생성 시스템

주요 기능:
- 자료 기반 내러티브 자동 생성
- 타임라인 스토리 구성
- 인물 중심 가족사 생성
- 이주 여정 스토리텔링
- 멀티미디어 통합 스토리
- 분기형/인터랙티브 스토리

Author: Diaspora Archive Project
"""

import os
import logging
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class StoryType(str, Enum):
    """스토리 유형"""
    FAMILY_HISTORY = "family_history"       # 가족사
    MIGRATION_JOURNEY = "migration_journey" # 이주 여정
    LIFE_CHRONICLE = "life_chronicle"       # 생애 연대기
    THEMATIC = "thematic"                   # 주제별 (명절, 결혼 등)
    PLACE_BASED = "place_based"            # 장소 중심
    INTERACTIVE = "interactive"             # 분기형 인터랙티브


class NarrativeStyle(str, Enum):
    """서술 스타일"""
    DOCUMENTARY = "documentary"     # 다큐멘터리 (객관적)
    PERSONAL = "personal"           # 개인적 (1인칭)
    STORYTELLING = "storytelling"   # 이야기체 (3인칭)
    POETIC = "poetic"              # 시적/감성적
    EDUCATIONAL = "educational"     # 교육적/설명적


@dataclass
class StorySegment:
    """스토리 세그먼트 (장면)"""
    segment_id: str
    title: str
    content: str
    narration: Optional[str] = None
    
    # 연결된 자료
    item_ids: List[str] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    
    # 메타데이터
    year: Optional[int] = None
    location: Optional[str] = None
    persons: List[str] = field(default_factory=list)
    
    # 인터랙티브 요소
    choices: List[Dict[str, str]] = field(default_factory=list)  # 분기 선택지
    next_segment_id: Optional[str] = None
    
    # 인용구
    quotes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "content": self.content,
            "narration": self.narration,
            "item_ids": self.item_ids,
            "image_urls": self.image_urls,
            "audio_url": self.audio_url,
            "video_url": self.video_url,
            "year": self.year,
            "location": self.location,
            "persons": self.persons,
            "choices": self.choices,
            "next_segment_id": self.next_segment_id,
            "quotes": self.quotes
        }


@dataclass
class Story:
    """스토리"""
    story_id: str
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None
    
    story_type: StoryType = StoryType.FAMILY_HISTORY
    narrative_style: NarrativeStyle = NarrativeStyle.STORYTELLING
    
    # 세그먼트
    segments: List[StorySegment] = field(default_factory=list)
    
    # 메타데이터
    main_persons: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    time_range: Tuple[Optional[int], Optional[int]] = (None, None)
    
    # 태그
    tags: List[str] = field(default_factory=list)
    
    # 생성 정보
    created_at: datetime = field(default_factory=datetime.now)
    generated_by: str = "ai"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "story_type": self.story_type.value,
            "narrative_style": self.narrative_style.value,
            "segments": [s.to_dict() for s in self.segments],
            "main_persons": self.main_persons,
            "locations": self.locations,
            "time_range": list(self.time_range),
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "generated_by": self.generated_by
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class PromptTemplates:
    """프롬프트 템플릿"""
    
    FAMILY_HISTORY_INTRO = """
다음 정보를 바탕으로 {family_name} 가족의 역사에 대한 감동적인 서문을 작성해주세요.

가족 정보:
- 1세대: {first_gen}
- 이주 시기: {migration_year}
- 이주 지역: {destination}
- 출신 지역: {origin}

자료 수: {item_count}개
시간 범위: {year_from} ~ {year_to}

서문은 200자 내외로, 따뜻하고 존경하는 어조로 작성해주세요.
"""

    MIGRATION_JOURNEY = """
다음 정보를 바탕으로 이주 여정 이야기를 작성해주세요.

이주자: {person_name}
출발지: {origin}
도착지: {destination}
이주 연도: {year}
이주 이유: {reason}

관련 자료:
{items_description}

이주 과정의 어려움과 희망을 담아 300자 내외로 작성해주세요.
"""

    PHOTO_DESCRIPTION = """
다음 사진에 대한 설명을 작성해주세요.

제목: {title}
날짜: {date}
장소: {location}
등장인물: {persons}
기존 설명: {description}

사진 속 순간의 감정과 의미를 담아 150자 내외로 작성해주세요.
"""

    TIMELINE_NARRATION = """
{year}년, {location}에서의 기록입니다.

관련 자료:
{items}

이 시기의 상황과 의미를 100자 내외로 설명해주세요.
"""

    ORAL_HISTORY_HIGHLIGHT = """
다음 구술 기록에서 핵심 내용을 추출하고 요약해주세요.

화자: {speaker}
주제: {topic}
전사 내용:
{transcription}

핵심 인용구 3개와 요약(200자)을 작성해주세요.
"""

    INTERACTIVE_CHOICE = """
스토리의 다음 장면을 위한 선택지를 만들어주세요.

현재 상황: {current_situation}
시간: {year}
장소: {location}
인물: {persons}

사용자가 선택할 수 있는 2-3개의 분기점을 제안해주세요.
각 선택지는 다른 관점이나 시간대로 이동합니다.
"""


class LLMClient:
    """LLM 클라이언트 (OpenAI/Claude API)"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.model = self.config.get("model", "gpt-4")
        self.client = None
        
        logger.info(f"LLMClient initialized (model: {self.model})")
    
    def initialize(self):
        """클라이언트 초기화"""
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("openai not installed")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """텍스트 생성"""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 한인 디아스포라 역사를 전문으로 하는 작가입니다. 따뜻하고 존경하는 어조로 가족사와 이주 역사를 서술합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
        
        # 폴백: 템플릿 기반 생성
        return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """폴백 생성 (LLM 없을 때)"""
        # 간단한 템플릿 기반 생성
        if "가족의 역사" in prompt:
            return "이 가족의 이야기는 희망과 도전의 여정입니다. 낯선 땅에서 새로운 삶을 일구어낸 그들의 용기와 사랑이 이 기록들 속에 담겨 있습니다."
        elif "이주 여정" in prompt:
            return "고향을 떠나 새로운 삶을 찾아 나선 여정은 쉽지 않았습니다. 그러나 더 나은 미래를 향한 희망이 그들을 이끌었습니다."
        elif "사진" in prompt:
            return "이 사진은 소중한 순간을 담고 있습니다. 시간이 흘러도 변하지 않는 가족의 사랑이 느껴집니다."
        else:
            return "이 기록은 우리 가족의 소중한 역사를 담고 있습니다."


class StoryGenerator:
    """스토리 생성기"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.llm = LLMClient(self.config.get("llm", {}))
        self.templates = PromptTemplates()
        
        logger.info("StoryGenerator initialized")
    
    def initialize(self):
        """초기화"""
        self.llm.initialize()
    
    def generate_family_history(
        self,
        items: List[Dict[str, Any]],
        persons: List[Dict[str, Any]],
        family_name: str = "가족"
    ) -> Story:
        """가족사 스토리 생성"""
        import uuid
        
        story = Story(
            story_id=str(uuid.uuid4()),
            title=f"{family_name}의 이야기",
            subtitle="사진과 기록으로 보는 가족의 역사",
            story_type=StoryType.FAMILY_HISTORY,
            narrative_style=NarrativeStyle.STORYTELLING
        )
        
        # 시간 범위 계산
        years = []
        for item in items:
            date = item.get("date", "")
            if date:
                try:
                    years.append(int(date[:4]))
                except:
                    pass
        
        if years:
            story.time_range = (min(years), max(years))
        
        # 1세대 찾기
        first_gen = [p for p in persons if p.get("generation") == 1]
        first_gen_names = ", ".join([p.get("name_korean", "") for p in first_gen[:2]])
        
        # 서문 생성
        intro_prompt = self.templates.FAMILY_HISTORY_INTRO.format(
            family_name=family_name,
            first_gen=first_gen_names or "알 수 없음",
            migration_year=first_gen[0].get("migration_year", "알 수 없음") if first_gen else "알 수 없음",
            destination=first_gen[0].get("migration_destination", "알 수 없음") if first_gen else "알 수 없음",
            origin=first_gen[0].get("birth_place", "알 수 없음") if first_gen else "알 수 없음",
            item_count=len(items),
            year_from=story.time_range[0] or "?",
            year_to=story.time_range[1] or "?"
        )
        
        intro_content = self.llm.generate(intro_prompt)
        
        # 서문 세그먼트
        intro_segment = StorySegment(
            segment_id="intro",
            title="프롤로그",
            content=intro_content,
            persons=[p.get("name_korean", "") for p in first_gen]
        )
        story.segments.append(intro_segment)
        
        # 연대순 정렬
        sorted_items = sorted(items, key=lambda x: x.get("date", "0000"))
        
        # 연도별 그룹화
        year_groups = {}
        for item in sorted_items:
            date = item.get("date", "")
            if date:
                try:
                    year = int(date[:4])
                    decade = (year // 10) * 10
                    if decade not in year_groups:
                        year_groups[decade] = []
                    year_groups[decade].append(item)
                except:
                    pass
        
        # 연대별 세그먼트 생성
        for decade in sorted(year_groups.keys()):
            decade_items = year_groups[decade]
            
            segment = StorySegment(
                segment_id=f"decade_{decade}",
                title=f"{decade}년대",
                content=self._generate_decade_narrative(decade, decade_items),
                year=decade,
                item_ids=[item.get("item_id", "") for item in decade_items],
                image_urls=[item.get("thumbnail_url", "") for item in decade_items if item.get("thumbnail_url")]
            )
            
            # 인용구 추가 (구술 자료에서)
            for item in decade_items:
                if item.get("material_type") == "oral_history" and item.get("transcription"):
                    quotes = self._extract_quotes(item.get("transcription", ""))
                    segment.quotes.extend(quotes[:2])
            
            story.segments.append(segment)
        
        # 에필로그
        epilogue = StorySegment(
            segment_id="epilogue",
            title="에필로그",
            content=f"이렇게 {family_name}의 이야기는 계속됩니다. 과거의 기록들은 미래 세대에게 전해질 소중한 유산입니다."
        )
        story.segments.append(epilogue)
        
        # 메타데이터
        story.main_persons = [p.get("name_korean", "") for p in persons[:5]]
        story.tags = ["가족사", "디아스포라", "이민"]
        
        return story
    
    def generate_migration_story(
        self,
        person: Dict[str, Any],
        items: List[Dict[str, Any]]
    ) -> Story:
        """이주 여정 스토리 생성"""
        import uuid
        
        name = person.get("name_korean", "이주자")
        
        story = Story(
            story_id=str(uuid.uuid4()),
            title=f"{name}의 이주 여정",
            subtitle="고향을 떠나 새로운 삶을 향해",
            story_type=StoryType.MIGRATION_JOURNEY,
            narrative_style=NarrativeStyle.PERSONAL
        )
        
        # 출발 세그먼트
        departure = StorySegment(
            segment_id="departure",
            title="떠남",
            content=self._generate_departure_narrative(person),
            year=person.get("migration_year"),
            location=person.get("birth_place")
        )
        story.segments.append(departure)
        
        # 여정 세그먼트
        journey = StorySegment(
            segment_id="journey",
            title="여정",
            content=self._generate_journey_narrative(person)
        )
        story.segments.append(journey)
        
        # 정착 세그먼트
        settlement = StorySegment(
            segment_id="settlement",
            title="정착",
            content=self._generate_settlement_narrative(person, items),
            location=str(person.get("migration_destination", ""))
        )
        story.segments.append(settlement)
        
        # 관련 자료 연결
        for item in items:
            item_year = None
            if item.get("date"):
                try:
                    item_year = int(item.get("date")[:4])
                except:
                    pass
            
            migration_year = person.get("migration_year")
            
            if item_year and migration_year:
                if item_year <= migration_year:
                    departure.item_ids.append(item.get("item_id", ""))
                elif item_year <= migration_year + 5:
                    settlement.item_ids.append(item.get("item_id", ""))
        
        story.main_persons = [name]
        story.time_range = (person.get("migration_year"), None)
        story.tags = ["이주", "여정", person.get("migration_destination", "")]
        
        return story
    
    def generate_timeline_story(
        self,
        items: List[Dict[str, Any]],
        title: str = "시간의 기록"
    ) -> Story:
        """타임라인 스토리 생성"""
        import uuid
        
        story = Story(
            story_id=str(uuid.uuid4()),
            title=title,
            story_type=StoryType.LIFE_CHRONICLE,
            narrative_style=NarrativeStyle.DOCUMENTARY
        )
        
        # 연도순 정렬
        sorted_items = sorted(items, key=lambda x: x.get("date", "0000"))
        
        for i, item in enumerate(sorted_items):
            date = item.get("date", "")
            year = None
            if date:
                try:
                    year = int(date[:4])
                except:
                    pass
            
            segment = StorySegment(
                segment_id=f"event_{i}",
                title=item.get("title", f"기록 {i+1}"),
                content=item.get("description") or self._generate_item_description(item),
                year=year,
                location=item.get("location"),
                item_ids=[item.get("item_id", "")],
                image_urls=[item.get("thumbnail_url")] if item.get("thumbnail_url") else []
            )
            
            # 다음 세그먼트 연결
            if i < len(sorted_items) - 1:
                segment.next_segment_id = f"event_{i+1}"
            
            story.segments.append(segment)
        
        return story
    
    def generate_interactive_story(
        self,
        items: List[Dict[str, Any]],
        persons: List[Dict[str, Any]],
        title: str = "선택의 여정"
    ) -> Story:
        """분기형 인터랙티브 스토리 생성"""
        import uuid
        
        story = Story(
            story_id=str(uuid.uuid4()),
            title=title,
            subtitle="당신의 선택으로 이야기가 펼쳐집니다",
            story_type=StoryType.INTERACTIVE,
            narrative_style=NarrativeStyle.STORYTELLING
        )
        
        # 시작 세그먼트
        start = StorySegment(
            segment_id="start",
            title="이야기의 시작",
            content="어느 가족의 역사 속으로 들어가봅니다. 어떤 이야기를 먼저 들어볼까요?",
            choices=[
                {"id": "person_focus", "text": "사람들의 이야기", "next": "person_intro"},
                {"id": "time_focus", "text": "시간순으로 보기", "next": "timeline_start"},
                {"id": "place_focus", "text": "장소별로 탐험하기", "next": "place_intro"}
            ]
        )
        story.segments.append(start)
        
        # 인물 중심 분기
        if persons:
            person_intro = StorySegment(
                segment_id="person_intro",
                title="사람들",
                content="이 가족에는 여러 사람들이 있습니다. 누구의 이야기를 들어볼까요?",
                choices=[
                    {"id": f"person_{i}", "text": p.get("name_korean", f"인물 {i+1}"), "next": f"person_{i}_story"}
                    for i, p in enumerate(persons[:3])
                ]
            )
            story.segments.append(person_intro)
            
            # 각 인물별 세그먼트
            for i, person in enumerate(persons[:3]):
                person_story = StorySegment(
                    segment_id=f"person_{i}_story",
                    title=person.get("name_korean", f"인물 {i+1}"),
                    content=self._generate_person_intro(person),
                    persons=[person.get("name_korean", "")],
                    choices=[
                        {"id": "back", "text": "다른 사람 보기", "next": "person_intro"},
                        {"id": "timeline", "text": "이 사람의 타임라인", "next": f"person_{i}_timeline"}
                    ]
                )
                story.segments.append(person_story)
        
        # 타임라인 분기
        timeline_start = StorySegment(
            segment_id="timeline_start",
            title="시간의 흐름",
            content="시간순으로 기록을 살펴봅니다.",
            choices=[]
        )
        
        # 연대별 선택지
        decades = set()
        for item in items:
            date = item.get("date", "")
            if date:
                try:
                    decade = (int(date[:4]) // 10) * 10
                    decades.add(decade)
                except:
                    pass
        
        for decade in sorted(decades)[:4]:
            timeline_start.choices.append({
                "id": f"decade_{decade}",
                "text": f"{decade}년대",
                "next": f"decade_{decade}_story"
            })
        
        story.segments.append(timeline_start)
        
        return story
    
    def generate_thematic_story(
        self,
        items: List[Dict[str, Any]],
        theme: str,
        title: Optional[str] = None
    ) -> Story:
        """주제별 스토리 생성"""
        import uuid
        
        theme_titles = {
            "wedding": "결혼, 새로운 시작",
            "graduation": "졸업, 성장의 기록",
            "holiday": "명절, 함께하는 시간",
            "immigration": "이민, 새로운 땅에서",
            "childhood": "어린 시절의 추억"
        }
        
        story = Story(
            story_id=str(uuid.uuid4()),
            title=title or theme_titles.get(theme, f"{theme} 이야기"),
            story_type=StoryType.THEMATIC,
            narrative_style=NarrativeStyle.STORYTELLING,
            tags=[theme]
        )
        
        # 주제 관련 자료 필터링
        theme_items = [
            item for item in items
            if theme.lower() in str(item.get("tags", [])).lower() or
               theme.lower() in str(item.get("title", "")).lower() or
               theme.lower() in str(item.get("description", "")).lower()
        ]
        
        if not theme_items:
            theme_items = items[:5]
        
        # 도입
        intro = StorySegment(
            segment_id="intro",
            title="이야기의 시작",
            content=self._generate_theme_intro(theme)
        )
        story.segments.append(intro)
        
        # 자료별 세그먼트
        for i, item in enumerate(theme_items):
            segment = StorySegment(
                segment_id=f"item_{i}",
                title=item.get("title", f"기록 {i+1}"),
                content=item.get("description") or self._generate_item_description(item),
                item_ids=[item.get("item_id", "")],
                image_urls=[item.get("thumbnail_url")] if item.get("thumbnail_url") else []
            )
            story.segments.append(segment)
        
        return story
    
    # ===== 내부 생성 함수 =====
    
    def _generate_decade_narrative(self, decade: int, items: List[Dict]) -> str:
        """연대별 내러티브 생성"""
        count = len(items)
        types = set(item.get("material_type", "") for item in items)
        
        type_desc = []
        if "photograph" in types or "group_photo" in types:
            type_desc.append("사진")
        if "letter" in types:
            type_desc.append("편지")
        if "oral_history" in types:
            type_desc.append("구술 기록")
        if "home_video" in types:
            type_desc.append("영상")
        
        type_str = ", ".join(type_desc) if type_desc else "자료"
        
        prompt = f"{decade}년대의 기록입니다. {count}개의 {type_str}이(가) 남아있습니다. 이 시기의 가족 이야기를 150자 내외로 서술해주세요."
        
        return self.llm.generate(prompt, max_tokens=200)
    
    def _generate_departure_narrative(self, person: Dict) -> str:
        """출발 내러티브 생성"""
        name = person.get("name_korean", "")
        origin = person.get("birth_place", "고향")
        year = person.get("migration_year", "")
        
        return self.llm.generate(
            f"{name}은(는) {year}년, {origin}을(를) 떠나기로 결심했습니다. 떠나는 마음을 담아 150자 내외로 서술해주세요.",
            max_tokens=200
        )
    
    def _generate_journey_narrative(self, person: Dict) -> str:
        """여정 내러티브 생성"""
        return self.llm.generate(
            "낯선 땅을 향한 긴 여정이 시작되었습니다. 여정의 어려움과 희망을 담아 150자 내외로 서술해주세요.",
            max_tokens=200
        )
    
    def _generate_settlement_narrative(self, person: Dict, items: List[Dict]) -> str:
        """정착 내러티브 생성"""
        destination = person.get("migration_destination", "새로운 땅")
        
        return self.llm.generate(
            f"{destination}에서의 새로운 삶이 시작되었습니다. 정착 과정의 도전과 성취를 담아 150자 내외로 서술해주세요.",
            max_tokens=200
        )
    
    def _generate_item_description(self, item: Dict) -> str:
        """자료 설명 생성"""
        title = item.get("title", "")
        date = item.get("date", "")
        location = item.get("location", "")
        
        prompt = f"제목: {title}, 날짜: {date}, 장소: {location}. 이 자료에 대해 100자 내외로 설명해주세요."
        return self.llm.generate(prompt, max_tokens=150)
    
    def _generate_person_intro(self, person: Dict) -> str:
        """인물 소개 생성"""
        name = person.get("name_korean", "")
        birth_year = person.get("birth_year", "")
        generation = person.get("generation", 0)
        
        gen_str = {0: "", 1: "1세대", 2: "2세대", 3: "3세대", 15: "1.5세대"}.get(generation, "")
        
        return self.llm.generate(
            f"{name}({birth_year}년생, {gen_str})에 대해 100자 내외로 소개해주세요.",
            max_tokens=150
        )
    
    def _generate_theme_intro(self, theme: str) -> str:
        """주제 도입 생성"""
        theme_intros = {
            "wedding": "결혼은 새로운 가정의 시작입니다. 이 가족의 결혼 이야기를 들어봅니다.",
            "graduation": "졸업은 성장의 증거입니다. 교육을 향한 열정이 담긴 기록입니다.",
            "holiday": "명절은 가족이 함께하는 소중한 시간입니다.",
            "immigration": "이민은 용기 있는 선택이었습니다. 새로운 땅에서의 이야기입니다."
        }
        return theme_intros.get(theme, f"{theme}에 관한 이야기입니다.")
    
    def _extract_quotes(self, transcription: str, max_quotes: int = 3) -> List[Dict[str, Any]]:
        """인용구 추출"""
        quotes = []
        
        # 간단한 문장 분리
        sentences = re.split(r'[.!?。]', transcription)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # 감정적인 문장 우선 선택
        emotional_keywords = ["기억", "그리워", "힘들", "행복", "슬프", "좋았", "사랑", "고마", "미안"]
        
        scored = []
        for sent in sentences:
            score = sum(1 for kw in emotional_keywords if kw in sent)
            scored.append((sent, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        for sent, score in scored[:max_quotes]:
            quotes.append({
                "text": sent[:200],
                "emotional_score": score
            })
        
        return quotes


class StoryExporter:
    """스토리 내보내기"""
    
    @staticmethod
    def to_html(story: Story) -> str:
        """HTML 내보내기"""
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{story.title}</title>
    <style>
        body {{ font-family: 'Noto Sans KR', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .subtitle {{ color: #7f8c8d; font-size: 1.2em; }}
        .segment {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
        .segment-title {{ color: #2980b9; margin-bottom: 10px; }}
        .segment-content {{ line-height: 1.8; }}
        .quote {{ border-left: 4px solid #3498db; padding-left: 15px; margin: 15px 0; font-style: italic; color: #555; }}
        .images {{ display: flex; gap: 10px; margin-top: 15px; }}
        .images img {{ max-width: 200px; border-radius: 5px; }}
        .meta {{ color: #95a5a6; font-size: 0.9em; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>{story.title}</h1>
    <p class="subtitle">{story.subtitle or ''}</p>
    <p>{story.description or ''}</p>
"""
        
        for segment in story.segments:
            html += f"""
    <div class="segment">
        <h2 class="segment-title">{segment.title}</h2>
        <div class="segment-content">{segment.content}</div>
"""
            
            # 인용구
            for quote in segment.quotes:
                html += f'        <div class="quote">"{quote.get("text", "")}"</div>\n'
            
            # 이미지
            if segment.image_urls:
                html += '        <div class="images">\n'
                for url in segment.image_urls[:3]:
                    if url:
                        html += f'            <img src="{url}" alt="관련 이미지">\n'
                html += '        </div>\n'
            
            # 메타
            meta_parts = []
            if segment.year:
                meta_parts.append(f"{segment.year}년")
            if segment.location:
                meta_parts.append(segment.location)
            if meta_parts:
                html += f'        <div class="meta">{" · ".join(meta_parts)}</div>\n'
            
            html += '    </div>\n'
        
        html += """
</body>
</html>
"""
        return html
    
    @staticmethod
    def to_markdown(story: Story) -> str:
        """Markdown 내보내기"""
        md = f"# {story.title}\n\n"
        
        if story.subtitle:
            md += f"*{story.subtitle}*\n\n"
        
        if story.description:
            md += f"{story.description}\n\n"
        
        md += "---\n\n"
        
        for segment in story.segments:
            md += f"## {segment.title}\n\n"
            md += f"{segment.content}\n\n"
            
            for quote in segment.quotes:
                md += f"> {quote.get('text', '')}\n\n"
            
            if segment.year or segment.location:
                meta = []
                if segment.year:
                    meta.append(f"📅 {segment.year}년")
                if segment.location:
                    meta.append(f"📍 {segment.location}")
                md += f"_{' · '.join(meta)}_\n\n"
            
            md += "---\n\n"
        
        return md


# ============ API 엔드포인트 확장 ============

def add_story_routes(app, story_generator: StoryGenerator, storage):
    """스토리 API 라우트 추가"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/stories", tags=["스토리"])
    
    @router.post("/generate/family")
    async def generate_family_story(
        family_name: str = "가족",
        item_ids: Optional[List[str]] = None
    ):
        """가족사 스토리 생성"""
        # 자료 로드
        items = []
        persons = []
        
        if storage and item_ids:
            for item_id in item_ids:
                metadata = storage.get(item_id)
                if metadata:
                    items.append({
                        "item_id": metadata.item_id,
                        "title": metadata.dublin_core.title,
                        "date": metadata.dublin_core.date,
                        "material_type": str(metadata.material_type),
                        "description": metadata.dublin_core.description,
                        "thumbnail_url": metadata.thumbnail_path
                    })
                    persons.extend([
                        {
                            "name_korean": p.name_korean,
                            "generation": int(p.generation) if p.generation else 0,
                            "migration_year": p.migration_year,
                            "birth_place": p.birth_place
                        }
                        for p in metadata.diaspora.persons
                    ])
        
        story = story_generator.generate_family_history(items, persons, family_name)
        return story.to_dict()
    
    @router.post("/generate/migration")
    async def generate_migration_story(person_id: str):
        """이주 여정 스토리 생성"""
        # 실제 구현에서는 person_id로 인물 정보 로드
        person = {"name_korean": "이주자", "migration_year": 1970}
        story = story_generator.generate_migration_story(person, [])
        return story.to_dict()
    
    @router.post("/generate/timeline")
    async def generate_timeline_story(
        item_ids: List[str],
        title: str = "시간의 기록"
    ):
        """타임라인 스토리 생성"""
        items = []
        if storage:
            for item_id in item_ids:
                metadata = storage.get(item_id)
                if metadata:
                    items.append({
                        "item_id": metadata.item_id,
                        "title": metadata.dublin_core.title,
                        "date": metadata.dublin_core.date,
                        "description": metadata.dublin_core.description
                    })
        
        story = story_generator.generate_timeline_story(items, title)
        return story.to_dict()
    
    @router.post("/generate/interactive")
    async def generate_interactive_story(
        item_ids: Optional[List[str]] = None,
        title: str = "선택의 여정"
    ):
        """인터랙티브 스토리 생성"""
        items = []
        persons = []
        story = story_generator.generate_interactive_story(items, persons, title)
        return story.to_dict()
    
    @router.get("/{story_id}")
    async def get_story(story_id: str):
        """스토리 조회"""
        # 실제 구현에서는 DB에서 로드
        return {"error": "스토리를 찾을 수 없습니다"}
    
    @router.get("/{story_id}/export/html")
    async def export_story_html(story_id: str):
        """HTML 내보내기"""
        # 실제 구현에서는 DB에서 로드 후 변환
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content="<html><body>스토리</body></html>")
    
    @router.get("/{story_id}/export/markdown")
    async def export_story_markdown(story_id: str):
        """Markdown 내보내기"""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content="# 스토리")
    
    app.include_router(router)


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="인터랙티브 스토리 생성기")
    parser.add_argument("--type", choices=["family", "migration", "timeline", "interactive", "thematic"],
                       default="family", help="스토리 유형")
    parser.add_argument("--output", "-o", default="story.json", help="출력 파일")
    parser.add_argument("--format", choices=["json", "html", "markdown"], default="json", help="출력 형식")
    args = parser.parse_args()
    
    # 생성기 초기화
    generator = StoryGenerator()
    generator.initialize()
    
    # 테스트 데이터
    test_items = [
        {"item_id": "1", "title": "1975년 가족 사진", "date": "1975-06-15", "material_type": "photograph"},
        {"item_id": "2", "title": "1980년 졸업식", "date": "1980-02-20", "material_type": "photograph"},
        {"item_id": "3", "title": "구술 인터뷰", "date": "2020-05-15", "material_type": "oral_history",
         "transcription": "제가 한국을 떠난 건 1972년이었어요. 그때 정말 힘들었지만, 가족을 위해 결심했습니다."}
    ]
    
    test_persons = [
        {"name_korean": "김철수", "generation": 1, "migration_year": 1972, "birth_place": "전남 목포"},
        {"name_korean": "김영희", "generation": 1, "migration_year": 1972}
    ]
    
    # 스토리 생성
    if args.type == "family":
        story = generator.generate_family_history(test_items, test_persons, "김씨 가족")
    elif args.type == "migration":
        story = generator.generate_migration_story(test_persons[0], test_items)
    elif args.type == "timeline":
        story = generator.generate_timeline_story(test_items)
    elif args.type == "interactive":
        story = generator.generate_interactive_story(test_items, test_persons)
    else:
        story = generator.generate_thematic_story(test_items, "family")
    
    # 출력
    if args.format == "json":
        output = story.to_json()
    elif args.format == "html":
        output = StoryExporter.to_html(story)
    else:
        output = StoryExporter.to_markdown(story)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"스토리 생성 완료: {args.output}")
    print(f"  - 유형: {story.story_type.value}")
    print(f"  - 세그먼트 수: {len(story.segments)}")
