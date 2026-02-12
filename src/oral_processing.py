"""
구술 기록 처리 모듈 (Oral History Processing Module)

한인 디아스포라 구술 인터뷰의 분석 및 스토리텔링을 위한 AI 기반 모듈

주요 기능:
- 구술 전사 (Transcription): Whisper 기반
- 화자 분리 및 식별 (Speaker Identification)
- 주제 구간 분할 (Topic Segmentation): BERTopic
- 핵심 인용구 추출 (Quote Extraction)
- 감정/톤 분석 (Sentiment Analysis)
- 자동 요약 (Summarization): GPT-4
- 타임라인 생성 (Timeline Generation)
- 내러티브 구조 분석 (Narrative Structure)

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
from collections import Counter

import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OralHistoryTopic(Enum):
    """구술 기록 주제 분류"""
    PRE_MIGRATION = "pre_migration"
    MIGRATION_JOURNEY = "migration_journey"
    EARLY_SETTLEMENT = "early_settlement"
    FAMILY_LIFE = "family_life"
    WORK_CAREER = "work_career"
    COMMUNITY = "community"
    IDENTITY = "identity"
    DISCRIMINATION = "discrimination"
    HOMELAND = "homeland"
    NEXT_GENERATION = "next_generation"
    REFLECTION = "reflection"
    OTHER = "other"


class SentimentType(Enum):
    """감정 유형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    NOSTALGIC = "nostalgic"
    PROUD = "proud"
    SAD = "sad"
    HOPEFUL = "hopeful"
    FRUSTRATED = "frustrated"


@dataclass
class Speaker:
    """화자 정보"""
    id: str
    label: str
    role: str = "unknown"
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    generation: Optional[int] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    migration_year: Optional[int] = None
    total_speaking_time: float = 0.0
    segment_count: int = 0


@dataclass
class TopicSegment:
    """주제 구간"""
    segment_id: int
    topic: OralHistoryTopic
    topic_keywords: List[str]
    start_time: float
    end_time: float
    text: str
    summary: str = ""
    confidence: float = 0.0
    speakers: List[str] = field(default_factory=list)
    sentiment: SentimentType = SentimentType.NEUTRAL


@dataclass
class Quote:
    """핵심 인용구"""
    quote_id: int
    text: str
    speaker: str
    start_time: float
    end_time: float
    topic: OralHistoryTopic
    sentiment: SentimentType
    significance_score: float
    context: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class TimelineEvent:
    """타임라인 이벤트"""
    event_id: int
    year: Optional[int]
    date_text: str
    description: str
    event_type: str
    source_text: str
    start_time: float
    confidence: float = 0.0


@dataclass
class NarrativeElement:
    """내러티브 요소"""
    element_type: str
    start_time: float
    end_time: float
    summary: str
    key_quotes: List[Quote]
    topics: List[OralHistoryTopic]


@dataclass
class OralHistoryAnalysis:
    """구술 기록 분석 결과"""
    audio_path: str
    duration: float
    full_transcript: str
    word_count: int
    speakers: List[Speaker]
    primary_interviewee: Optional[Speaker]
    topic_segments: List[TopicSegment]
    topic_distribution: Dict[str, float]
    key_quotes: List[Quote]
    overall_sentiment: SentimentType
    sentiment_timeline: List[Dict[str, Any]]
    timeline_events: List[TimelineEvent]
    narrative_elements: List[NarrativeElement]
    keywords: List[Tuple[str, float]]
    named_entities: Dict[str, List[str]]
    processing_time: float
    confidence: float


class TopicModeler:
    """주제 모델링"""
    
    TOPIC_KEYWORDS = {
        OralHistoryTopic.PRE_MIGRATION: [
            "고향", "어린시절", "학교", "가족", "부모님", "형제", "마을", "한국", "서울", "부산",
            "hometown", "childhood", "school", "family", "parents", "village", "Korea"
        ],
        OralHistoryTopic.MIGRATION_JOURNEY: [
            "이민", "출국", "비행기", "배", "공항", "비자", "여권", "떠나다", "도착", "출발",
            "immigration", "departure", "airplane", "airport", "visa", "passport", "leave"
        ],
        OralHistoryTopic.EARLY_SETTLEMENT: [
            "처음", "도착", "어려움", "영어", "적응", "집", "아파트", "동네", "낯선", "힘들",
            "first", "arrival", "difficulty", "English", "adaptation", "house", "strange"
        ],
        OralHistoryTopic.FAMILY_LIFE: [
            "결혼", "자녀", "아이들", "가정", "남편", "아내", "교육", "양육", "자식", "손자",
            "marriage", "children", "kids", "home", "husband", "wife", "education", "raising"
        ],
        OralHistoryTopic.WORK_CAREER: [
            "일", "직장", "사업", "가게", "회사", "돈", "경제", "직업", "장사", "식당",
            "work", "job", "business", "store", "company", "money", "career", "restaurant"
        ],
        OralHistoryTopic.COMMUNITY: [
            "한인회", "교회", "성당", "한글학교", "모임", "친구", "동포", "커뮤니티", "행사",
            "Korean association", "church", "Korean school", "gathering", "community"
        ],
        OralHistoryTopic.IDENTITY: [
            "한국인", "미국인", "정체성", "뿌리", "문화", "전통", "언어", "한글", "자부심",
            "Korean", "American", "identity", "roots", "culture", "tradition", "language", "pride"
        ],
        OralHistoryTopic.DISCRIMINATION: [
            "차별", "인종", "편견", "무시", "힘들", "서러움", "외국인", "다름", "소외",
            "discrimination", "racism", "prejudice", "ignored", "hardship", "foreigner"
        ],
        OralHistoryTopic.HOMELAND: [
            "고국", "귀국", "방문", "그리움", "향수", "부모님", "형제", "친척", "명절", "제사",
            "homeland", "return", "visit", "longing", "nostalgia", "relatives", "holiday"
        ],
        OralHistoryTopic.NEXT_GENERATION: [
            "자녀", "2세", "3세", "손자", "미래", "교육", "한글", "전통", "물려주다",
            "children", "second generation", "grandchildren", "future", "pass down"
        ],
        OralHistoryTopic.REFLECTION: [
            "돌아보면", "생각하면", "후회", "보람", "감사", "인생", "삶", "행복", "성공",
            "looking back", "thinking", "regret", "rewarding", "grateful", "life", "success"
        ]
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        logger.info(f"TopicModeler initialized on {device}")
    
    def load_model(self):
        """BERTopic 모델 로드"""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.model = BERTopic(embedding_model=embedding_model, language="multilingual")
            logger.info("BERTopic model loaded")
        except ImportError:
            logger.warning("BERTopic not available, using keyword-based approach")
    
    def segment_by_topic(
        self,
        transcript_segments: List[Dict[str, Any]],
        min_segment_duration: float = 30.0
    ) -> List[TopicSegment]:
        """주제별 구간 분할"""
        if not transcript_segments:
            return []
        
        topic_segments = []
        current_topic = None
        current_start = 0.0
        current_texts = []
        current_speakers = []
        segment_id = 0
        
        for seg in transcript_segments:
            text = seg.get("text", "")
            start_time = seg.get("start_time", 0)
            end_time = seg.get("end_time", 0)
            speaker = seg.get("speaker", "")
            
            # 주제 감지
            detected_topic = self._detect_topic(text)
            
            # 주제 변경 감지
            if current_topic is None:
                current_topic = detected_topic
                current_start = start_time
            elif detected_topic != current_topic and (start_time - current_start) >= min_segment_duration:
                # 이전 구간 저장
                topic_segment = TopicSegment(
                    segment_id=segment_id,
                    topic=current_topic,
                    topic_keywords=self._get_keywords_for_topic(current_topic),
                    start_time=current_start,
                    end_time=start_time,
                    text=" ".join(current_texts),
                    speakers=list(set(current_speakers)),
                    confidence=0.8
                )
                topic_segments.append(topic_segment)
                
                # 새 구간 시작
                segment_id += 1
                current_topic = detected_topic
                current_start = start_time
                current_texts = []
                current_speakers = []
            
            current_texts.append(text)
            if speaker:
                current_speakers.append(speaker)
        
        # 마지막 구간 저장
        if current_texts:
            last_end = transcript_segments[-1].get("end_time", 0) if transcript_segments else 0
            topic_segment = TopicSegment(
                segment_id=segment_id,
                topic=current_topic or OralHistoryTopic.OTHER,
                topic_keywords=self._get_keywords_for_topic(current_topic or OralHistoryTopic.OTHER),
                start_time=current_start,
                end_time=last_end,
                text=" ".join(current_texts),
                speakers=list(set(current_speakers)),
                confidence=0.8
            )
            topic_segments.append(topic_segment)
        
        return topic_segments
    
    def _detect_topic(self, text: str) -> OralHistoryTopic:
        """텍스트에서 주제 감지"""
        text_lower = text.lower()
        
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            topic_scores[topic] = score
        
        if max(topic_scores.values()) == 0:
            return OralHistoryTopic.OTHER
        
        return max(topic_scores, key=topic_scores.get)
    
    def _get_keywords_for_topic(self, topic: OralHistoryTopic) -> List[str]:
        """주제에 해당하는 키워드 반환"""
        return self.TOPIC_KEYWORDS.get(topic, [])[:5]
    
    def get_topic_distribution(self, topic_segments: List[TopicSegment]) -> Dict[str, float]:
        """주제 분포 계산"""
        total_duration = sum(seg.end_time - seg.start_time for seg in topic_segments)
        
        if total_duration == 0:
            return {}
        
        distribution = {}
        for seg in topic_segments:
            topic_name = seg.topic.value
            duration = seg.end_time - seg.start_time
            distribution[topic_name] = distribution.get(topic_name, 0) + duration
        
        # 비율로 변환
        for topic in distribution:
            distribution[topic] /= total_duration
        
        return distribution


class QuoteExtractor:
    """핵심 인용구 추출"""
    
    # 인용구 가치 지표 키워드
    SIGNIFICANCE_KEYWORDS = {
        "high": [
            "처음으로", "절대 잊지 못할", "가장 힘들", "가장 행복", "인생에서", "평생",
            "그때 깨달았", "그 순간", "전환점", "결정적", "중요한",
            "never forget", "first time", "most difficult", "happiest", "lifetime",
            "realized", "turning point", "decisive", "important"
        ],
        "emotional": [
            "눈물", "울었", "감사", "후회", "그리움", "자랑스러", "서러", "행복",
            "tears", "cried", "grateful", "regret", "longing", "proud", "sad", "happy"
        ],
        "narrative": [
            "그래서", "결국", "하지만", "그런데", "왜냐하면", "덕분에",
            "so", "finally", "but", "however", "because", "thanks to"
        ]
    }
    
    def __init__(self):
        logger.info("QuoteExtractor initialized")
    
    def extract_quotes(
        self,
        topic_segments: List[TopicSegment],
        max_quotes: int = 20,
        min_length: int = 20,
        max_length: int = 200
    ) -> List[Quote]:
        """핵심 인용구 추출"""
        all_quotes = []
        quote_id = 0
        
        for segment in topic_segments:
            # 문장 분리
            sentences = self._split_sentences(segment.text)
            
            for sentence in sentences:
                if len(sentence) < min_length or len(sentence) > max_length:
                    continue
                
                # 중요도 점수 계산
                significance = self._calculate_significance(sentence)
                
                if significance > 0.3:  # 임계값
                    # 감정 분석
                    sentiment = self._analyze_sentiment(sentence)
                    
                    quote = Quote(
                        quote_id=quote_id,
                        text=sentence.strip(),
                        speaker=segment.speakers[0] if segment.speakers else "Unknown",
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        topic=segment.topic,
                        sentiment=sentiment,
                        significance_score=significance,
                        keywords=self._extract_keywords(sentence)
                    )
                    all_quotes.append(quote)
                    quote_id += 1
        
        # 중요도 순 정렬 및 상위 선택
        all_quotes.sort(key=lambda q: q.significance_score, reverse=True)
        return all_quotes[:max_quotes]
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 한국어와 영어 문장 구분자
        sentences = re.split(r'[.!?。]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_significance(self, sentence: str) -> float:
        """인용구 중요도 계산"""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # 고중요도 키워드
        for keyword in self.SIGNIFICANCE_KEYWORDS["high"]:
            if keyword.lower() in sentence_lower:
                score += 0.3
        
        # 감정적 키워드
        for keyword in self.SIGNIFICANCE_KEYWORDS["emotional"]:
            if keyword.lower() in sentence_lower:
                score += 0.2
        
        # 내러티브 키워드
        for keyword in self.SIGNIFICANCE_KEYWORDS["narrative"]:
            if keyword.lower() in sentence_lower:
                score += 0.1
        
        # 길이 보정 (너무 짧거나 긴 문장 페널티)
        length = len(sentence)
        if 50 <= length <= 150:
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_sentiment(self, text: str) -> SentimentType:
        """감정 분석"""
        text_lower = text.lower()
        
        positive_words = ["행복", "감사", "자랑", "성공", "기쁨", "좋", "happy", "grateful", "proud", "success"]
        negative_words = ["힘들", "어려", "슬프", "후회", "서러", "차별", "difficult", "sad", "regret", "discrimination"]
        nostalgic_words = ["그리움", "향수", "추억", "옛날", "그때", "longing", "nostalgia", "memory", "those days"]
        
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        nostalgic_count = sum(1 for w in nostalgic_words if w in text_lower)
        
        if nostalgic_count > 0:
            return SentimentType.NOSTALGIC
        elif positive_count > negative_count:
            return SentimentType.POSITIVE
        elif negative_count > positive_count:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 간단한 명사 추출 (한국어/영어)
        korean_nouns = re.findall(r'[가-힣]{2,}', text)
        english_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        
        # 빈도수 기반 상위 5개
        all_words = korean_nouns + english_nouns
        word_freq = Counter(all_words)
        
        return [word for word, _ in word_freq.most_common(5)]


class SentimentAnalyzer:
    """감정/톤 분석"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        logger.info(f"SentimentAnalyzer initialized on {device}")
    
    def load_model(self):
        """감정 분석 모델 로드"""
        try:
            from transformers import pipeline
            self.model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Sentiment model loaded")
        except ImportError:
            logger.warning("Transformers not available for sentiment analysis")
    
    def analyze_segments(
        self,
        topic_segments: List[TopicSegment]
    ) -> Tuple[SentimentType, List[Dict[str, Any]]]:
        """구간별 감정 분석"""
        sentiment_timeline = []
        sentiment_scores = []
        
        for segment in topic_segments:
            sentiment, score = self._analyze_text(segment.text)
            
            sentiment_timeline.append({
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "sentiment": sentiment.value,
                "score": score,
                "topic": segment.topic.value
            })
            
            segment.sentiment = sentiment
            sentiment_scores.append((sentiment, score))
        
        # 전체 감정 결정
        overall = self._determine_overall_sentiment(sentiment_scores)
        
        return overall, sentiment_timeline
    
    def _analyze_text(self, text: str) -> Tuple[SentimentType, float]:
        """텍스트 감정 분석"""
        if self.model is not None:
            try:
                # 텍스트가 너무 길면 잘라서 분석
                text_chunk = text[:512]
                result = self.model(text_chunk)[0]
                
                # 5점 척도를 감정으로 매핑
                label = result["label"]
                score = result["score"]
                
                if "5" in label or "4" in label:
                    return SentimentType.POSITIVE, score
                elif "1" in label or "2" in label:
                    return SentimentType.NEGATIVE, score
                else:
                    return SentimentType.NEUTRAL, score
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
        
        # 폴백: 규칙 기반
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """규칙 기반 감정 분석"""
        text_lower = text.lower()
        
        positive = ["행복", "감사", "좋", "기쁨", "happy", "grateful", "good", "joy"]
        negative = ["슬프", "힘들", "어려", "아프", "sad", "difficult", "hard", "painful"]
        nostalgic = ["그리움", "추억", "향수", "옛날", "longing", "memory", "old days"]
        
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        nos_count = sum(1 for w in nostalgic if w in text_lower)
        
        if nos_count > 0:
            return SentimentType.NOSTALGIC, 0.7
        elif pos_count > neg_count:
            return SentimentType.POSITIVE, 0.6
        elif neg_count > pos_count:
            return SentimentType.NEGATIVE, 0.6
        else:
            return SentimentType.NEUTRAL, 0.5
    
    def _determine_overall_sentiment(
        self,
        sentiments: List[Tuple[SentimentType, float]]
    ) -> SentimentType:
        """전체 감정 결정"""
        if not sentiments:
            return SentimentType.NEUTRAL
        
        # 가중 평균
        sentiment_weights = {}
        for sentiment, score in sentiments:
            sentiment_weights[sentiment] = sentiment_weights.get(sentiment, 0) + score
        
        return max(sentiment_weights, key=sentiment_weights.get)


class TimelineExtractor:
    """타임라인 추출"""
    
    # 연도 패턴
    YEAR_PATTERNS = [
        r'(\d{4})년',
        r'(\d{4})년대',
        r'(\d{2})년도',
        r'in (\d{4})',
        r'(\d{4})s',
    ]
    
    # 이벤트 키워드
    EVENT_KEYWORDS = {
        "migration": ["이민", "이주", "떠나", "도착", "immigration", "emigration", "arrived"],
        "family": ["결혼", "출산", "태어나", "돌아가시", "marriage", "birth", "born", "passed"],
        "work": ["취직", "창업", "사업", "은퇴", "job", "business", "started", "retired"],
        "education": ["졸업", "입학", "학교", "graduated", "enrolled", "school"],
        "community": ["설립", "참여", "회장", "founded", "joined", "president"],
    }
    
    def __init__(self):
        logger.info("TimelineExtractor initialized")
    
    def extract_timeline(
        self,
        topic_segments: List[TopicSegment]
    ) -> List[TimelineEvent]:
        """타임라인 이벤트 추출"""
        events = []
        event_id = 0
        
        for segment in topic_segments:
            # 연도 추출
            years = self._extract_years(segment.text)
            
            for year, date_text in years:
                # 이벤트 유형 감지
                event_type = self._detect_event_type(segment.text)
                
                # 이벤트 설명 추출
                description = self._extract_event_description(segment.text, date_text)
                
                if description:
                    event = TimelineEvent(
                        event_id=event_id,
                        year=year,
                        date_text=date_text,
                        description=description,
                        event_type=event_type,
                        source_text=segment.text[:200],
                        start_time=segment.start_time,
                        confidence=0.7
                    )
                    events.append(event)
                    event_id += 1
        
        # 연도순 정렬
        events.sort(key=lambda e: e.year if e.year else 0)
        
        # 중복 제거
        return self._deduplicate_events(events)
    
    def _extract_years(self, text: str) -> List[Tuple[int, str]]:
        """연도 추출"""
        years = []
        
        for pattern in self.YEAR_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                year_str = match.group(1)
                
                # 2자리 연도 처리
                if len(year_str) == 2:
                    year = int(year_str)
                    if year > 50:
                        year += 1900
                    else:
                        year += 2000
                else:
                    year = int(year_str)
                
                # 유효 범위 체크
                if 1900 <= year <= 2030:
                    years.append((year, match.group(0)))
        
        return years
    
    def _detect_event_type(self, text: str) -> str:
        """이벤트 유형 감지"""
        text_lower = text.lower()
        
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return event_type
        
        return "other"
    
    def _extract_event_description(self, text: str, date_text: str) -> str:
        """이벤트 설명 추출"""
        # 날짜 주변 문맥 추출
        idx = text.find(date_text)
        if idx == -1:
            return ""
        
        # 앞뒤 100자
        start = max(0, idx - 50)
        end = min(len(text), idx + len(date_text) + 100)
        
        context = text[start:end].strip()
        
        # 문장 정리
        sentences = re.split(r'[.!?。]', context)
        for sentence in sentences:
            if date_text in sentence:
                return sentence.strip()
        
        return context[:100]
    
    def _deduplicate_events(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """중복 이벤트 제거"""
        seen = set()
        unique_events = []
        
        for event in events:
            key = (event.year, event.event_type)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events


class NarrativeAnalyzer:
    """내러티브 구조 분석"""
    
    def __init__(self):
        logger.info("NarrativeAnalyzer initialized")
    
    def analyze_structure(
        self,
        topic_segments: List[TopicSegment],
        quotes: List[Quote]
    ) -> List[NarrativeElement]:
        """내러티브 구조 분석"""
        if not topic_segments:
            return []
        
        total_duration = topic_segments[-1].end_time - topic_segments[0].start_time
        elements = []
        
        # 5단계 내러티브 구조로 분할
        stage_duration = total_duration / 5
        
        stages = [
            ("exposition", "배경과 인물 소개"),
            ("rising_action", "갈등과 도전의 시작"),
            ("climax", "핵심 전환점"),
            ("falling_action", "결과와 적응"),
            ("resolution", "성찰과 결론")
        ]
        
        for i, (stage_type, _) in enumerate(stages):
            start_time = topic_segments[0].start_time + i * stage_duration
            end_time = start_time + stage_duration
            
            # 해당 구간의 세그먼트
            stage_segments = [
                seg for seg in topic_segments
                if seg.start_time >= start_time and seg.end_time <= end_time
            ]
            
            # 해당 구간의 인용구
            stage_quotes = [
                q for q in quotes
                if q.start_time >= start_time and q.end_time <= end_time
            ]
            
            # 주제 추출
            topics = list(set(seg.topic for seg in stage_segments))
            
            # 요약 생성
            summary = self._generate_stage_summary(stage_type, stage_segments)
            
            element = NarrativeElement(
                element_type=stage_type,
                start_time=start_time,
                end_time=end_time,
                summary=summary,
                key_quotes=stage_quotes[:3],
                topics=topics
            )
            elements.append(element)
        
        return elements
    
    def _generate_stage_summary(
        self,
        stage_type: str,
        segments: List[TopicSegment]
    ) -> str:
        """단계별 요약 생성"""
        if not segments:
            return ""
        
        # 주요 주제
        topics = [seg.topic.value for seg in segments]
        topic_counts = Counter(topics)
        main_topics = [t for t, _ in topic_counts.most_common(2)]
        
        # 기본 요약 템플릿
        templates = {
            "exposition": f"이 구간에서는 {', '.join(main_topics)} 등의 주제로 배경을 설명합니다.",
            "rising_action": f"이 구간에서는 {', '.join(main_topics)} 등의 도전과 어려움을 다룹니다.",
            "climax": f"이 구간은 {', '.join(main_topics)} 등 핵심적인 전환점을 담고 있습니다.",
            "falling_action": f"이 구간에서는 {', '.join(main_topics)} 등의 결과와 적응 과정을 설명합니다.",
            "resolution": f"이 구간에서는 {', '.join(main_topics)} 등에 대한 성찰과 결론을 다룹니다."
        }
        
        return templates.get(stage_type, f"주요 주제: {', '.join(main_topics)}")


class Summarizer:
    """자동 요약"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        logger.info(f"Summarizer initialized on {device}")
    
    def load_model(self):
        """요약 모델 로드"""
        try:
            from transformers import pipeline
            self.model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Summarization model loaded")
        except ImportError:
            logger.warning("Transformers not available for summarization")
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """텍스트 요약"""
        if not text:
            return ""
        
        if self.model is not None:
            try:
                # 텍스트 길이 제한
                text_chunk = text[:1024]
                result = self.model(text_chunk, max_length=max_length, min_length=30)
                return result[0]["summary_text"]
            except Exception as e:
                logger.error(f"Summarization error: {e}")
        
        # 폴백: 첫 3문장
        sentences = re.split(r'[.!?。]', text)
        return ". ".join(sentences[:3]) + "."
    
    def summarize_segments(
        self,
        topic_segments: List[TopicSegment]
    ) -> Dict[str, str]:
        """주제별 요약"""
        summaries = {}
        
        # 주제별 텍스트 그룹화
        topic_texts = {}
        for segment in topic_segments:
            topic = segment.topic.value
            if topic not in topic_texts:
                topic_texts[topic] = []
            topic_texts[topic].append(segment.text)
        
        # 주제별 요약
        for topic, texts in topic_texts.items():
            combined = " ".join(texts)
            summaries[topic] = self.summarize(combined)
        
        return summaries


class NamedEntityExtractor:
    """개체명 추출"""
    
    def __init__(self):
        logger.info("NamedEntityExtractor initialized")
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """개체명 추출"""
        entities = {
            "persons": [],
            "locations": [],
            "organizations": [],
            "dates": [],
            "events": []
        }
        
        # 인명 (한국어 이름 패턴)
        korean_names = re.findall(r'[가-힣]{2,4}(?=\s*(씨|님|선생|할머니|할아버지|아버지|어머니))', text)
        entities["persons"].extend(korean_names)
        
        # 지명
        location_keywords = [
            "서울", "부산", "대구", "인천", "광주", "대전", "목포", "전주",
            "Los Angeles", "LA", "New York", "Chicago", "Hawaii", "San Francisco"
        ]
        for loc in location_keywords:
            if loc in text:
                entities["locations"].append(loc)
        
        # 기관명
        org_patterns = [
            r'[가-힣]+한인회',
            r'[가-힣]+교회',
            r'[가-힣]+학교',
            r'[A-Za-z\s]+Church',
            r'[A-Za-z\s]+Association'
        ]
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities["organizations"].extend(matches)
        
        # 날짜
        date_patterns = [
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',
            r'\d{4}년\s*\d{1,2}월',
            r'\d{4}년'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities["dates"].extend(matches)
        
        # 중복 제거
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


class OralHistoryProcessor:
    """통합 구술 기록 처리 파이프라인"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        
        # 음성 처리 모듈 (외부 의존)
        self.audio_processor = None
        
        # 분석 모듈
        self.topic_modeler = TopicModeler(self.device)
        self.quote_extractor = QuoteExtractor()
        self.sentiment_analyzer = SentimentAnalyzer(self.device)
        self.timeline_extractor = TimelineExtractor()
        self.narrative_analyzer = NarrativeAnalyzer()
        self.summarizer = Summarizer(self.device)
        self.entity_extractor = NamedEntityExtractor()
        
        self.models_loaded = False
        logger.info(f"OralHistoryProcessor initialized on {self.device}")
    
    def load_models(self):
        """모든 모델 로드"""
        try:
            # 음성 처리 모듈 로드
            from .transcription import AudioProcessor
            self.audio_processor = AudioProcessor({"device": self.device})
            self.audio_processor.load_models()
            
            # 분석 모듈 로드
            self.topic_modeler.load_model()
            self.sentiment_analyzer.load_model()
            self.summarizer.load_model()
            
            self.models_loaded = True
            logger.info("All oral history models loaded")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def process(
        self,
        audio_path: str,
        speaker_info: Optional[Dict] = None,
        language: str = "ko"
    ) -> OralHistoryAnalysis:
        """구술 기록 처리"""
        import time
        start_time = time.time()
        
        # 1. 음성 전사 및 화자 분리
        logger.info("Step 1: Transcribing audio...")
        if self.audio_processor:
            transcription_result = self.audio_processor.process(
                audio_path,
                language=language,
                diarize=True
            )
            full_transcript = transcription_result.full_text
            transcript_segments = [
                {
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "speaker": seg.speaker
                }
                for seg in transcription_result.segments
            ]
            duration = transcription_result.duration
            speakers = self._create_speakers(transcription_result, speaker_info)
        else:
            # 폴백: 텍스트 파일에서 로드
            full_transcript, transcript_segments, duration = self._load_transcript_fallback(audio_path)
            speakers = []
        
        # 2. 주제 분할
        logger.info("Step 2: Segmenting by topic...")
        topic_segments = self.topic_modeler.segment_by_topic(transcript_segments)
        topic_distribution = self.topic_modeler.get_topic_distribution(topic_segments)
        
        # 3. 핵심 인용구 추출
        logger.info("Step 3: Extracting key quotes...")
        key_quotes = self.quote_extractor.extract_quotes(topic_segments)
        
        # 4. 감정 분석
        logger.info("Step 4: Analyzing sentiment...")
        overall_sentiment, sentiment_timeline = self.sentiment_analyzer.analyze_segments(topic_segments)
        
        # 5. 타임라인 추출
        logger.info("Step 5: Extracting timeline...")
        timeline_events = self.timeline_extractor.extract_timeline(topic_segments)
        
        # 6. 내러티브 구조 분석
        logger.info("Step 6: Analyzing narrative structure...")
        narrative_elements = self.narrative_analyzer.analyze_structure(topic_segments, key_quotes)
        
        # 7. 키워드 및 개체명 추출
        logger.info("Step 7: Extracting keywords and entities...")
        keywords = self._extract_keywords(full_transcript)
        named_entities = self.entity_extractor.extract(full_transcript)
        
        # 8. 주제별 요약 생성
        logger.info("Step 8: Generating summaries...")
        for segment in topic_segments:
            segment.summary = self.summarizer.summarize(segment.text, max_length=100)
        
        processing_time = time.time() - start_time
        
        # 주 인터뷰이 찾기
        primary_interviewee = None
        if speakers:
            interviewees = [s for s in speakers if s.role == "interviewee"]
            if interviewees:
                primary_interviewee = max(interviewees, key=lambda s: s.total_speaking_time)
        
        return OralHistoryAnalysis(
            audio_path=audio_path,
            duration=duration,
            full_transcript=full_transcript,
            word_count=len(full_transcript.split()),
            speakers=speakers,
            primary_interviewee=primary_interviewee,
            topic_segments=topic_segments,
            topic_distribution=topic_distribution,
            key_quotes=key_quotes,
            overall_sentiment=overall_sentiment,
            sentiment_timeline=sentiment_timeline,
            timeline_events=timeline_events,
            narrative_elements=narrative_elements,
            keywords=keywords,
            named_entities=named_entities,
            processing_time=processing_time,
            confidence=0.85
        )
    
    def _create_speakers(
        self,
        transcription_result,
        speaker_info: Optional[Dict]
    ) -> List[Speaker]:
        """화자 정보 생성"""
        speakers = []
        speaker_info = speaker_info or {}
        
        for speaker_id in transcription_result.speakers:
            info = speaker_info.get(speaker_id, {})
            
            # 발화 시간 계산
            total_time = sum(
                seg.end_time - seg.start_time
                for seg in transcription_result.segments
                if seg.speaker == speaker_id
            )
            
            segment_count = sum(
                1 for seg in transcription_result.segments
                if seg.speaker == speaker_id
            )
            
            speaker = Speaker(
                id=speaker_id,
                label=speaker_id,
                role=info.get("role", "unknown"),
                name=info.get("name"),
                age=info.get("age"),
                gender=info.get("gender"),
                generation=info.get("generation"),
                origin=info.get("origin"),
                destination=info.get("destination"),
                migration_year=info.get("migration_year"),
                total_speaking_time=total_time,
                segment_count=segment_count
            )
            speakers.append(speaker)
        
        return speakers
    
    def _load_transcript_fallback(
        self,
        audio_path: str
    ) -> Tuple[str, List[Dict], float]:
        """텍스트 파일에서 전사 로드 (폴백)"""
        txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 간단한 세그먼트 분할
            sentences = re.split(r'[.!?。]\s*', text)
            segments = []
            time_per_sentence = 5.0  # 가정
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    segments.append({
                        "text": sentence.strip(),
                        "start_time": i * time_per_sentence,
                        "end_time": (i + 1) * time_per_sentence,
                        "speaker": "Speaker_1"
                    })
            
            duration = len(segments) * time_per_sentence
            return text, segments, duration
        
        return "", [], 0.0
    
    def _extract_keywords(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """키워드 추출"""
        # 불용어
        stopwords = {
            "이", "그", "저", "것", "수", "등", "때", "더", "또", "안",
            "the", "a", "an", "is", "are", "was", "were", "be", "have", "has"
        }
        
        # 토큰화
        korean = re.findall(r'[가-힣]{2,}', text)
        english = re.findall(r'[a-zA-Z]{3,}', text)
        
        tokens = [t.lower() for t in korean + english if t.lower() not in stopwords]
        
        # 빈도 계산
        freq = Counter(tokens)
        total = sum(freq.values())
        
        return [(word, count / total) for word, count in freq.most_common(top_n)]
    
    def save_analysis(self, analysis: OralHistoryAnalysis, output_dir: str):
        """분석 결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(analysis.audio_path))[0]
        
        # 전체 전사 저장
        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(analysis.full_transcript)
        
        # 주제 구간 저장
        topics_path = os.path.join(output_dir, f"{base_name}_topics.json")
        topics_data = [
            {
                "segment_id": seg.segment_id,
                "topic": seg.topic.value,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "summary": seg.summary,
                "sentiment": seg.sentiment.value
            }
            for seg in analysis.topic_segments
        ]
        with open(topics_path, 'w', encoding='utf-8') as f:
            json.dump(topics_data, f, ensure_ascii=False, indent=2)
        
        # 핵심 인용구 저장
        quotes_path = os.path.join(output_dir, f"{base_name}_quotes.json")
        quotes_data = [
            {
                "quote_id": q.quote_id,
                "text": q.text,
                "speaker": q.speaker,
                "topic": q.topic.value,
                "sentiment": q.sentiment.value,
                "significance": q.significance_score
            }
            for q in analysis.key_quotes
        ]
        with open(quotes_path, 'w', encoding='utf-8') as f:
            json.dump(quotes_data, f, ensure_ascii=False, indent=2)
        
        # 타임라인 저장
        timeline_path = os.path.join(output_dir, f"{base_name}_timeline.json")
        timeline_data = [
            {
                "event_id": e.event_id,
                "year": e.year,
                "description": e.description,
                "event_type": e.event_type
            }
            for e in analysis.timeline_events
        ]
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, ensure_ascii=False, indent=2)
        
        # 전체 분석 요약 저장
        summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
        summary_data = {
            "duration": analysis.duration,
            "word_count": analysis.word_count,
            "speaker_count": len(analysis.speakers),
            "topic_count": len(analysis.topic_segments),
            "quote_count": len(analysis.key_quotes),
            "event_count": len(analysis.timeline_events),
            "overall_sentiment": analysis.overall_sentiment.value,
            "topic_distribution": analysis.topic_distribution,
            "keywords": analysis.keywords[:10],
            "processing_time": analysis.processing_time
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis saved to {output_dir}")


# CLI 인터페이스
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diaspora Oral History Processing")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--language", default="ko", help="Language code")
    parser.add_argument("--speaker-info", help="Speaker info JSON file")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    processor = OralHistoryProcessor({"device": args.device})
    processor.load_models()
    
    # 화자 정보 로드
    speaker_info = None
    if args.speaker_info and os.path.exists(args.speaker_info):
        with open(args.speaker_info, 'r', encoding='utf-8') as f:
            speaker_info = json.load(f)
    
    analysis = processor.process(args.input, speaker_info, args.language)
    processor.save_analysis(analysis, args.output)
    
    print(f"\n=== Oral History Analysis Complete ===")
    print(f"Duration: {analysis.duration:.1f}s")
    print(f"Word count: {analysis.word_count}")
    print(f"Speakers: {len(analysis.speakers)}")
    print(f"Topic segments: {len(analysis.topic_segments)}")
    print(f"Key quotes: {len(analysis.key_quotes)}")
    print(f"Timeline events: {len(analysis.timeline_events)}")
    print(f"Overall sentiment: {analysis.overall_sentiment.value}")
    print(f"Processing time: {analysis.processing_time:.1f}s")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
