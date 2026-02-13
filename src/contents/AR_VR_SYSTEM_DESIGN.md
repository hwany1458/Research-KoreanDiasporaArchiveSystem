# AR/VR 시스템 설계서

## 한인 디아스포라 기록유산 디지털화 시스템
## 4단계: 실감형 콘텐츠 - AR/VR 모듈

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처 설계](#2-아키텍처-설계)
3. [AR 증강 뷰어 설계](#3-ar-증강-뷰어-설계)
4. [VR 가상 전시 설계](#4-vr-가상-전시-설계)
5. [API 연동 설계](#5-api-연동-설계)
6. [데이터 모델](#6-데이터-모델)
7. [UI/UX 설계](#7-uiux-설계)
8. [기술 스택](#8-기술-스택)

---

## 1. 시스템 개요

### 1.1 목적

한인 디아스포라 기록유산을 AR/VR 기술을 통해 실감형 콘텐츠로 제공하여, 사용자가 역사적 자료를 보다 몰입감 있게 체험할 수 있도록 합니다.

### 1.2 주요 기능

| 모듈 | 기능 | 대상 플랫폼 |
|------|------|------------|
| **AR 증강 뷰어** | 실물 사진 인식, 정보 오버레이, 인물 태그, 음성 설명 | iOS, Android |
| **VR 가상 전시** | 3D 전시 공간, 자료 탐색, 가이드 투어, 음성 내레이션 | Oculus Quest, PC VR |

### 1.3 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 디바이스                            │
├─────────────────────────────┬───────────────────────────────────┤
│      AR 모바일 앱           │         VR 헤드셋 앱              │
│  ┌─────────────────────┐   │   ┌─────────────────────────┐     │
│  │  AR Foundation      │   │   │  XR Interaction Toolkit │     │
│  │  Vuforia Engine     │   │   │  OpenXR                 │     │
│  │  이미지 인식        │   │   │  텔레포트/그랩          │     │
│  │  정보 오버레이      │   │   │  3D 전시 공간           │     │
│  └─────────────────────┘   │   └─────────────────────────┘     │
└─────────────────────────────┴───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Unity 공통 모듈                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ API Manager  │  │ Data Manager │  │ Audio Manager        │  │
│  │ REST 통신    │  │ 캐시/동기화  │  │ TTS/음성 재생        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                    백엔드 서버 (Python FastAPI)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 자료 API     │  │ 검색 API     │  │ 스토리 API           │  │
│  │ /items       │  │ /search      │  │ /stories             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 아키텍처 설계

### 2.1 Unity 프로젝트 구조

```
DiasporaAR_VR/
├── Assets/
│   ├── Scripts/
│   │   ├── Common/           # 공통 유틸리티
│   │   │   ├── Singleton.cs
│   │   │   ├── EventManager.cs
│   │   │   └── Constants.cs
│   │   ├── API/              # 서버 통신
│   │   │   ├── APIManager.cs
│   │   │   ├── APIModels.cs
│   │   │   └── APIEndpoints.cs
│   │   ├── Data/             # 데이터 관리
│   │   │   ├── DataManager.cs
│   │   │   ├── CacheManager.cs
│   │   │   └── ItemData.cs
│   │   ├── AR/               # AR 모듈
│   │   │   ├── ARSessionManager.cs
│   │   │   ├── ImageTracker.cs
│   │   │   ├── InfoOverlay.cs
│   │   │   ├── PersonTagManager.cs
│   │   │   └── ARPhotoViewer.cs
│   │   ├── VR/               # VR 모듈
│   │   │   ├── VRSessionManager.cs
│   │   │   ├── GalleryManager.cs
│   │   │   ├── ExhibitItem.cs
│   │   │   ├── TeleportManager.cs
│   │   │   ├── GuidedTourManager.cs
│   │   │   └── VRInteractionManager.cs
│   │   ├── Audio/            # 오디오
│   │   │   ├── AudioManager.cs
│   │   │   ├── TTSManager.cs
│   │   │   └── NarrationPlayer.cs
│   │   └── UI/               # UI 컴포넌트
│   │       ├── UIManager.cs
│   │       ├── ItemDetailPanel.cs
│   │       ├── TimelineView.cs
│   │       └── SearchPanel.cs
│   ├── Prefabs/
│   │   ├── AR/
│   │   │   ├── ARInfoPanel.prefab
│   │   │   ├── PersonTag.prefab
│   │   │   └── PhotoFrame.prefab
│   │   ├── VR/
│   │   │   ├── ExhibitFrame.prefab
│   │   │   ├── InfoStand.prefab
│   │   │   ├── TeleportPoint.prefab
│   │   │   └── GalleryRoom.prefab
│   │   └── UI/
│   │       ├── DetailPanel.prefab
│   │       └── LoadingIndicator.prefab
│   ├── Scenes/
│   │   ├── AR_Main.unity
│   │   ├── VR_Gallery.unity
│   │   └── VR_Tour.unity
│   ├── Resources/
│   │   ├── Materials/
│   │   ├── Textures/
│   │   └── Audio/
│   └── StreamingAssets/
│       └── ImageTargets/     # AR 인식 이미지
├── Packages/
│   └── manifest.json
└── ProjectSettings/
```

### 2.2 모듈 의존성

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ AR Module   │  │ VR Module   │  │ UI Module           │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ DataManager │  │AudioManager │  │ EventManager        │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   APIManager                         │    │
│  │  - GET /items, /persons, /locations                  │    │
│  │  - POST /search                                      │    │
│  │  - GET /stories                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. AR 증강 뷰어 설계

### 3.1 기능 목록

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| 이미지 인식 | 실물 사진/문서 카메라 인식 | 필수 |
| 정보 오버레이 | 인식된 자료 위에 메타데이터 표시 | 필수 |
| 인물 태그 | 사진 속 인물 위치에 태그 표시 | 필수 |
| 상세 정보 | 태그 터치 시 인물/자료 상세 | 필수 |
| 음성 설명 | TTS 또는 녹음된 설명 재생 | 선택 |
| 관련 자료 | 연결된 다른 자료 탐색 | 선택 |
| 타임라인 | 시간순 자료 네비게이션 | 선택 |

### 3.2 AR 워크플로우

```
[시작] → [카메라 활성화] → [이미지 스캔]
                              │
                    ┌─────────┴─────────┐
                    │ 이미지 인식 성공?  │
                    └─────────┬─────────┘
                         Yes  │  No
              ┌───────────────┴───────────────┐
              ▼                               ▼
    [서버에서 자료 정보 조회]          [스캔 계속]
              │
              ▼
    [정보 오버레이 표시]
              │
              ▼
    [인물 태그 배치]
              │
              ▼
    [사용자 인터랙션 대기]
              │
    ┌─────────┴─────────────────────┐
    │                               │
    ▼                               ▼
[태그 터치]                    [음성 버튼]
    │                               │
    ▼                               ▼
[상세 패널 표시]              [음성 설명 재생]
```

### 3.3 이미지 인식 방식

#### 방식 1: Vuforia Image Target (권장)
```
- 사전 등록된 이미지 타겟 인식
- 높은 인식 정확도
- 오프라인 동작 가능
- 이미지 타겟 DB 필요
```

#### 방식 2: 클라우드 이미지 인식
```
- 서버에서 이미지 매칭
- 동적 이미지 추가 가능
- 네트워크 필요
- 지연 시간 발생
```

### 3.4 AR 화면 레이아웃

```
┌─────────────────────────────────────────┐
│  ← 뒤로     AR 뷰어        🔍 검색      │ ← 상단바
├─────────────────────────────────────────┤
│                                         │
│         ┌─────────────────┐             │
│         │   [인식된 사진]  │             │
│         │                 │             │
│         │  👤 김철수      │             │ ← 인물 태그
│         │       👤 김영희  │             │
│         │                 │             │
│         └─────────────────┘             │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 📷 1975년 LA 가족 사진          │    │ ← 정보 오버레이
│  │ 📍 Los Angeles, CA              │    │
│  │ 📅 1975-06-15                   │    │
│  └─────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│  🔊 음성설명  │  📋 상세보기  │  🔗 관련자료 │ ← 하단 버튼
└─────────────────────────────────────────┘
```

---

## 4. VR 가상 전시 설계

### 4.1 기능 목록

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| 3D 갤러리 공간 | 가상 전시관 환경 | 필수 |
| 자료 전시 | 사진/문서/영상 액자 배치 | 필수 |
| 텔레포트 이동 | 포인터로 위치 이동 | 필수 |
| 자료 상호작용 | 그랩, 확대, 상세 보기 | 필수 |
| 음성 가이드 | 전시 설명 내레이션 | 선택 |
| 가이드 투어 | 자동 순회 경로 | 선택 |
| 타임라인 룸 | 연대별 전시 공간 | 선택 |

### 4.2 VR 갤러리 구조

```
                    [입구 로비]
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    [1970년대 룸]  [1980년대 룸]  [1990년대 룸]
          │              │              │
          │         [중앙 홀]          │
          │              │              │
          └──────────────┼──────────────┘
                         │
                    [특별 전시실]
                    (구술 기록)
```

### 4.3 전시 공간 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│                        천장 (조명)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌───┐     ┌───┐     ┌───┐     ┌───┐     ┌───┐          │
│    │ 1 │     │ 2 │     │ 3 │     │ 4 │     │ 5 │    벽면   │
│    └───┘     └───┘     └───┘     └───┘     └───┘    전시   │
│                                                             │
│         ⬡              ⬡              ⬡                    │
│      텔레포트        텔레포트        텔레포트               │
│       포인트          포인트          포인트                │
│                                                             │
│              ┌─────────────────┐                           │
│              │   중앙 안내대    │                           │
│              │   (인터랙티브)   │                           │
│              └─────────────────┘                           │
│                                                             │
│    ┌───┐     ┌───┐     ┌───┐     ┌───┐     ┌───┐          │
│    │ 6 │     │ 7 │     │ 8 │     │ 9 │     │10 │          │
│    └───┘     └───┘     └───┘     └───┘     └───┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      바닥 (텔레포트 가능 영역)               │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 VR 인터랙션 설계

| 인터랙션 | 입력 | 동작 |
|---------|------|------|
| 텔레포트 | 컨트롤러 포인터 + 트리거 | 해당 위치로 이동 |
| 자료 선택 | 레이캐스트 + 트리거 | 상세 정보 패널 |
| 자료 확대 | 그랩 + 당기기 | 자료 확대 보기 |
| 메뉴 열기 | 메뉴 버튼 | 옵션 메뉴 표시 |
| 음성 재생 | 자료 근처 접근 | 자동 재생 |

### 4.5 가이드 투어 경로

```
[시작: 로비]
     │
     ▼ "환영합니다. 이 전시는 김씨 가족의 이민 역사를 담고 있습니다."
[정거장 1: 1972년 이민 결심]
     │
     ▼ "1972년, 김철수 씨는 가족과 함께 새로운 삶을 찾아 떠났습니다."
[정거장 2: 1975년 정착]
     │
     ▼ "LA 코리아타운에서의 첫 가족 사진입니다."
[정거장 3: 1980년대 성장]
     │
     ▼ "자녀들은 미국에서 교육을 받으며 성장했습니다."
[정거장 4: 구술 기록실]
     │
     ▼ "김영희 할머니의 생생한 이야기를 들어보세요."
[종료: 로비]
```

---

## 5. API 연동 설계

### 5.1 API 엔드포인트

| 엔드포인트 | 메서드 | 용도 | Unity 호출 시점 |
|-----------|--------|------|----------------|
| `/items` | GET | 자료 목록 | VR 갤러리 로드 |
| `/items/{id}` | GET | 자료 상세 | 상세 패널 표시 |
| `/items/by-image/{hash}` | GET | 이미지로 자료 조회 | AR 인식 후 |
| `/persons` | GET | 인물 목록 | 인물 태그 데이터 |
| `/persons/{id}` | GET | 인물 상세 | 태그 터치 시 |
| `/search` | POST | 검색 | 검색 기능 |
| `/timeline` | GET | 타임라인 | 연대별 전시 |
| `/stories/{id}` | GET | 스토리 | 가이드 투어 |
| `/graph/related/{id}` | GET | 관련 자료 | 관련 자료 표시 |

### 5.2 API 요청/응답 예시

#### 자료 조회
```json
// GET /items/item_001
// Response:
{
    "item_id": "item_001",
    "title": "1975년 LA 가족 사진",
    "material_type": "photograph",
    "description": "미국 이민 후 첫 가족 사진",
    "date": "1975-06-15",
    "location": "Los Angeles, CA",
    "file_url": "/files/photo_1975.jpg",
    "thumbnail_url": "/thumbnails/photo_1975_thumb.jpg",
    "persons": [
        {
            "person_id": "person_001",
            "name": "김철수",
            "position": {"x": 0.3, "y": 0.5}  // 이미지 내 위치 (0-1)
        },
        {
            "person_id": "person_002", 
            "name": "김영희",
            "position": {"x": 0.6, "y": 0.5}
        }
    ],
    "audio_url": "/audio/narration_001.mp3",
    "tags": ["가족", "LA", "1970년대"]
}
```

#### AR 이미지 인식 조회
```json
// GET /items/by-image/abc123hash
// Response:
{
    "found": true,
    "item": { ... },
    "confidence": 0.95
}
```

### 5.3 Unity-Server 통신 흐름

```
[Unity AR/VR App]                    [Python FastAPI Server]
       │                                      │
       │  1. GET /items (갤러리 로드)          │
       │─────────────────────────────────────>│
       │<─────────────────────────────────────│
       │     [자료 목록 JSON]                  │
       │                                      │
       │  2. GET /items/{id} (상세 조회)       │
       │─────────────────────────────────────>│
       │<─────────────────────────────────────│
       │     [자료 상세 + 인물 위치]           │
       │                                      │
       │  3. GET /files/{filename} (이미지)   │
       │─────────────────────────────────────>│
       │<─────────────────────────────────────│
       │     [이미지 바이너리]                 │
       │                                      │
       │  4. GET /audio/{filename} (음성)     │
       │─────────────────────────────────────>│
       │<─────────────────────────────────────│
       │     [오디오 스트림]                   │
```

---

## 6. 데이터 모델

### 6.1 Unity 데이터 클래스

```csharp
// 자료 데이터
[System.Serializable]
public class ItemData
{
    public string item_id;
    public string title;
    public string material_type;
    public string description;
    public string date;
    public string location;
    public string file_url;
    public string thumbnail_url;
    public string audio_url;
    public List<PersonTagData> persons;
    public List<string> tags;
}

// 인물 태그 데이터
[System.Serializable]
public class PersonTagData
{
    public string person_id;
    public string name;
    public Vector2 position;  // 이미지 내 위치 (0-1 정규화)
}

// 인물 상세 데이터
[System.Serializable]
public class PersonData
{
    public string person_id;
    public string name_korean;
    public string name_romanized;
    public int birth_year;
    public string gender;
    public int generation;
    public string birth_place;
    public int migration_year;
    public string occupation;
}

// 스토리/투어 데이터
[System.Serializable]
public class StoryData
{
    public string story_id;
    public string title;
    public List<StorySegment> segments;
}

[System.Serializable]
public class StorySegment
{
    public string segment_id;
    public string title;
    public string content;
    public string narration;
    public List<string> item_ids;
    public Vector3 tour_position;  // VR 투어 위치
}
```

### 6.2 로컬 캐시 구조

```
/PersistentDataPath/
├── cache/
│   ├── items/
│   │   ├── item_001.json
│   │   └── item_002.json
│   ├── images/
│   │   ├── photo_1975.jpg
│   │   └── photo_1975_thumb.jpg
│   └── audio/
│       └── narration_001.mp3
└── settings.json
```

---

## 7. UI/UX 설계

### 7.1 AR UI 컴포넌트

| 컴포넌트 | 위치 | 기능 |
|---------|------|------|
| 상단바 | Screen Top | 뒤로가기, 제목, 검색 |
| 정보 오버레이 | World Space (자료 하단) | 제목, 날짜, 장소 |
| 인물 태그 | World Space (인물 위치) | 이름, 터치 가능 |
| 상세 패널 | Screen Center | 자료/인물 상세 정보 |
| 하단 버튼 | Screen Bottom | 음성, 상세, 관련 자료 |
| 로딩 인디케이터 | Screen Center | 로딩 상태 |

### 7.2 VR UI 컴포넌트

| 컴포넌트 | 위치 | 기능 |
|---------|------|------|
| 손목 메뉴 | 왼손 컨트롤러 | 메인 메뉴, 설정 |
| 정보 패널 | World Space (자료 옆) | 자료 설명 |
| 포인터 | 오른손 컨트롤러 | 선택, 텔레포트 |
| 투어 가이드 | World Space (고정) | 현재 위치, 진행률 |
| 미니맵 | 손목 메뉴 | 갤러리 전체 지도 |

### 7.3 접근성 고려사항

- 고대비 텍스트 옵션
- 자막 지원 (음성 설명)
- 좌식/기립 모드 (VR)
- 멀미 방지 옵션 (터널 비전)
- 음성 명령 지원

---

## 8. 기술 스택

### 8.1 개발 환경

| 항목 | 버전/도구 |
|------|----------|
| Unity | 2022.3 LTS |
| Scripting Runtime | .NET Standard 2.1 |
| IDE | Visual Studio 2022 / Rider |

### 8.2 필수 패키지

| 패키지 | 용도 | 버전 |
|-------|------|------|
| AR Foundation | AR 기반 | 5.x |
| ARCore XR Plugin | Android AR | 5.x |
| ARKit XR Plugin | iOS AR | 5.x |
| XR Interaction Toolkit | VR 인터랙션 | 2.x |
| OpenXR Plugin | VR 표준 | 1.x |
| Vuforia Engine | 이미지 인식 | 10.x |
| TextMeshPro | UI 텍스트 | 3.x |
| Newtonsoft JSON | JSON 파싱 | 3.x |
| UniTask | 비동기 처리 | 2.x |
| DOTween | 애니메이션 | 1.x |

### 8.3 빌드 타겟

| 플랫폼 | 최소 사양 | 용도 |
|-------|----------|------|
| Android | API 26+ (ARCore) | AR 모바일 |
| iOS | 12.0+ (ARKit) | AR 모바일 |
| Oculus Quest | Quest 2+ | VR 스탠드얼론 |
| Windows PC | OpenXR 호환 | VR PC |

---

## 부록: 개발 일정 (예시)

| 단계 | 기간 | 산출물 |
|------|------|--------|
| 1. 프로젝트 설정 | 1주 | Unity 프로젝트, 패키지 설치 |
| 2. API 연동 | 1주 | APIManager, 데이터 모델 |
| 3. AR 기본 기능 | 2주 | 이미지 인식, 오버레이 |
| 4. AR 고급 기능 | 1주 | 인물 태그, 음성 |
| 5. VR 기본 기능 | 2주 | 갤러리, 텔레포트 |
| 6. VR 고급 기능 | 1주 | 가이드 투어 |
| 7. UI/UX 완성 | 1주 | 폴리싱, 테스트 |
| 8. 빌드/배포 | 1주 | APK, Quest 빌드 |
| **합계** | **10주** | |
