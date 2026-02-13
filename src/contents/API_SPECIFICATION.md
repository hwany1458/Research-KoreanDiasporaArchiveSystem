# API 연동 명세서

## Unity ↔ Python FastAPI 서버 통신 규격

---

## 1. 개요

### 1.1 기본 정보

| 항목 | 값 |
|------|-----|
| Base URL (개발) | `http://localhost:8000` |
| Base URL (프로덕션) | `https://api.diaspora-archive.com` |
| 데이터 형식 | JSON (UTF-8) |
| 타임아웃 | 30초 (일반), 60초 (파일) |

### 1.2 엔드포인트 요약

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/items` | GET | 자료 목록 |
| `/items/{id}` | GET | 자료 상세 |
| `/persons` | GET | 인물 목록 |
| `/persons/{id}` | GET | 인물 상세 |
| `/search` | POST | 전문 검색 |
| `/timeline` | GET | 타임라인 |
| `/graph` | GET | 관계 그래프 |
| `/graph/related/{id}` | GET | 관련 자료 |
| `/stories/{id}` | GET | 스토리 |
| `/files/{name}` | GET | 파일 다운로드 |
| `/health` | GET | 헬스 체크 |

---

## 2. 자료 API

### 2.1 자료 목록

```
GET /items?page=1&page_size=20&material_type=photograph
```

**응답:**
```json
{
    "total": 150,
    "page": 1,
    "page_size": 20,
    "items": [
        {
            "item_id": "item_001",
            "title": "1975년 LA 가족 사진",
            "material_type": "photograph",
            "date": "1975-06-15",
            "thumbnail_url": "/thumbnails/photo_1975.jpg"
        }
    ]
}
```

### 2.2 자료 상세

```
GET /items/{item_id}
```

**응답:**
```json
{
    "item_id": "item_001",
    "title": "1975년 LA 가족 사진",
    "material_type": "photograph",
    "description": "미국 이민 후 첫 가족 사진",
    "date": "1975-06-15",
    "location": "Los Angeles, CA",
    "file_url": "/files/photo_1975.jpg",
    "thumbnail_url": "/thumbnails/photo_1975.jpg",
    "audio_url": "/audio/narration_001.mp3",
    "tags": ["가족", "LA", "1970년대"],
    "persons": [
        {
            "person_id": "person_001",
            "name": "김철수",
            "position": {"x": 0.3, "y": 0.5}
        },
        {
            "person_id": "person_002",
            "name": "김영희",
            "position": {"x": 0.6, "y": 0.5}
        }
    ]
}
```

**Unity 모델:**
```csharp
[Serializable]
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
    public List<string> tags;
    public List<PersonTagData> persons;
}

[Serializable]
public class PersonTagData
{
    public string person_id;
    public string name;
    public PositionData position;
}

[Serializable]
public class PositionData
{
    public float x;  // 0-1 정규화
    public float y;
}
```

---

## 3. 인물 API

### 3.1 인물 상세

```
GET /persons/{person_id}
```

**응답:**
```json
{
    "person_id": "person_001",
    "name_korean": "김철수",
    "name_romanized": "Kim Chul-soo",
    "birth_year": 1945,
    "gender": "male",
    "generation": 1,
    "birth_place": "전라남도 목포",
    "migration_year": 1972,
    "migration_destination": "usa_la"
}
```

**Unity 모델:**
```csharp
[Serializable]
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
    public string migration_destination;
    
    public string DisplayName => name_korean ?? name_romanized;
}
```

---

## 4. 검색 API

### 4.1 전문 검색

```
POST /search
Content-Type: application/json

{
    "query": "가족 사진 LA",
    "material_type": "photograph",
    "page": 1,
    "page_size": 20
}
```

**응답:**
```json
{
    "total": 5,
    "page": 1,
    "page_size": 20,
    "items": [...]
}
```

---

## 5. 타임라인 API

```
GET /timeline?year_from=1970&year_to=2000
```

**응답:**
```json
{
    "timeline": [
        {
            "year": 1975,
            "count": 8,
            "items": [
                {"item_id": "item_001", "title": "1975년 가족 사진"}
            ]
        }
    ]
}
```

---

## 6. 그래프 API

### 6.1 관계 그래프

```
GET /graph?center_id=item_001&depth=2
```

**응답:**
```json
{
    "nodes": [
        {"id": "item_001", "type": "Item", "label": "1975년 가족 사진"},
        {"id": "person_001", "type": "Person", "label": "김철수"}
    ],
    "edges": [
        {"source": "item_001", "target": "person_001", "type": "DEPICTS", "weight": 0.95}
    ]
}
```

### 6.2 관련 자료

```
GET /graph/related/item_001?limit=10
```

**응답:**
```json
{
    "item_id": "item_001",
    "related": [
        {"item_id": "item_002", "title": "1976년 가족 사진", "connection_count": 3}
    ]
}
```

---

## 7. 스토리 API

```
GET /stories/{story_id}
```

**응답:**
```json
{
    "story_id": "story_001",
    "title": "김씨 가족의 이야기",
    "segments": [
        {
            "segment_id": "intro",
            "title": "프롤로그",
            "content": "이 가족의 이야기는...",
            "audio_url": "/audio/narration_intro.mp3",
            "item_ids": ["item_001"],
            "year": 1970,
            "choices": [
                {"id": "next", "text": "다음", "next": "segment_1"}
            ]
        }
    ]
}
```

**Unity 모델:**
```csharp
[Serializable]
public class StoryData
{
    public string story_id;
    public string title;
    public List<StorySegment> segments;
}

[Serializable]
public class StorySegment
{
    public string segment_id;
    public string title;
    public string content;
    public string audio_url;
    public List<string> item_ids;
    public int? year;
    public List<StoryChoice> choices;
}

[Serializable]
public class StoryChoice
{
    public string id;
    public string text;
    public string next;
}
```

---

## 8. 파일 API

### 8.1 이미지

```
GET /files/{filename}
GET /thumbnails/{filename}
```

### 8.2 오디오

```
GET /audio/{filename}
```

**Unity 다운로드:**
```csharp
// 이미지
IEnumerator DownloadImage(string url, Action<Texture2D> callback)
{
    using (var request = UnityWebRequestTexture.GetTexture(baseUrl + url))
    {
        yield return request.SendWebRequest();
        if (request.result == UnityWebRequest.Result.Success)
            callback(DownloadHandlerTexture.GetContent(request));
    }
}

// 오디오
IEnumerator DownloadAudio(string url, Action<AudioClip> callback)
{
    using (var request = UnityWebRequestMultimedia.GetAudioClip(baseUrl + url, AudioType.MPEG))
    {
        yield return request.SendWebRequest();
        if (request.result == UnityWebRequest.Result.Success)
            callback(DownloadHandlerAudioClip.GetContent(request));
    }
}
```

---

## 9. 에러 처리

### 에러 응답

```json
{
    "detail": "자료를 찾을 수 없습니다",
    "status_code": 404
}
```

### HTTP 상태 코드

| 코드 | 의미 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 404 | 찾을 수 없음 |
| 500 | 서버 오류 |

### Unity 에러 처리

```csharp
switch (request.result)
{
    case UnityWebRequest.Result.Success:
        // 성공 처리
        break;
    case UnityWebRequest.Result.ConnectionError:
        Debug.LogError("네트워크 연결 오류");
        break;
    case UnityWebRequest.Result.ProtocolError:
        Debug.LogError($"HTTP 오류: {request.responseCode}");
        break;
}
```

---

## 10. 자료 유형 코드

| 코드 | 설명 |
|------|------|
| photograph | 사진 |
| group_photo | 단체 사진 |
| letter | 편지 |
| diary | 일기 |
| home_video | 홈비디오 |
| oral_history | 구술 기록 |

## 11. 디아스포라 지역 코드

| 코드 | 설명 |
|------|------|
| usa_la | 미국 LA |
| usa_ny | 미국 뉴욕 |
| japan | 일본 |
| china_yanbian | 중국 연변 |
| russia_sakhalin | 러시아 사할린 |
