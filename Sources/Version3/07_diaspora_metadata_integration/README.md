# 07_diaspora_metadata_integration — 메타데이터 통합 모듈

02~06 모듈의 출력을 통합하여 Neo4j 지식그래프 구축.

## 기능
- 02~06 JSON 결과 파싱 및 개체 추출
- 인물/장소/조직/연도/사진/문서 노드 생성
- 자료 간 관계 자동 추론

## 그래프 스키마

| 노드 타입 | 속성 |
|----------|------|
| Person | name |
| Place | name |
| Organization | name |
| Year | value |
| Photo | source, caption, face_count |
| Document | source, text_preview |
| Scene | name |

## 의존성
```powershell
pip install neo4j
```

Neo4j 서버 실행 필요 (Docker 권장):
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

## 사용법

```powershell
# Neo4j 사용
python main.py --input data/input/json_results/ --neo4j-password password

# Neo4j 없이 JSON 통합만
python main.py --input data/input/json_results/ --no-neo4j
```

## 출력
- `consolidated_metadata.json` — 통합된 모든 메타데이터
- `integration_report_<timestamp>.json` — 통계
- (Neo4j 사용 시) http://localhost:7474 에서 그래프 조회
