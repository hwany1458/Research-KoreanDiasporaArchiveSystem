"""
디지털 아카이브 API 서버 (Digital Archive API Server)

FastAPI 기반 REST API 서버

주요 기능:
- 자료 CRUD (생성, 조회, 수정, 삭제)
- 검색 및 필터링
- 파일 업로드/다운로드
- 관계 그래프 조회
- 타임라인 조회
- 인물/장소/행사 관리

Author: Diaspora Archive Project
"""

import os
import logging
import uuid
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 내부 모듈
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.metadata.schema import (
    UnifiedMetadata, MetadataFactory, PersonInfo, LocationInfo, EventInfo,
    MaterialType, DiasporaRegion, ProcessingStatus
)
from src.core.metadata.storage import UnifiedStorage
from src.core.metadata.knowledge_graph import KnowledgeGraph
from src.core.metadata.relation_engine import RelationInferenceEngine

logger = logging.getLogger(__name__)

# ============ Pydantic 모델 (API 스키마) ============

class ItemCreate(BaseModel):
    """자료 생성 요청"""
    title: str
    material_type: str = "photograph"
    description: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    creator: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
class ItemUpdate(BaseModel):
    """자료 수정 요청"""
    title: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    tags: Optional[List[str]] = None
    
class ItemResponse(BaseModel):
    """자료 응답"""
    item_id: str
    title: str
    material_type: str
    description: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    creator: Optional[str] = None
    tags: List[str] = []
    thumbnail_url: Optional[str] = None
    file_url: Optional[str] = None
    processing_status: str = "uploaded"
    created_at: Optional[str] = None

class PersonCreate(BaseModel):
    """인물 생성 요청"""
    name_korean: str
    name_romanized: Optional[str] = None
    birth_year: Optional[int] = None
    gender: Optional[str] = None
    generation: int = 0
    birth_place: Optional[str] = None
    migration_year: Optional[int] = None

class PersonResponse(BaseModel):
    """인물 응답"""
    person_id: str
    name_korean: Optional[str] = None
    name_romanized: Optional[str] = None
    birth_year: Optional[int] = None
    gender: Optional[str] = None
    generation: int = 0
    item_count: int = 0

class LocationCreate(BaseModel):
    """장소 생성 요청"""
    name_korean: str
    name_english: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class LocationResponse(BaseModel):
    """장소 응답"""
    location_id: str
    name_korean: Optional[str] = None
    name_english: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    item_count: int = 0

class SearchRequest(BaseModel):
    """검색 요청"""
    query: str
    material_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    location: Optional[str] = None
    person: Optional[str] = None
    tags: Optional[List[str]] = None
    page: int = 1
    page_size: int = 20

class SearchResponse(BaseModel):
    """검색 응답"""
    total: int
    page: int
    page_size: int
    items: List[ItemResponse]

class TimelineEvent(BaseModel):
    """타임라인 이벤트"""
    year: int
    month: Optional[int] = None
    items: List[ItemResponse]
    event_count: int

class GraphNode(BaseModel):
    """그래프 노드"""
    id: str
    type: str
    label: str
    properties: Dict[str, Any] = {}

class GraphEdge(BaseModel):
    """그래프 엣지"""
    source: str
    target: str
    type: str
    weight: float = 1.0

class GraphResponse(BaseModel):
    """그래프 응답"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class StatsResponse(BaseModel):
    """통계 응답"""
    total_items: int
    total_persons: int
    total_locations: int
    total_events: int
    items_by_type: Dict[str, int]
    items_by_year: Dict[str, int]
    items_by_region: Dict[str, int]


# ============ API 서버 클래스 ============

class DigitalArchiveAPI:
    """디지털 아카이브 API 서버"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 저장소 설정
        self.storage_config = self.config.get("storage", {
            "local_path": "./data",
            "use_local_fallback": True
        })
        
        # 업로드 디렉토리
        self.upload_dir = self.config.get("upload_dir", "./uploads")
        self.thumbnail_dir = self.config.get("thumbnail_dir", "./thumbnails")
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.thumbnail_dir, exist_ok=True)
        
        # 저장소 및 그래프 초기화
        self.storage: Optional[UnifiedStorage] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.relation_engine: Optional[RelationInferenceEngine] = None
        
        # FastAPI 앱
        self.app = self._create_app()
        
        logger.info("DigitalArchiveAPI initialized")
    
    def _create_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        app = FastAPI(
            title="디아스포라 디지털 아카이브 API",
            description="한인 디아스포라 기록유산 디지털화 시스템 API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS 설정
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 정적 파일 서빙
        if os.path.exists(self.upload_dir):
            app.mount("/files", StaticFiles(directory=self.upload_dir), name="files")
        if os.path.exists(self.thumbnail_dir):
            app.mount("/thumbnails", StaticFiles(directory=self.thumbnail_dir), name="thumbnails")
        
        # 이벤트 핸들러
        @app.on_event("startup")
        async def startup():
            await self._initialize_services()
        
        @app.on_event("shutdown")
        async def shutdown():
            await self._cleanup_services()
        
        # 라우트 등록
        self._register_routes(app)
        
        return app
    
    async def _initialize_services(self):
        """서비스 초기화"""
        logger.info("Initializing services...")
        
        # 저장소 연결
        self.storage = UnifiedStorage(self.storage_config)
        self.storage.connect_all()
        
        # 지식그래프 연결
        self.knowledge_graph = KnowledgeGraph(self.config.get("neo4j", {}))
        self.knowledge_graph.connect()
        
        # 관계 추론 엔진
        self.relation_engine = RelationInferenceEngine({"device": "cpu"})
        
        logger.info("Services initialized")
    
    async def _cleanup_services(self):
        """서비스 정리"""
        if self.storage:
            self.storage.disconnect_all()
        if self.knowledge_graph:
            self.knowledge_graph.disconnect()
        logger.info("Services cleaned up")
    
    def _register_routes(self, app: FastAPI):
        """라우트 등록"""
        
        # ===== 기본 =====
        
        @app.get("/", tags=["기본"])
        async def root():
            """API 루트"""
            return {
                "name": "디아스포라 디지털 아카이브 API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @app.get("/health", tags=["기본"])
        async def health():
            """헬스 체크"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @app.get("/stats", response_model=StatsResponse, tags=["기본"])
        async def get_stats():
            """전체 통계"""
            stats = self.storage.get_statistics() if self.storage else {}
            graph_stats = self.knowledge_graph.get_statistics() if self.knowledge_graph else {}
            
            return StatsResponse(
                total_items=stats.get("total_items", 0),
                total_persons=graph_stats.get("nodes", {}).get("Person", 0),
                total_locations=graph_stats.get("nodes", {}).get("Location", 0),
                total_events=graph_stats.get("nodes", {}).get("Event", 0),
                items_by_type=stats.get("items_by_type", {}),
                items_by_year={},
                items_by_region={}
            )
        
        # ===== 자료 (Items) =====
        
        @app.get("/items", response_model=SearchResponse, tags=["자료"])
        async def list_items(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            material_type: Optional[str] = None,
            sort_by: str = "created_at",
            sort_order: str = "desc"
        ):
            """자료 목록 조회"""
            results = self.storage.search(
                query="*",
                filters={"material_type": material_type} if material_type else None,
                from_=(page - 1) * page_size,
                size=page_size
            ) if self.storage else {"total": 0, "hits": []}
            
            items = [
                ItemResponse(
                    item_id=hit.get("item_id", ""),
                    title=hit.get("title", ""),
                    material_type=hit.get("material_type", ""),
                    description=hit.get("description"),
                    thumbnail_url=hit.get("thumbnail_path"),
                    processing_status=hit.get("processing_status", "uploaded")
                )
                for hit in results.get("hits", [])
            ]
            
            return SearchResponse(
                total=results.get("total", 0),
                page=page,
                page_size=page_size,
                items=items
            )
        
        @app.get("/items/{item_id}", response_model=ItemResponse, tags=["자료"])
        async def get_item(item_id: str):
            """자료 상세 조회"""
            metadata = self.storage.get(item_id) if self.storage else None
            
            if not metadata:
                raise HTTPException(status_code=404, detail="자료를 찾을 수 없습니다")
            
            return ItemResponse(
                item_id=metadata.item_id,
                title=metadata.dublin_core.title,
                material_type=str(metadata.material_type),
                description=metadata.dublin_core.description,
                date=metadata.dublin_core.date,
                location=metadata.dublin_core.coverage_spatial,
                creator=metadata.dublin_core.creator,
                tags=metadata.tags,
                thumbnail_url=metadata.thumbnail_path,
                file_url=metadata.storage_path,
                processing_status=str(metadata.processing_status),
                created_at=metadata.created_at.isoformat() if metadata.created_at else None
            )
        
        @app.post("/items", response_model=ItemResponse, tags=["자료"])
        async def create_item(item: ItemCreate):
            """자료 생성"""
            # 메타데이터 생성
            material_type = MaterialType(item.material_type) if item.material_type else MaterialType.OTHER
            
            metadata = MetadataFactory.create_for_image(
                title=item.title,
                file_path="",
                width=0,
                height=0,
                material_type=material_type
            )
            
            metadata.dublin_core.description = item.description
            metadata.dublin_core.date = item.date
            metadata.dublin_core.coverage_spatial = item.location
            metadata.dublin_core.creator = item.creator
            metadata.tags = item.tags
            
            # 저장
            if self.storage:
                self.storage.save(metadata)
            
            # 지식그래프에 추가
            if self.knowledge_graph:
                self.knowledge_graph.build_from_metadata(metadata)
            
            return ItemResponse(
                item_id=metadata.item_id,
                title=metadata.dublin_core.title,
                material_type=str(metadata.material_type),
                description=metadata.dublin_core.description,
                date=metadata.dublin_core.date,
                location=metadata.dublin_core.coverage_spatial,
                creator=metadata.dublin_core.creator,
                tags=metadata.tags,
                processing_status="uploaded",
                created_at=metadata.created_at.isoformat()
            )
        
        @app.put("/items/{item_id}", response_model=ItemResponse, tags=["자료"])
        async def update_item(item_id: str, item: ItemUpdate):
            """자료 수정"""
            metadata = self.storage.get(item_id) if self.storage else None
            
            if not metadata:
                raise HTTPException(status_code=404, detail="자료를 찾을 수 없습니다")
            
            if item.title:
                metadata.dublin_core.title = item.title
            if item.description is not None:
                metadata.dublin_core.description = item.description
            if item.date is not None:
                metadata.dublin_core.date = item.date
            if item.location is not None:
                metadata.dublin_core.coverage_spatial = item.location
            if item.tags is not None:
                metadata.tags = item.tags
            
            metadata.update_timestamp()
            
            if self.storage:
                self.storage.save(metadata)
            
            return ItemResponse(
                item_id=metadata.item_id,
                title=metadata.dublin_core.title,
                material_type=str(metadata.material_type),
                description=metadata.dublin_core.description,
                date=metadata.dublin_core.date,
                location=metadata.dublin_core.coverage_spatial,
                tags=metadata.tags
            )
        
        @app.delete("/items/{item_id}", tags=["자료"])
        async def delete_item(item_id: str):
            """자료 삭제"""
            # 실제 삭제 로직 구현 필요
            return {"message": "삭제되었습니다", "item_id": item_id}
        
        @app.post("/items/{item_id}/upload", tags=["자료"])
        async def upload_file(item_id: str, file: UploadFile = File(...)):
            """파일 업로드"""
            metadata = self.storage.get(item_id) if self.storage else None
            
            if not metadata:
                raise HTTPException(status_code=404, detail="자료를 찾을 수 없습니다")
            
            # 파일 저장
            file_ext = Path(file.filename).suffix
            file_name = f"{item_id}{file_ext}"
            file_path = os.path.join(self.upload_dir, file_name)
            
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # 메타데이터 업데이트
            metadata.file_path = file_path
            metadata.original_filename = file.filename
            
            if self.storage:
                self.storage.save(metadata, file_path=file_path)
            
            return {
                "message": "업로드 완료",
                "item_id": item_id,
                "file_url": f"/files/{file_name}"
            }
        
        # ===== 검색 =====
        
        @app.post("/search", response_model=SearchResponse, tags=["검색"])
        async def search_items(request: SearchRequest):
            """자료 검색"""
            filters = {}
            if request.material_type:
                filters["material_type"] = request.material_type
            if request.person:
                filters["person"] = request.person
            
            results = self.storage.search(
                query=request.query,
                filters=filters if filters else None,
                from_=(request.page - 1) * request.page_size,
                size=request.page_size
            ) if self.storage else {"total": 0, "hits": []}
            
            items = [
                ItemResponse(
                    item_id=hit.get("item_id", ""),
                    title=hit.get("title", ""),
                    material_type=hit.get("material_type", ""),
                    description=hit.get("description"),
                    thumbnail_url=hit.get("thumbnail_path")
                )
                for hit in results.get("hits", [])
            ]
            
            return SearchResponse(
                total=results.get("total", 0),
                page=request.page,
                page_size=request.page_size,
                items=items
            )
        
        @app.get("/search/suggestions", tags=["검색"])
        async def get_suggestions(q: str = Query(..., min_length=1)):
            """검색어 자동완성"""
            if self.storage and self.storage.elasticsearch.connected:
                suggestions = self.storage.elasticsearch.suggest(q, field="title", size=10)
                return {"suggestions": suggestions}
            return {"suggestions": []}
        
        # ===== 인물 (Persons) =====
        
        @app.get("/persons", tags=["인물"])
        async def list_persons(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100)
        ):
            """인물 목록"""
            if self.knowledge_graph:
                results = []
                for node_id, node in self.knowledge_graph.nodes.items():
                    if node.node_type.value == "Person":
                        results.append(PersonResponse(
                            person_id=node.node_id,
                            name_korean=node.properties.get("name_korean"),
                            name_romanized=node.properties.get("name_romanized"),
                            birth_year=node.properties.get("birth_year"),
                            gender=node.properties.get("gender"),
                            generation=node.properties.get("generation", 0),
                            item_count=len(self.knowledge_graph.get_neighbors(node.node_id))
                        ))
                return {"total": len(results), "persons": results[:page_size]}
            return {"total": 0, "persons": []}
        
        @app.get("/persons/{person_id}", response_model=PersonResponse, tags=["인물"])
        async def get_person(person_id: str):
            """인물 상세"""
            if self.knowledge_graph:
                node = self.knowledge_graph.get_node(person_id)
                if node and node.node_type.value == "Person":
                    return PersonResponse(
                        person_id=node.node_id,
                        name_korean=node.properties.get("name_korean"),
                        name_romanized=node.properties.get("name_romanized"),
                        birth_year=node.properties.get("birth_year"),
                        gender=node.properties.get("gender"),
                        generation=node.properties.get("generation", 0)
                    )
            raise HTTPException(status_code=404, detail="인물을 찾을 수 없습니다")
        
        @app.get("/persons/{person_id}/items", tags=["인물"])
        async def get_person_items(person_id: str):
            """인물 관련 자료"""
            if self.knowledge_graph:
                results = self.knowledge_graph.search_by_person(person_id)
                return {"total": len(results), "items": results}
            return {"total": 0, "items": []}
        
        # ===== 장소 (Locations) =====
        
        @app.get("/locations", tags=["장소"])
        async def list_locations(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100)
        ):
            """장소 목록"""
            if self.knowledge_graph:
                results = []
                for node_id, node in self.knowledge_graph.nodes.items():
                    if node.node_type.value == "Location":
                        results.append(LocationResponse(
                            location_id=node.node_id,
                            name_korean=node.properties.get("name_korean"),
                            name_english=node.properties.get("name_english"),
                            country=node.properties.get("country"),
                            city=node.properties.get("city")
                        ))
                return {"total": len(results), "locations": results[:page_size]}
            return {"total": 0, "locations": []}
        
        @app.get("/locations/{location_id}/items", tags=["장소"])
        async def get_location_items(location_id: str):
            """장소 관련 자료"""
            if self.knowledge_graph:
                results = self.knowledge_graph.search_by_location(location_id)
                return {"total": len(results), "items": results}
            return {"total": 0, "items": []}
        
        # ===== 타임라인 =====
        
        @app.get("/timeline", tags=["타임라인"])
        async def get_timeline(
            year_from: Optional[int] = None,
            year_to: Optional[int] = None
        ):
            """타임라인 조회"""
            # 연도별 그룹화된 자료 반환
            timeline = []
            
            if self.knowledge_graph:
                year_items = {}
                for node_id, node in self.knowledge_graph.nodes.items():
                    if node.node_type.value == "Item":
                        date = node.properties.get("date")
                        if date:
                            try:
                                year = int(date[:4])
                                if year_from and year < year_from:
                                    continue
                                if year_to and year > year_to:
                                    continue
                                if year not in year_items:
                                    year_items[year] = []
                                year_items[year].append({
                                    "item_id": node.node_id,
                                    "title": node.properties.get("title"),
                                    "material_type": node.properties.get("material_type")
                                })
                            except:
                                pass
                
                for year in sorted(year_items.keys()):
                    timeline.append({
                        "year": year,
                        "items": year_items[year],
                        "count": len(year_items[year])
                    })
            
            return {"timeline": timeline}
        
        # ===== 관계 그래프 =====
        
        @app.get("/graph", response_model=GraphResponse, tags=["그래프"])
        async def get_graph(
            center_id: Optional[str] = None,
            depth: int = Query(2, ge=1, le=4),
            node_types: Optional[str] = None
        ):
            """관계 그래프 조회"""
            nodes = []
            edges = []
            
            if self.knowledge_graph:
                if center_id:
                    # 특정 노드 중심
                    visited = set()
                    queue = [(center_id, 0)]
                    
                    while queue:
                        current_id, current_depth = queue.pop(0)
                        if current_id in visited or current_depth > depth:
                            continue
                        visited.add(current_id)
                        
                        node = self.knowledge_graph.get_node(current_id)
                        if node:
                            nodes.append(GraphNode(
                                id=node.node_id,
                                type=node.node_type.value,
                                label=node.properties.get("title") or node.properties.get("name_korean") or node.properties.get("name") or node.node_id[:8],
                                properties=node.properties
                            ))
                            
                            for neighbor, edge in self.knowledge_graph.get_neighbors(current_id):
                                edges.append(GraphEdge(
                                    source=current_id,
                                    target=neighbor.node_id,
                                    type=edge.edge_type,
                                    weight=edge.properties.get("confidence", 1.0)
                                ))
                                queue.append((neighbor.node_id, current_depth + 1))
                else:
                    # 전체 그래프 (제한적)
                    count = 0
                    for node_id, node in self.knowledge_graph.nodes.items():
                        if count >= 100:
                            break
                        nodes.append(GraphNode(
                            id=node.node_id,
                            type=node.node_type.value,
                            label=node.properties.get("title") or node.properties.get("name_korean") or node.node_id[:8],
                            properties={}
                        ))
                        count += 1
                    
                    for edge_id, edge in list(self.knowledge_graph.edges.items())[:200]:
                        edges.append(GraphEdge(
                            source=edge.source_id,
                            target=edge.target_id,
                            type=edge.edge_type
                        ))
            
            return GraphResponse(nodes=nodes, edges=edges)
        
        @app.get("/graph/related/{item_id}", tags=["그래프"])
        async def get_related_items(
            item_id: str,
            limit: int = Query(10, ge=1, le=50)
        ):
            """관련 자료 추천"""
            if self.knowledge_graph:
                related = self.knowledge_graph.get_related_items(item_id, max_depth=2, limit=limit)
                return {"item_id": item_id, "related": related}
            return {"item_id": item_id, "related": []}
        
        # ===== 처리 작업 =====
        
        @app.post("/items/{item_id}/process", tags=["처리"])
        async def process_item(item_id: str, background_tasks: BackgroundTasks):
            """자료 AI 처리 요청"""
            metadata = self.storage.get(item_id) if self.storage else None
            
            if not metadata:
                raise HTTPException(status_code=404, detail="자료를 찾을 수 없습니다")
            
            # 백그라운드 처리 (실제 구현 필요)
            # background_tasks.add_task(process_item_async, item_id)
            
            return {"message": "처리 요청됨", "item_id": item_id, "status": "queued"}
        
        @app.get("/items/{item_id}/process/status", tags=["처리"])
        async def get_process_status(item_id: str):
            """처리 상태 조회"""
            metadata = self.storage.get(item_id) if self.storage else None
            
            if not metadata:
                raise HTTPException(status_code=404, detail="자료를 찾을 수 없습니다")
            
            return {
                "item_id": item_id,
                "status": str(metadata.processing_status),
                "progress": 100 if metadata.processing_status == ProcessingStatus.PROCESSED else 0
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """서버 실행"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# ============ 서버 생성 함수 ============

def create_api_server(config: Optional[Dict] = None) -> FastAPI:
    """API 서버 생성"""
    api = DigitalArchiveAPI(config)
    return api.app


# ============ CLI 실행 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="디지털 아카이브 API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="호스트")
    parser.add_argument("--port", type=int, default=8000, help="포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드")
    args = parser.parse_args()
    
    config = {
        "storage": {
            "local_path": "./data",
            "use_local_fallback": True
        },
        "upload_dir": "./uploads",
        "thumbnail_dir": "./thumbnails"
    }
    
    api = DigitalArchiveAPI(config)
    
    if args.reload:
        import uvicorn
        uvicorn.run("server:create_api_server", host=args.host, port=args.port, reload=True)
    else:
        api.run(host=args.host, port=args.port)
