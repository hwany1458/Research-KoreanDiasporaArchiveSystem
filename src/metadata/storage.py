"""
통합 저장소 모듈 (Unified Storage Module)

PostgreSQL, Elasticsearch, MinIO 연동 저장소

주요 기능:
- PostgreSQL: 메타데이터 관계형 저장
- Elasticsearch: 전문 검색 인덱싱
- MinIO/S3: 파일 저장소
- Redis: 캐시 (선택)

Author: Diaspora Archive Project
"""

import os
import logging
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import uuid

from .schema import UnifiedMetadata, MaterialType, ProcessingStatus

logger = logging.getLogger(__name__)


# ============ PostgreSQL 저장소 ============

class PostgreSQLStorage:
    """PostgreSQL 메타데이터 저장소"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 5432)
        self.database = self.config.get("database", "diaspora_archive")
        self.user = self.config.get("user", "postgres")
        self.password = self.config.get("password", "")
        
        self.connection = None
        self.connected = False
        
        logger.info(f"PostgreSQLStorage initialized ({self.host}:{self.port}/{self.database})")
    
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host=self.host, port=self.port, database=self.database,
                user=self.user, password=self.password
            )
            self.connected = True
            logger.info("Connected to PostgreSQL")
            return True
        except ImportError:
            logger.warning("psycopg2 not installed")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def disconnect(self):
        """연결 해제"""
        if self.connection:
            self.connection.close()
            self.connected = False
    
    def create_tables(self):
        """테이블 생성"""
        if not self.connected:
            return
        
        queries = [
            """
            CREATE TABLE IF NOT EXISTS items (
                item_id UUID PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                material_type VARCHAR(50),
                resource_type VARCHAR(50),
                description TEXT,
                creator VARCHAR(200),
                date_created VARCHAR(50),
                coverage_spatial VARCHAR(200),
                coverage_temporal VARCHAR(100),
                rights VARCHAR(50),
                file_path VARCHAR(500),
                storage_path VARCHAR(500),
                thumbnail_path VARCHAR(500),
                file_hash VARCHAR(64),
                processing_status VARCHAR(50),
                collection_id UUID,
                metadata_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS persons (
                person_id UUID PRIMARY KEY,
                name_korean VARCHAR(100),
                name_romanized VARCHAR(100),
                name_local VARCHAR(100),
                birth_year INTEGER,
                death_year INTEGER,
                gender VARCHAR(20),
                generation INTEGER,
                birth_place VARCHAR(200),
                migration_year INTEGER,
                migration_destination VARCHAR(50),
                occupation VARCHAR(100),
                family_role VARCHAR(50),
                metadata_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS locations (
                location_id UUID PRIMARY KEY,
                name_korean VARCHAR(200),
                name_english VARCHAR(200),
                country VARCHAR(100),
                country_code VARCHAR(10),
                region VARCHAR(100),
                city VARCHAR(100),
                address VARCHAR(500),
                latitude DECIMAL(10, 7),
                longitude DECIMAL(10, 7),
                location_type VARCHAR(50),
                diaspora_region VARCHAR(50),
                metadata_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id UUID PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                event_type VARCHAR(50),
                date VARCHAR(50),
                year INTEGER,
                location_id UUID REFERENCES locations(location_id),
                location_name VARCHAR(200),
                description TEXT,
                participant_count INTEGER,
                metadata_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS collections (
                collection_id UUID PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                parent_id UUID REFERENCES collections(collection_id),
                item_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS item_persons (
                item_id UUID REFERENCES items(item_id),
                person_id UUID REFERENCES persons(person_id),
                relation_type VARCHAR(50),
                confidence DECIMAL(3, 2),
                PRIMARY KEY (item_id, person_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS item_locations (
                item_id UUID REFERENCES items(item_id),
                location_id UUID REFERENCES locations(location_id),
                PRIMARY KEY (item_id, location_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS item_events (
                item_id UUID REFERENCES items(item_id),
                event_id UUID REFERENCES events(event_id),
                PRIMARY KEY (item_id, event_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS item_relations (
                relation_id UUID PRIMARY KEY,
                source_id UUID REFERENCES items(item_id),
                target_id UUID REFERENCES items(item_id),
                relation_type VARCHAR(50),
                confidence DECIMAL(3, 2),
                evidence TEXT,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # 인덱스
            "CREATE INDEX IF NOT EXISTS idx_items_material_type ON items(material_type)",
            "CREATE INDEX IF NOT EXISTS idx_items_date ON items(date_created)",
            "CREATE INDEX IF NOT EXISTS idx_items_status ON items(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name_korean)",
            "CREATE INDEX IF NOT EXISTS idx_locations_city ON locations(city)",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
        ]
        
        cursor = self.connection.cursor()
        for query in queries:
            try:
                cursor.execute(query)
            except Exception as e:
                logger.warning(f"Table creation warning: {e}")
        self.connection.commit()
        cursor.close()
        logger.info("PostgreSQL tables created")
    
    def save_metadata(self, metadata: UnifiedMetadata) -> bool:
        """메타데이터 저장"""
        if not self.connected:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # 메인 아이템
            cursor.execute("""
                INSERT INTO items (item_id, title, material_type, resource_type, description,
                    creator, date_created, coverage_spatial, coverage_temporal, rights,
                    file_path, storage_path, thumbnail_path, file_hash, processing_status,
                    collection_id, metadata_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (item_id) DO UPDATE SET
                    title = EXCLUDED.title, material_type = EXCLUDED.material_type,
                    processing_status = EXCLUDED.processing_status, metadata_json = EXCLUDED.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                metadata.item_id, metadata.dublin_core.title, str(metadata.material_type),
                str(metadata.dublin_core.type), metadata.dublin_core.description,
                metadata.dublin_core.creator, metadata.dublin_core.date_created,
                metadata.dublin_core.coverage_spatial, metadata.dublin_core.coverage_temporal,
                str(metadata.dublin_core.rights), metadata.file_path, metadata.storage_path,
                metadata.thumbnail_path, metadata.file_hash, str(metadata.processing_status),
                metadata.collection_id, json.dumps(metadata.to_dict(), ensure_ascii=False, default=str),
                datetime.now()
            ))
            
            # 인물
            for person in metadata.diaspora.persons:
                cursor.execute("""
                    INSERT INTO persons (person_id, name_korean, name_romanized, birth_year,
                        gender, generation, migration_year, occupation, family_role)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (person_id) DO NOTHING
                """, (
                    person.person_id, person.name_korean, person.name_romanized, person.birth_year,
                    person.gender, int(person.generation) if person.generation else None,
                    person.migration_year, person.occupation, person.family_role
                ))
                
                cursor.execute("""
                    INSERT INTO item_persons (item_id, person_id, relation_type)
                    VALUES (%s, %s, 'depicts')
                    ON CONFLICT DO NOTHING
                """, (metadata.item_id, person.person_id))
            
            # 장소
            for location in metadata.diaspora.locations:
                cursor.execute("""
                    INSERT INTO locations (location_id, name_korean, name_english, country,
                        region, city, latitude, longitude, location_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (location_id) DO NOTHING
                """, (
                    location.location_id, location.name_korean, location.name_english,
                    location.country, location.region, location.city,
                    location.latitude, location.longitude, location.location_type
                ))
                
                cursor.execute("""
                    INSERT INTO item_locations (item_id, location_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (metadata.item_id, location.location_id))
            
            # 행사
            for event in metadata.diaspora.events:
                cursor.execute("""
                    INSERT INTO events (event_id, name, event_type, date, year, location_name, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                """, (
                    event.event_id, event.name, str(event.event_type),
                    event.date, event.year, event.location_name, event.description
                ))
                
                cursor.execute("""
                    INSERT INTO item_events (item_id, event_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (metadata.item_id, event.event_id))
            
            self.connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Save metadata error: {e}")
            self.connection.rollback()
            return False
    
    def get_metadata(self, item_id: str) -> Optional[UnifiedMetadata]:
        """메타데이터 조회"""
        if not self.connected:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT metadata_json FROM items WHERE item_id = %s", (item_id,))
            row = cursor.fetchone()
            cursor.close()
            
            if row and row[0]:
                return UnifiedMetadata.from_json(json.dumps(row[0]))
            return None
        except Exception as e:
            logger.error(f"Get metadata error: {e}")
            return None
    
    def search(self, query: str, material_type: Optional[str] = None,
               date_from: Optional[str] = None, date_to: Optional[str] = None,
               limit: int = 50) -> List[Dict[str, Any]]:
        """검색"""
        if not self.connected:
            return []
        
        try:
            sql = """
                SELECT item_id, title, material_type, date_created, 
                       description, thumbnail_path
                FROM items
                WHERE (title ILIKE %s OR description ILIKE %s)
            """
            params = [f"%{query}%", f"%{query}%"]
            
            if material_type:
                sql += " AND material_type = %s"
                params.append(material_type)
            
            if date_from:
                sql += " AND date_created >= %s"
                params.append(date_from)
            
            if date_to:
                sql += " AND date_created <= %s"
                params.append(date_to)
            
            sql += f" ORDER BY created_at DESC LIMIT {limit}"
            
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []


# ============ Elasticsearch 저장소 ============

class ElasticsearchStorage:
    """Elasticsearch 전문 검색 저장소"""
    
    INDEX_NAME = "diaspora_items"
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.hosts = self.config.get("hosts", ["localhost:9200"])
        
        self.client = None
        self.connected = False
        
        logger.info(f"ElasticsearchStorage initialized ({self.hosts})")
    
    def connect(self) -> bool:
        """Elasticsearch 연결"""
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch(self.hosts)
            if self.client.ping():
                self.connected = True
                logger.info("Connected to Elasticsearch")
                return True
            return False
        except ImportError:
            logger.warning("elasticsearch not installed")
            return False
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            return False
    
    def create_index(self):
        """인덱스 생성"""
        if not self.connected:
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "item_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "korean"},
                    "title_keyword": {"type": "keyword"},
                    "description": {"type": "text", "analyzer": "korean"},
                    "material_type": {"type": "keyword"},
                    "resource_type": {"type": "keyword"},
                    "creator": {"type": "text"},
                    "date": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy||epoch_millis", "ignore_malformed": True},
                    "date_text": {"type": "text"},
                    "location": {"type": "text"},
                    "location_keyword": {"type": "keyword"},
                    "persons": {"type": "text"},
                    "person_names": {"type": "keyword"},
                    "events": {"type": "text"},
                    "event_types": {"type": "keyword"},
                    "topics": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "transcription": {"type": "text", "analyzer": "korean"},
                    "processing_status": {"type": "keyword"},
                    "collection_id": {"type": "keyword"},
                    "thumbnail_path": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "analysis": {
                    "analyzer": {
                        "korean": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=self.INDEX_NAME):
                self.client.indices.create(index=self.INDEX_NAME, body=mapping)
                logger.info(f"Created Elasticsearch index: {self.INDEX_NAME}")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def index_metadata(self, metadata: UnifiedMetadata) -> bool:
        """메타데이터 인덱싱"""
        if not self.connected:
            return False
        
        try:
            doc = {
                "item_id": metadata.item_id,
                "title": metadata.dublin_core.title,
                "title_keyword": metadata.dublin_core.title,
                "description": metadata.dublin_core.description,
                "material_type": str(metadata.material_type),
                "resource_type": str(metadata.dublin_core.type),
                "creator": metadata.dublin_core.creator,
                "date_text": metadata.dublin_core.date,
                "location": metadata.dublin_core.coverage_spatial,
                "location_keyword": metadata.dublin_core.coverage_spatial,
                "persons": " ".join([p.name_korean or "" for p in metadata.diaspora.persons]),
                "person_names": [p.name_korean for p in metadata.diaspora.persons if p.name_korean],
                "events": " ".join([e.name for e in metadata.diaspora.events]),
                "event_types": [str(e.event_type) for e in metadata.diaspora.events],
                "topics": metadata.diaspora.topics,
                "tags": metadata.tags,
                "transcription": metadata.diaspora.transcription,
                "processing_status": str(metadata.processing_status),
                "collection_id": metadata.collection_id,
                "thumbnail_path": metadata.thumbnail_path,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "updated_at": datetime.now().isoformat()
            }
            
            self.client.index(index=self.INDEX_NAME, id=metadata.item_id, body=doc)
            return True
        except Exception as e:
            logger.error(f"Index error: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None,
               from_: int = 0, size: int = 20) -> Dict[str, Any]:
        """전문 검색"""
        if not self.connected:
            return {"total": 0, "hits": []}
        
        try:
            must = [{"multi_match": {
                "query": query,
                "fields": ["title^3", "description^2", "transcription", "persons", "location", "tags"],
                "type": "best_fields"
            }}]
            
            filter_clauses = []
            if filters:
                if filters.get("material_type"):
                    filter_clauses.append({"term": {"material_type": filters["material_type"]}})
                if filters.get("event_type"):
                    filter_clauses.append({"term": {"event_types": filters["event_type"]}})
                if filters.get("person"):
                    filter_clauses.append({"term": {"person_names": filters["person"]}})
            
            body = {
                "query": {"bool": {"must": must, "filter": filter_clauses}},
                "from": from_,
                "size": size,
                "highlight": {
                    "fields": {"title": {}, "description": {}, "transcription": {}},
                    "pre_tags": ["<em>"], "post_tags": ["</em>"]
                }
            }
            
            result = self.client.search(index=self.INDEX_NAME, body=body)
            
            return {
                "total": result["hits"]["total"]["value"],
                "hits": [{
                    "item_id": hit["_source"]["item_id"],
                    "title": hit["_source"]["title"],
                    "material_type": hit["_source"]["material_type"],
                    "thumbnail_path": hit["_source"].get("thumbnail_path"),
                    "score": hit["_score"],
                    "highlight": hit.get("highlight", {})
                } for hit in result["hits"]["hits"]]
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "hits": []}
    
    def suggest(self, prefix: str, field: str = "title", size: int = 10) -> List[str]:
        """자동 완성"""
        if not self.connected:
            return []
        
        try:
            body = {
                "query": {"prefix": {field: prefix}},
                "size": size,
                "_source": [field]
            }
            result = self.client.search(index=self.INDEX_NAME, body=body)
            return [hit["_source"][field] for hit in result["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Suggest error: {e}")
            return []


# ============ MinIO 파일 저장소 ============

class MinIOStorage:
    """MinIO/S3 파일 저장소"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.endpoint = self.config.get("endpoint", "localhost:9000")
        self.access_key = self.config.get("access_key", "")
        self.secret_key = self.config.get("secret_key", "")
        self.bucket = self.config.get("bucket", "diaspora-archive")
        self.secure = self.config.get("secure", False)
        
        self.client = None
        self.connected = False
        
        logger.info(f"MinIOStorage initialized ({self.endpoint}/{self.bucket})")
    
    def connect(self) -> bool:
        """MinIO 연결"""
        try:
            from minio import Minio
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            
            # 버킷 생성
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            
            self.connected = True
            logger.info("Connected to MinIO")
            return True
        except ImportError:
            logger.warning("minio not installed")
            return False
        except Exception as e:
            logger.error(f"MinIO connection failed: {e}")
            return False
    
    def upload_file(self, file_path: str, object_name: Optional[str] = None,
                   content_type: Optional[str] = None) -> Optional[str]:
        """파일 업로드"""
        if not self.connected:
            return None
        
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            # 중복 방지를 위한 해시 추가
            file_hash = hashlib.md5(open(file_path, 'rb').read()[:8192]).hexdigest()[:8]
            name, ext = os.path.splitext(object_name)
            object_name = f"{name}_{file_hash}{ext}"
            
            self.client.fput_object(
                self.bucket, object_name, file_path,
                content_type=content_type
            )
            
            storage_path = f"{self.bucket}/{object_name}"
            logger.info(f"Uploaded: {storage_path}")
            return storage_path
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None
    
    def upload_bytes(self, data: bytes, object_name: str,
                    content_type: str = "application/octet-stream") -> Optional[str]:
        """바이트 데이터 업로드"""
        if not self.connected:
            return None
        
        try:
            from io import BytesIO
            data_stream = BytesIO(data)
            
            self.client.put_object(
                self.bucket, object_name, data_stream, len(data),
                content_type=content_type
            )
            
            storage_path = f"{self.bucket}/{object_name}"
            return storage_path
        except Exception as e:
            logger.error(f"Upload bytes error: {e}")
            return None
    
    def download_file(self, object_name: str, file_path: str) -> bool:
        """파일 다운로드"""
        if not self.connected:
            return False
        
        try:
            self.client.fget_object(self.bucket, object_name, file_path)
            return True
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def get_url(self, object_name: str, expires_hours: int = 24) -> Optional[str]:
        """임시 URL 생성"""
        if not self.connected:
            return None
        
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                self.bucket, object_name,
                expires=timedelta(hours=expires_hours)
            )
            return url
        except Exception as e:
            logger.error(f"Get URL error: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """파일 삭제"""
        if not self.connected:
            return False
        
        try:
            self.client.remove_object(self.bucket, object_name)
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def list_files(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """파일 목록"""
        if not self.connected:
            return []
        
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=recursive)
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"List files error: {e}")
            return []


# ============ 통합 저장소 ============

class UnifiedStorage:
    """통합 저장소 관리자"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 개별 저장소 초기화
        self.postgresql = PostgreSQLStorage(self.config.get("postgresql", {}))
        self.elasticsearch = ElasticsearchStorage(self.config.get("elasticsearch", {}))
        self.minio = MinIOStorage(self.config.get("minio", {}))
        
        # 로컬 폴백 저장소
        self.local_path = self.config.get("local_path", "./data")
        self.use_local_fallback = self.config.get("use_local_fallback", True)
        
        logger.info("UnifiedStorage initialized")
    
    def connect_all(self) -> Dict[str, bool]:
        """모든 저장소 연결"""
        results = {
            "postgresql": self.postgresql.connect(),
            "elasticsearch": self.elasticsearch.connect(),
            "minio": self.minio.connect()
        }
        
        if results["postgresql"]:
            self.postgresql.create_tables()
        if results["elasticsearch"]:
            self.elasticsearch.create_index()
        
        logger.info(f"Storage connections: {results}")
        return results
    
    def disconnect_all(self):
        """모든 저장소 연결 해제"""
        self.postgresql.disconnect()
    
    def save(self, metadata: UnifiedMetadata, file_path: Optional[str] = None) -> Dict[str, Any]:
        """메타데이터 및 파일 저장"""
        result = {"item_id": metadata.item_id, "success": False, "storage_path": None}
        
        # 파일 업로드
        if file_path and os.path.exists(file_path):
            if self.minio.connected:
                storage_path = self.minio.upload_file(file_path)
                if storage_path:
                    metadata.storage_path = storage_path
                    result["storage_path"] = storage_path
            elif self.use_local_fallback:
                # 로컬 폴백
                local_dir = os.path.join(self.local_path, "files")
                os.makedirs(local_dir, exist_ok=True)
                import shutil
                dest = os.path.join(local_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest)
                metadata.storage_path = dest
                result["storage_path"] = dest
        
        # PostgreSQL 저장
        if self.postgresql.connected:
            self.postgresql.save_metadata(metadata)
        elif self.use_local_fallback:
            # 로컬 JSON 폴백
            local_dir = os.path.join(self.local_path, "metadata")
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, f"{metadata.item_id}.json"), 'w', encoding='utf-8') as f:
                f.write(metadata.to_json())
        
        # Elasticsearch 인덱싱
        if self.elasticsearch.connected:
            self.elasticsearch.index_metadata(metadata)
        
        result["success"] = True
        logger.info(f"Saved: {metadata.item_id}")
        return result
    
    def get(self, item_id: str) -> Optional[UnifiedMetadata]:
        """메타데이터 조회"""
        if self.postgresql.connected:
            return self.postgresql.get_metadata(item_id)
        elif self.use_local_fallback:
            path = os.path.join(self.local_path, "metadata", f"{item_id}.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return UnifiedMetadata.from_json(f.read())
        return None
    
    def search(self, query: str, filters: Optional[Dict] = None,
               from_: int = 0, size: int = 20) -> Dict[str, Any]:
        """통합 검색"""
        if self.elasticsearch.connected:
            return self.elasticsearch.search(query, filters, from_, size)
        elif self.postgresql.connected:
            results = self.postgresql.search(query, limit=size)
            return {"total": len(results), "hits": results}
        return {"total": 0, "hits": []}
    
    def get_file_url(self, storage_path: str) -> Optional[str]:
        """파일 URL 조회"""
        if self.minio.connected and storage_path.startswith(self.minio.bucket):
            object_name = storage_path.replace(f"{self.minio.bucket}/", "")
            return self.minio.get_url(object_name)
        return storage_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """저장소 통계"""
        stats = {}
        
        if self.postgresql.connected:
            try:
                cursor = self.postgresql.connection.cursor()
                cursor.execute("SELECT material_type, COUNT(*) FROM items GROUP BY material_type")
                stats["items_by_type"] = dict(cursor.fetchall())
                cursor.execute("SELECT COUNT(*) FROM items")
                stats["total_items"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM persons")
                stats["total_persons"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM locations")
                stats["total_locations"] = cursor.fetchone()[0]
                cursor.close()
            except Exception as e:
                logger.error(f"Statistics error: {e}")
        
        if self.minio.connected:
            try:
                files = self.minio.list_files()
                stats["total_files"] = len(files)
            except:
                pass
        
        return stats
