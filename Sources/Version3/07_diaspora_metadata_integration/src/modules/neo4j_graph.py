"""
src/modules/neo4j_graph.py
Neo4j 지식그래프 구축 모듈
"""

from pathlib import Path
from typing import Dict, List, Optional

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jGraphBuilder:
    """Neo4j 지식그래프 빌더."""
    
    def __init__(
        self,
        uri: str = 'bolt://localhost:7687',
        user: str = 'neo4j',
        password: str = 'password'
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j 라이브러리 필요: pip install neo4j")
        
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"  Neo4j 연결 성공: {uri}")
        except Exception as e:
            print(f"  ⚠ Neo4j 연결 실패: {e}")
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def is_connected(self) -> bool:
        return self.driver is not None
    
    def clear_database(self):
        """기존 데이터 삭제 (주의)."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def build_graph(self, entities_data: Dict) -> Dict:
        """
        entities_data로부터 그래프 구축.
        
        스키마:
            - 노드: Person, Place, Organization, Year, Photo, Document
            - 엣지: APPEARS_IN, LOCATED_AT, BELONGS_TO, FROM_YEAR, MENTIONS
        """
        if not self.driver:
            return {'success': False, 'error': 'Neo4j 미연결'}
        
        stats = {'nodes_created': 0, 'edges_created': 0}
        
        with self.driver.session() as session:
            # 1. 개체 노드 생성
            entities = entities_data.get('entities', {})
            
            for person in entities.get('persons', []):
                session.run(
                    "MERGE (p:Person {name: $name})", name=person
                )
                stats['nodes_created'] += 1
            
            for place in entities.get('places', []):
                session.run(
                    "MERGE (p:Place {name: $name})", name=place
                )
                stats['nodes_created'] += 1
            
            for org in entities.get('organizations', []):
                session.run(
                    "MERGE (o:Organization {name: $name})", name=org
                )
                stats['nodes_created'] += 1
            
            for year in entities.get('years', []):
                session.run(
                    "MERGE (y:Year {value: $value})", value=year
                )
                stats['nodes_created'] += 1
            
            # 2. 사진 노드 생성
            for photo in entities_data.get('photos', []):
                source = photo.get('source_file', '')
                if not source:
                    continue
                
                session.run(
                    """
                    MERGE (ph:Photo {source: $source})
                    SET ph.caption = $caption,
                        ph.face_count = $face_count
                    """,
                    source=source,
                    caption=photo.get('caption'),
                    face_count=photo.get('face_count', 0)
                )
                stats['nodes_created'] += 1
                
                # 사진과 장면 연결
                for scene in photo.get('scenes', []):
                    if isinstance(scene, (list, tuple)) and len(scene) >= 1:
                        scene_name = scene[0]
                        session.run(
                            """
                            MERGE (s:Scene {name: $name})
                            WITH s
                            MATCH (ph:Photo {source: $source})
                            MERGE (ph)-[:DEPICTS]->(s)
                            """,
                            name=scene_name,
                            source=source
                        )
                        stats['edges_created'] += 1
            
            # 3. 문서 노드 생성
            for doc in entities_data.get('documents', []):
                source = doc.get('source_file', '')
                if not source:
                    continue
                
                session.run(
                    """
                    MERGE (d:Document {source: $source})
                    SET d.text_preview = $preview
                    """,
                    source=source,
                    preview=(doc.get('full_text', '') or '')[:200]
                )
                stats['nodes_created'] += 1
        
        return {
            'success': True,
            'stats': stats,
        }
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계 조회."""
        if not self.driver:
            return {}
        
        stats = {}
        with self.driver.session() as session:
            for label in ['Person', 'Place', 'Organization', 'Year', 'Photo', 'Document', 'Scene']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
                stats[f'{label}_count'] = result.single()['cnt']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as cnt")
            stats['total_edges'] = result.single()['cnt']
        
        return stats
