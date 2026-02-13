"""
지식그래프 모듈 (Knowledge Graph Module)

Neo4j 기반 지식그래프 구축 및 관리

주요 기능:
- 노드 생성/관리 (자료, 인물, 장소, 행사)
- 엣지(관계) 생성/관리
- 그래프 쿼리 및 탐색
- 경로 탐색 및 추천

Author: Diaspora Archive Project
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque, defaultdict
import json

from .schema import UnifiedMetadata, PersonInfo, LocationInfo, EventInfo
from .relation_engine import Relation, RelationType

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """노드 유형"""
    ITEM = "Item"
    PERSON = "Person"
    LOCATION = "Location"
    EVENT = "Event"
    COLLECTION = "Collection"
    TOPIC = "Topic"


@dataclass
class Node:
    """그래프 노드"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    labels: List[str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = [self.node_type.value]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "properties": self.properties,
            "labels": self.labels
        }


@dataclass
class Edge:
    """그래프 엣지"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "properties": self.properties
        }


class KnowledgeGraph:
    """지식그래프 (Neo4j 연동)"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.uri = self.config.get("uri", "bolt://localhost:7687")
        self.user = self.config.get("user", "neo4j")
        self.password = self.config.get("password", "")
        
        self.driver = None
        self.connected = False
        
        # 인메모리 저장소 (Neo4j 미연결 시 사용)
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(f"KnowledgeGraph initialized (uri: {self.uri})")
    
    def connect(self) -> bool:
        """Neo4j 연결"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
            logger.info("Connected to Neo4j")
            return True
        except ImportError:
            logger.warning("neo4j driver not installed, using in-memory mode")
            return False
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
    
    def disconnect(self):
        """Neo4j 연결 해제"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j")
    
    def create_indexes(self):
        """인덱스 생성"""
        if not self.connected:
            return
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Item) ON (n.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Location) ON (n.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Event) ON (n.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Topic) ON (n.name)",
        ]
        
        with self.driver.session() as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        logger.info("Indexes created")
    
    # ============ 노드 생성 ============
    
    def create_item_node(self, metadata: UnifiedMetadata) -> Node:
        """자료 노드 생성"""
        props = {
            "item_id": metadata.item_id,
            "title": metadata.dublin_core.title,
            "material_type": str(metadata.material_type),
            "resource_type": str(metadata.dublin_core.type),
            "date": metadata.dublin_core.date,
            "description": metadata.dublin_core.description,
            "creator": metadata.dublin_core.creator,
            "location": metadata.dublin_core.coverage_spatial,
            "file_path": metadata.file_path,
            "thumbnail_path": metadata.thumbnail_path,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            "tags": metadata.tags,
        }
        
        node = Node(
            node_id=metadata.item_id,
            node_type=NodeType.ITEM,
            properties={k: v for k, v in props.items() if v is not None},
            labels=["Item", str(metadata.dublin_core.type).replace(" ", "")]
        )
        
        self._save_node(node)
        return node
    
    def create_person_node(self, person: PersonInfo) -> Node:
        """인물 노드 생성"""
        props = {
            "person_id": person.person_id,
            "name_korean": person.name_korean,
            "name_romanized": person.name_romanized,
            "name_local": person.name_local,
            "birth_year": person.birth_year,
            "death_year": person.death_year,
            "gender": person.gender,
            "generation": int(person.generation) if person.generation else None,
            "birth_place": person.birth_place,
            "migration_year": person.migration_year,
            "occupation": person.occupation,
            "family_role": person.family_role,
        }
        
        node = Node(
            node_id=person.person_id,
            node_type=NodeType.PERSON,
            properties={k: v for k, v in props.items() if v is not None},
            labels=["Person"]
        )
        
        self._save_node(node)
        return node
    
    def create_location_node(self, location: LocationInfo) -> Node:
        """장소 노드 생성"""
        props = {
            "location_id": location.location_id,
            "name_korean": location.name_korean,
            "name_english": location.name_english,
            "country": location.country,
            "country_code": location.country_code,
            "region": location.region,
            "city": location.city,
            "address": location.address,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "location_type": location.location_type,
        }
        
        node = Node(
            node_id=location.location_id,
            node_type=NodeType.LOCATION,
            properties={k: v for k, v in props.items() if v is not None},
            labels=["Location"]
        )
        
        self._save_node(node)
        return node
    
    def create_event_node(self, event: EventInfo) -> Node:
        """행사 노드 생성"""
        props = {
            "event_id": event.event_id,
            "name": event.name,
            "event_type": str(event.event_type),
            "date": event.date,
            "year": event.year,
            "location_name": event.location_name,
            "description": event.description,
            "participant_count": len(event.participants),
        }
        
        node = Node(
            node_id=event.event_id,
            node_type=NodeType.EVENT,
            properties={k: v for k, v in props.items() if v is not None},
            labels=["Event"]
        )
        
        self._save_node(node)
        return node
    
    def create_topic_node(self, topic: str) -> Node:
        """주제 노드 생성"""
        topic_id = f"topic_{topic.lower().replace(' ', '_')}"
        
        node = Node(
            node_id=topic_id,
            node_type=NodeType.TOPIC,
            properties={"topic_id": topic_id, "name": topic, "name_lower": topic.lower()},
            labels=["Topic"]
        )
        
        self._save_node(node)
        return node
    
    def _save_node(self, node: Node):
        """노드 저장"""
        if self.connected:
            self._save_node_neo4j(node)
        else:
            self.nodes[node.node_id] = node
    
    def _save_node_neo4j(self, node: Node):
        """Neo4j에 노드 저장"""
        labels = ":".join(node.labels)
        query = f"""
        MERGE (n:{labels} {{node_id: $node_id}})
        SET n += $properties
        """
        with self.driver.session() as session:
            session.run(query, node_id=node.node_id, properties=node.properties)
    
    # ============ 엣지 생성 ============
    
    def create_edge(self, relation: Relation) -> Edge:
        """관계 엣지 생성"""
        edge = Edge(
            edge_id=relation.relation_id or str(uuid.uuid4()),
            source_id=relation.source_id,
            target_id=relation.target_id,
            edge_type=relation.relation_type.value.upper(),
            properties={
                "confidence": relation.confidence,
                "evidence": relation.evidence,
                "is_verified": relation.is_verified,
                "created_at": relation.created_at.isoformat(),
                **relation.metadata
            }
        )
        
        self._save_edge(edge)
        return edge
    
    def create_depicts_edge(self, item_id: str, person_id: str, confidence: float = 1.0) -> Edge:
        """'등장' 관계 생성"""
        edge = Edge(
            edge_id=str(uuid.uuid4()),
            source_id=item_id,
            target_id=person_id,
            edge_type="DEPICTS",
            properties={"confidence": confidence}
        )
        self._save_edge(edge)
        return edge
    
    def create_located_at_edge(self, item_id: str, location_id: str) -> Edge:
        """'장소' 관계 생성"""
        edge = Edge(
            edge_id=str(uuid.uuid4()),
            source_id=item_id,
            target_id=location_id,
            edge_type="LOCATED_AT",
            properties={}
        )
        self._save_edge(edge)
        return edge
    
    def create_part_of_event_edge(self, item_id: str, event_id: str) -> Edge:
        """'행사' 관계 생성"""
        edge = Edge(
            edge_id=str(uuid.uuid4()),
            source_id=item_id,
            target_id=event_id,
            edge_type="PART_OF_EVENT",
            properties={}
        )
        self._save_edge(edge)
        return edge
    
    def create_tagged_with_edge(self, item_id: str, topic_id: str) -> Edge:
        """'태그' 관계 생성"""
        edge = Edge(
            edge_id=str(uuid.uuid4()),
            source_id=item_id,
            target_id=topic_id,
            edge_type="TAGGED_WITH",
            properties={}
        )
        self._save_edge(edge)
        return edge
    
    def _save_edge(self, edge: Edge):
        """엣지 저장"""
        if self.connected:
            self._save_edge_neo4j(edge)
        else:
            self.edges[edge.edge_id] = edge
            self.adjacency[edge.source_id].append(edge.edge_id)
            self.adjacency[edge.target_id].append(edge.edge_id)
    
    def _save_edge_neo4j(self, edge: Edge):
        """Neo4j에 엣지 저장"""
        query = f"""
        MATCH (a {{node_id: $source_id}})
        MATCH (b {{node_id: $target_id}})
        MERGE (a)-[r:{edge.edge_type}]->(b)
        SET r += $properties
        SET r.edge_id = $edge_id
        """
        props = {k: v for k, v in edge.properties.items() if v is not None}
        with self.driver.session() as session:
            session.run(query, source_id=edge.source_id, target_id=edge.target_id,
                       edge_id=edge.edge_id, properties=props)
    
    # ============ 그래프 구축 ============
    
    def build_from_metadata(self, metadata: UnifiedMetadata):
        """메타데이터에서 그래프 구축"""
        # 자료 노드
        self.create_item_node(metadata)
        
        # 인물 노드 및 관계
        for person in metadata.diaspora.persons:
            self.create_person_node(person)
            self.create_depicts_edge(metadata.item_id, person.person_id)
        
        # 장소 노드 및 관계
        for location in metadata.diaspora.locations:
            self.create_location_node(location)
            self.create_located_at_edge(metadata.item_id, location.location_id)
        
        # 행사 노드 및 관계
        for event in metadata.diaspora.events:
            self.create_event_node(event)
            self.create_part_of_event_edge(metadata.item_id, event.event_id)
        
        # 태그 노드 및 관계
        all_tags = set(metadata.tags + metadata.dublin_core.subject + metadata.diaspora.topics)
        for tag in all_tags:
            if tag:
                topic_node = self.create_topic_node(tag)
                self.create_tagged_with_edge(metadata.item_id, topic_node.node_id)
        
        logger.info(f"Built graph for item: {metadata.item_id}")
    
    def build_from_relations(self, relations: List[Relation]):
        """관계 목록에서 그래프 구축"""
        for rel in relations:
            self.create_edge(rel)
        logger.info(f"Built graph from {len(relations)} relations")
    
    # ============ 쿼리 ============
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """노드 조회"""
        if self.connected:
            query = """
            MATCH (n {node_id: $node_id})
            RETURN labels(n) as labels, properties(n) as props
            """
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                if record:
                    node_type = NodeType(record["labels"][0]) if record["labels"] else NodeType.ITEM
                    return Node(node_id=node_id, node_type=node_type,
                               properties=dict(record["props"]), labels=record["labels"])
        else:
            return self.nodes.get(node_id)
        return None
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                     direction: str = "both") -> List[Tuple[Node, Edge]]:
        """이웃 노드 조회"""
        neighbors = []
        
        if self.connected:
            if direction == "out":
                pattern = "(a)-[r]->(b)"
            elif direction == "in":
                pattern = "(a)<-[r]-(b)"
            else:
                pattern = "(a)-[r]-(b)"
            
            edge_filter = ":" + "|".join(edge_types) if edge_types else ""
            query = f"""
            MATCH {pattern.replace('[r]', f'[r{edge_filter}]')}
            WHERE a.node_id = $node_id
            RETURN b.node_id as neighbor_id, labels(b) as labels,
                   properties(b) as props, type(r) as edge_type,
                   properties(r) as edge_props, r.edge_id as edge_id
            """
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    node_type = NodeType(record["labels"][0]) if record["labels"] else NodeType.ITEM
                    neighbor = Node(node_id=record["neighbor_id"], node_type=node_type,
                                   properties=dict(record["props"]), labels=record["labels"])
                    edge = Edge(edge_id=record["edge_id"] or "", source_id=node_id,
                               target_id=record["neighbor_id"], edge_type=record["edge_type"],
                               properties=dict(record["edge_props"]) if record["edge_props"] else {})
                    neighbors.append((neighbor, edge))
        else:
            for edge_id in self.adjacency.get(node_id, []):
                edge = self.edges.get(edge_id)
                if edge:
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                    neighbor = self.nodes.get(neighbor_id)
                    if neighbor:
                        neighbors.append((neighbor, edge))
        
        return neighbors
    
    def find_path(self, start_id: str, end_id: str, max_depth: int = 4) -> List[List[str]]:
        """경로 탐색"""
        if self.connected:
            query = """
            MATCH path = shortestPath((a {node_id: $start_id})-[*1..$max_depth]-(b {node_id: $end_id}))
            RETURN [node in nodes(path) | node.node_id] as path_nodes
            """
            with self.driver.session() as session:
                result = session.run(query, start_id=start_id, end_id=end_id, max_depth=max_depth)
                return [record["path_nodes"] for record in result]
        else:
            # BFS
            queue = deque([(start_id, [start_id])])
            visited = {start_id}
            paths = []
            
            while queue:
                current, path = queue.popleft()
                if len(path) > max_depth + 1:
                    continue
                if current == end_id:
                    paths.append(path)
                    continue
                
                for neighbor, _ in self.get_neighbors(current):
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        queue.append((neighbor.node_id, path + [neighbor.node_id]))
            
            return paths
    
    def get_related_items(self, item_id: str, max_depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """관련 자료 추천"""
        if self.connected:
            query = """
            MATCH (a:Item {node_id: $item_id})-[*1..$max_depth]-(b:Item)
            WHERE a <> b
            WITH b, count(*) as connection_count
            ORDER BY connection_count DESC
            LIMIT $limit
            RETURN b.node_id as item_id, b.title as title,
                   b.material_type as type, connection_count
            """
            with self.driver.session() as session:
                result = session.run(query, item_id=item_id, max_depth=max_depth, limit=limit)
                return [dict(record) for record in result]
        else:
            # 인메모리 탐색
            related = {}
            visited = {item_id}
            queue = [(item_id, 0)]
            
            while queue:
                current, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                
                for neighbor, edge in self.get_neighbors(current):
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        queue.append((neighbor.node_id, depth + 1))
                        
                        if neighbor.node_type == NodeType.ITEM:
                            if neighbor.node_id not in related:
                                related[neighbor.node_id] = {
                                    "item_id": neighbor.node_id,
                                    "title": neighbor.properties.get("title", ""),
                                    "type": neighbor.properties.get("material_type", ""),
                                    "connection_count": 0
                                }
                            related[neighbor.node_id]["connection_count"] += 1
            
            return sorted(related.values(), key=lambda x: x["connection_count"], reverse=True)[:limit]
    
    def search_by_person(self, person_name: str) -> List[Dict[str, Any]]:
        """인물로 자료 검색"""
        if self.connected:
            query = """
            MATCH (p:Person)-[:DEPICTS]-(i:Item)
            WHERE p.name_korean CONTAINS $name OR p.name_romanized CONTAINS $name
            RETURN i.node_id as item_id, i.title as title, p.name_korean as person_name
            """
            with self.driver.session() as session:
                result = session.run(query, name=person_name)
                return [dict(record) for record in result]
        else:
            results = []
            for node in self.nodes.values():
                if node.node_type == NodeType.PERSON:
                    name_k = node.properties.get("name_korean", "")
                    name_r = node.properties.get("name_romanized", "")
                    if person_name in str(name_k) or person_name in str(name_r):
                        for neighbor, edge in self.get_neighbors(node.node_id):
                            if neighbor.node_type == NodeType.ITEM:
                                results.append({
                                    "item_id": neighbor.node_id,
                                    "title": neighbor.properties.get("title", ""),
                                    "person_name": name_k or name_r
                                })
            return results
    
    def search_by_location(self, location_name: str) -> List[Dict[str, Any]]:
        """장소로 자료 검색"""
        if self.connected:
            query = """
            MATCH (l:Location)-[:LOCATED_AT]-(i:Item)
            WHERE l.name_korean CONTAINS $name OR l.name_english CONTAINS $name OR l.city CONTAINS $name
            RETURN i.node_id as item_id, i.title as title, l.name_korean as location_name
            """
            with self.driver.session() as session:
                result = session.run(query, name=location_name)
                return [dict(record) for record in result]
        else:
            results = []
            for node in self.nodes.values():
                if node.node_type == NodeType.LOCATION:
                    props = node.properties
                    if (location_name in str(props.get("name_korean", "")) or
                        location_name in str(props.get("name_english", "")) or
                        location_name in str(props.get("city", ""))):
                        for neighbor, edge in self.get_neighbors(node.node_id):
                            if neighbor.node_type == NodeType.ITEM:
                                results.append({
                                    "item_id": neighbor.node_id,
                                    "title": neighbor.properties.get("title", ""),
                                    "location_name": props.get("name_korean") or props.get("name_english")
                                })
            return results
    
    def search_by_event(self, event_type: str) -> List[Dict[str, Any]]:
        """행사 유형으로 자료 검색"""
        if self.connected:
            query = """
            MATCH (e:Event)-[:PART_OF_EVENT]-(i:Item)
            WHERE e.event_type CONTAINS $event_type
            RETURN i.node_id as item_id, i.title as title, e.name as event_name, e.date as event_date
            """
            with self.driver.session() as session:
                result = session.run(query, event_type=event_type)
                return [dict(record) for record in result]
        else:
            results = []
            for node in self.nodes.values():
                if node.node_type == NodeType.EVENT:
                    if event_type.lower() in str(node.properties.get("event_type", "")).lower():
                        for neighbor, edge in self.get_neighbors(node.node_id):
                            if neighbor.node_type == NodeType.ITEM:
                                results.append({
                                    "item_id": neighbor.node_id,
                                    "title": neighbor.properties.get("title", ""),
                                    "event_name": node.properties.get("name"),
                                    "event_date": node.properties.get("date")
                                })
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """그래프 통계"""
        if self.connected:
            query = """
            MATCH (n) 
            RETURN labels(n)[0] as type, count(*) as count
            """
            with self.driver.session() as session:
                result = session.run(query)
                node_stats = {record["type"]: record["count"] for record in result}
            
            query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            """
            with self.driver.session() as session:
                result = session.run(query)
                edge_stats = {record["type"]: record["count"] for record in result}
            
            return {"nodes": node_stats, "edges": edge_stats}
        else:
            node_stats = defaultdict(int)
            for node in self.nodes.values():
                node_stats[node.node_type.value] += 1
            
            edge_stats = defaultdict(int)
            for edge in self.edges.values():
                edge_stats[edge.edge_type] += 1
            
            return {"nodes": dict(node_stats), "edges": dict(edge_stats)}
    
    # ============ 내보내기/가져오기 ============
    
    def export_graph(self, output_path: str):
        """그래프 내보내기"""
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    
    def import_graph(self, input_path: str):
        """그래프 가져오기"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for node_data in data.get("nodes", []):
            node = Node(
                node_id=node_data["node_id"],
                node_type=NodeType(node_data["node_type"]),
                properties=node_data["properties"],
                labels=node_data.get("labels")
            )
            self._save_node(node)
        
        for edge_data in data.get("edges", []):
            edge = Edge(
                edge_id=edge_data["edge_id"],
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=edge_data["edge_type"],
                properties=edge_data.get("properties", {})
            )
            self._save_edge(edge)
        
        logger.info(f"Imported graph: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
