"""
src/pipeline.py
메타데이터 통합 파이프라인
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.modules.entity_extraction import EntityExtractor


class MetadataIntegrationPipeline:
    """메타데이터 통합 파이프라인."""
    
    def __init__(
        self,
        use_neo4j: bool = True,
        neo4j_uri: str = 'bolt://localhost:7687',
        neo4j_user: str = 'neo4j',
        neo4j_password: str = 'password',
        verbose: bool = False
    ):
        self.use_neo4j = use_neo4j
        self.verbose = verbose
        self.extractor = EntityExtractor()
        
        self.graph_builder = None
        if use_neo4j:
            try:
                from src.modules.neo4j_graph import Neo4jGraphBuilder
                self.graph_builder = Neo4jGraphBuilder(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                if not self.graph_builder.is_connected():
                    self.graph_builder = None
            except ImportError:
                if verbose:
                    print("  ⚠ neo4j 라이브러리 미설치, JSON 통합만 수행")
    
    def integrate(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """JSON 결과들을 통합."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_dir': str(input_dir),
        }
        
        if self.verbose:
            print(f"\n[1/3] JSON 결과들에서 개체 추출 ({input_dir})...")
        
        # 1. 개체 추출
        entities_data = self.extractor.extract_from_directory(input_dir)
        result['entity_extraction'] = {
            'persons_count': len(entities_data['entities']['persons']),
            'places_count': len(entities_data['entities']['places']),
            'organizations_count': len(entities_data['entities']['organizations']),
            'dates_count': len(entities_data['entities']['dates']),
            'years_count': len(entities_data['entities']['years']),
            'photos_count': entities_data['photo_count'],
            'documents_count': entities_data['document_count'],
            'source_files': entities_data['source_files_count'],
        }
        
        if self.verbose:
            print(f"  추출된 인물: {len(entities_data['entities']['persons'])}명")
            print(f"  추출된 장소: {len(entities_data['entities']['places'])}곳")
            print(f"  추출된 사진: {entities_data['photo_count']}장")
            print(f"  추출된 문서: {entities_data['document_count']}건")
        
        # 2. 통합 JSON 저장
        if self.verbose:
            print("[2/3] 통합 JSON 저장...")
        
        consolidated_path = output_dir / 'consolidated_metadata.json'
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        result['consolidated_path'] = str(consolidated_path)
        
        # 3. Neo4j 그래프 구축
        if self.graph_builder:
            if self.verbose:
                print("[3/3] Neo4j 그래프 구축...")
            graph_result = self.graph_builder.build_graph(entities_data)
            result['graph'] = graph_result
            
            if graph_result.get('success'):
                stats = self.graph_builder.get_graph_stats()
                result['graph_stats'] = stats
                if self.verbose:
                    print(f"  생성된 노드: {graph_result['stats']['nodes_created']}")
                    print(f"  생성된 엣지: {graph_result['stats']['edges_created']}")
            
            self.graph_builder.close()
        else:
            if self.verbose:
                print("[3/3] Neo4j 비활성화 (JSON 통합만 수행됨)")
            result['graph'] = {'skipped': True}
        
        result['finished_at'] = datetime.now().isoformat()
        return result
