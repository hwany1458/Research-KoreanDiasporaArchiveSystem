"""
src/pipeline.py
AR/VR 콘텐츠 생성 파이프라인

웹 기반 갤러리, AR 마커 데이터, Unity 데이터 export를 생성합니다.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ARVRContentPipeline:
    """AR/VR 콘텐츠 생성 통합 파이프라인."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def generate(
        self,
        input_path: Path,
        output_dir: Path,
        content_type: str = 'web',
        gallery_title: str = '디아스포라 디지털 아카이브',
    ) -> Dict:
        """콘텐츠 생성."""
        result = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(input_path),
            'content_type': content_type,
        }
        
        # 입력 데이터 수집
        items = self._collect_items(input_path)
        result['items_count'] = len(items)
        
        if not items:
            result['error'] = '입력 데이터 없음'
            return result
        
        if self.verbose:
            print(f"\n수집된 항목: {len(items)}개")
        
        # 콘텐츠 생성
        if content_type in ('web', 'all'):
            web_result = self._generate_web_gallery(items, output_dir, gallery_title)
            result['web_gallery'] = web_result
        
        if content_type in ('unity', 'all'):
            unity_result = self._generate_unity_export(items, output_dir)
            result['unity_export'] = unity_result
        
        if content_type in ('ar_marker', 'all'):
            ar_result = self._generate_ar_marker(items, output_dir)
            result['ar_marker'] = ar_result
        
        result['finished_at'] = datetime.now().isoformat()
        return result
    
    def _collect_items(self, input_path: Path) -> List[Dict]:
        """입력 디렉토리에서 처리된 사진들 수집."""
        items = []
        
        # 입력이 단일 JSON인 경우
        if input_path.is_file() and input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                items.extend(self._extract_items_from_records(data))
            elif isinstance(data, dict):
                if 'photos' in data:
                    items.extend(data['photos'])
                else:
                    items.extend(self._extract_items_from_records([data]))
        
        # 입력이 디렉토리인 경우
        elif input_path.is_dir():
            # JSON 파일 찾기
            json_files = list(input_path.rglob('*.json'))
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        items.extend(self._extract_items_from_records(data))
                    elif isinstance(data, dict):
                        if 'photos' in data:
                            items.extend(data['photos'])
                        else:
                            items.extend(self._extract_items_from_records([data]))
                except Exception:
                    continue
            
            # 이미지 파일 직접 수집 (메타데이터 없이)
            if not items:
                for ext in ['.jpg', '.jpeg', '.png']:
                    for img in input_path.rglob(f'*{ext}'):
                        items.append({
                            'source_file': str(img),
                            'caption': img.stem,
                        })
        
        return items
    
    def _extract_items_from_records(self, records: List) -> List[Dict]:
        """JSON 레코드에서 표시 가능한 아이템 추출."""
        items = []
        for r in records:
            if not isinstance(r, dict):
                continue
            
            # 03 모듈 결과 형식
            if 'caption' in r or 'face_count' in r:
                items.append({
                    'source_file': r.get('input_path') or r.get('output_path', ''),
                    'caption': r.get('caption', ''),
                    'scenes': r.get('scenes', []),
                    'face_count': r.get('face_count', 0),
                })
        
        return items
    
    def _generate_web_gallery(
        self,
        items: List[Dict],
        output_dir: Path,
        title: str
    ) -> Dict:
        """반응형 HTML 갤러리 생성."""
        gallery_dir = output_dir / 'web_gallery'
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML 생성
        html_path = gallery_dir / 'index.html'
        html_content = self._render_html(items, title)
        html_path.write_text(html_content, encoding='utf-8')
        
        # 데이터 JSON 분리 저장
        data_path = gallery_dir / 'gallery_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"  HTML 갤러리: {html_path}")
        
        return {
            'success': True,
            'html_path': str(html_path),
            'data_path': str(data_path),
            'item_count': len(items),
        }
    
    def _generate_unity_export(
        self,
        items: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Unity AR Foundation/XR Toolkit용 데이터 export."""
        unity_dir = output_dir / 'unity_export'
        unity_dir.mkdir(parents=True, exist_ok=True)
        
        # Unity ScriptableObject 호환 JSON
        unity_data = {
            'galleryName': 'DiasporaArchive',
            'createdAt': datetime.now().isoformat(),
            'spatialLayout': self._compute_3d_layout(items),
            'items': []
        }
        
        for i, item in enumerate(items):
            unity_data['items'].append({
                'id': f'item_{i:04d}',
                'sourceFile': item.get('source_file', ''),
                'caption': item.get('caption', ''),
                'metadata': {
                    'sceneTags': [
                        s[0] if isinstance(s, (list, tuple)) else str(s)
                        for s in item.get('scenes', [])
                    ],
                    'faceCount': item.get('face_count', 0),
                }
            })
        
        unity_data_path = unity_dir / 'gallery_data.json'
        with open(unity_data_path, 'w', encoding='utf-8') as f:
            json.dump(unity_data, f, ensure_ascii=False, indent=2)
        
        # README
        readme_path = unity_dir / 'README.md'
        readme_path.write_text(
            "# Unity AR/VR Export\n\n"
            "본 디렉토리는 Unity 2022.3 LTS 프로젝트에서 import할 수 있는\n"
            "디아스포라 아카이브 데이터입니다.\n\n"
            "## 사용 방법\n"
            "1. `gallery_data.json`을 Unity 프로젝트의 `Assets/StreamingAssets/`에 복사\n"
            "2. C# 스크립트에서 `JsonUtility.FromJson<GalleryData>()`로 로드\n"
            "3. AR Foundation 또는 XR Interaction Toolkit으로 시각화\n\n"
            "## 공간 배치\n"
            "`spatialLayout.positions`는 각 항목의 (x, y, z) 좌표를 제공합니다.\n"
            "본 좌표계는 Unity의 right-handed coordinate system 기준입니다.\n",
            encoding='utf-8'
        )
        
        if self.verbose:
            print(f"  Unity export: {unity_data_path}")
        
        return {
            'success': True,
            'unity_data_path': str(unity_data_path),
            'item_count': len(items),
        }
    
    def _generate_ar_marker(
        self,
        items: List[Dict],
        output_dir: Path
    ) -> Dict:
        """AR 마커 매칭 데이터 생성 (실물 사진 인식용)."""
        ar_dir = output_dir / 'ar_markers'
        ar_dir.mkdir(parents=True, exist_ok=True)
        
        marker_data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'markers': []
        }
        
        for i, item in enumerate(items):
            marker_data['markers'].append({
                'id': f'marker_{i:04d}',
                'image_source': item.get('source_file', ''),
                'metadata': {
                    'caption': item.get('caption', ''),
                    'overlay_text': item.get('caption', '')[:100],
                }
            })
        
        marker_path = ar_dir / 'markers.json'
        with open(marker_path, 'w', encoding='utf-8') as f:
            json.dump(marker_data, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"  AR 마커 데이터: {marker_path}")
        
        return {
            'success': True,
            'marker_path': str(marker_path),
            'marker_count': len(items),
        }
    
    def _compute_3d_layout(self, items: List[Dict]) -> Dict:
        """간단한 그리드 기반 3D 공간 배치."""
        import math
        
        n = len(items)
        if n == 0:
            return {'positions': []}
        
        # 원형 배치 (간단)
        positions = []
        radius = 5.0
        for i in range(n):
            angle = (i / n) * 2 * math.pi
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = 1.5  # 사람 눈높이
            positions.append({'x': round(x, 2), 'y': y, 'z': round(z, 2)})
        
        return {
            'layout_type': 'circular',
            'radius': radius,
            'positions': positions,
        }
    
    def _render_html(self, items: List[Dict], title: str) -> str:
        """반응형 HTML 갤러리 렌더링."""
        cards = []
        for i, item in enumerate(items):
            source = item.get('source_file', '')
            caption = item.get('caption', '') or '(설명 없음)'
            face_count = item.get('face_count', 0)
            scenes = item.get('scenes', [])
            scene_str = ''
            if scenes and isinstance(scenes[0], (list, tuple)):
                scene_str = scenes[0][0]
            
            cards.append(f"""
            <div class="card">
                <img src="{source}" alt="{caption}" onerror="this.style.display='none'"/>
                <div class="caption">{caption}</div>
                <div class="meta">
                    <span class="tag">얼굴 {face_count}명</span>
                    {f'<span class="tag">{scene_str}</span>' if scene_str else ''}
                </div>
            </div>
            """)
        
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Malgun Gothic', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            color: #4fc3f7;
            margin-bottom: 30px;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 30px;
            color: #90caf9;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            border-color: #4fc3f7;
        }}
        .card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #2a2a3e;
        }}
        .caption {{
            padding: 12px;
            font-size: 13px;
            line-height: 1.4;
        }}
        .meta {{
            padding: 0 12px 12px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .tag {{
            background: rgba(79, 195, 247, 0.2);
            color: #4fc3f7;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">총 {len(items)}개 자료</div>
    <div class="grid">
        {''.join(cards)}
    </div>
</body>
</html>
"""
        return html
