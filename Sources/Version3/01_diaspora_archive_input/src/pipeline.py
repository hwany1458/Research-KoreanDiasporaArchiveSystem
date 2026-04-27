"""
src/pipeline.py
아카이브 입력 시스템 메인 파이프라인

자료 유형을 자동 분류하고 후속 모듈로 라우팅 정보 생성.
"""

import os
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


# 자료 유형 분류 맵
EXTENSION_MAP = {
    'image': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'},
    'document': {'.pdf', '.doc', '.docx', '.txt', '.hwp', '.rtf'},
    'video': {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'},
    'audio': {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'},
}

# 구술 자료 키워드 (파일명 기반 휴리스틱)
ORAL_KEYWORDS = ['oral', 'interview', '구술', '인터뷰', '증언', 'testimony']

# 후속 모듈 매핑
MODULE_ROUTING = {
    'image': '03_diaspora_image_processing',
    'document': '02_diaspora_document_processing',
    'video': '04_diaspora_video_processing',
    'audio': '05_diaspora_audio_processing',
    'oral': '06_diaspora_oral_processing',
}


class ArchiveInputPipeline:
    """아카이브 입력 자료의 분류 및 라우팅 파이프라인."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def classify_file(self, file_path: Path) -> Dict:
        """단일 파일의 자료 유형 분류."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        filename_lower = file_path.name.lower()
        
        # 1차: 확장자 기반 분류
        category = 'unknown'
        for cat, extensions in EXTENSION_MAP.items():
            if ext in extensions:
                category = cat
                break
        
        # 2차: 음성/구술 추가 분류
        if category == 'audio':
            for keyword in ORAL_KEYWORDS:
                if keyword in filename_lower:
                    category = 'oral'
                    break
        
        # MIME 타입 보조
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            'filename': file_path.name,
            'absolute_path': str(file_path.absolute()),
            'extension': ext,
            'mime_type': mime_type or 'unknown',
            'category': category,
            'target_module': MODULE_ROUTING.get(category, 'manual_review'),
            'file_size_bytes': file_path.stat().st_size if file_path.exists() else 0,
        }
    
    def extract_metadata(self, file_path: Path) -> Dict:
        """파일 메타데이터 추출."""
        file_path = Path(file_path)
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        
        # 파일 해시 (재현성/중복 검사용)
        file_hash = self._compute_hash(file_path)
        
        return {
            'file_hash_md5': file_hash,
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
        }
    
    def _compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """파일의 MD5 해시 계산."""
        try:
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception:
            return ''
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        classify_only: bool = False
    ) -> Dict:
        """단일 파일 처리."""
        if not input_path.exists():
            return {'error': f'파일이 존재하지 않습니다: {input_path}'}
        
        classification = self.classify_file(input_path)
        metadata = self.extract_metadata(input_path) if not classify_only else {}
        
        result = {
            'started_at': datetime.now().isoformat(),
            'mode': 'single_file',
            'classification': classification,
            'metadata': metadata,
            'classify_only': classify_only,
        }
        
        if self.verbose:
            print(f"\n[분류] {input_path.name}")
            print(f"  카테고리: {classification['category']}")
            print(f"  대상 모듈: {classification['target_module']}")
            if metadata:
                print(f"  크기: {metadata.get('size_mb', 0)} MB")
        
        return result
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        classify_only: bool = False
    ) -> Dict:
        """디렉토리 일괄 처리."""
        if not input_dir.is_dir():
            return {'error': f'디렉토리가 아닙니다: {input_dir}'}
        
        files = sorted([p for p in input_dir.rglob('*') if p.is_file()])
        
        records = []
        category_counts: Dict[str, int] = {}
        
        for file_path in files:
            classification = self.classify_file(file_path)
            metadata = self.extract_metadata(file_path) if not classify_only else {}
            
            record = {
                'classification': classification,
                'metadata': metadata,
            }
            records.append(record)
            
            category = classification['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if self.verbose:
                print(f"  [{classification['category']:>8}] {file_path.name}")
        
        result = {
            'started_at': datetime.now().isoformat(),
            'mode': 'directory_batch',
            'input_dir': str(input_dir),
            'total_files': len(files),
            'category_counts': category_counts,
            'records': records,
        }
        
        print(f"\n총 {len(files)}개 파일 분류 완료:")
        for cat, count in sorted(category_counts.items()):
            target = MODULE_ROUTING.get(cat, 'manual_review')
            print(f"  {cat:>10}: {count:>4}개 → {target}")
        
        return result


if __name__ == "__main__":
    # 간단한 테스트
    pipeline = ArchiveInputPipeline(verbose=True)
    print("ArchiveInputPipeline 초기화 완료")
