"""
통합 API 서버 (Full API Server)

디지털 아카이브 + 스토리 생성 통합 서버

실행:
    python -m src.core.api.main --port 8000

Author: Diaspora Archive Project
"""

import os
import logging
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .server import DigitalArchiveAPI
from ..story.generator import StoryGenerator, add_story_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_full_api(config: Optional[Dict] = None) -> FastAPI:
    """
    통합 API 서버 생성
    
    - 디지털 아카이브 API
    - 스토리 생성 API
    """
    config = config or {}
    
    # 기본 API 생성
    archive_api = DigitalArchiveAPI(config)
    app = archive_api.app
    
    # 스토리 생성기
    story_config = config.get("story", {})
    story_generator = StoryGenerator(story_config)
    
    # 스토리 라우트 추가
    add_story_routes(app, story_generator, archive_api.storage)
    
    # 추가 미들웨어
    @app.middleware("http")
    async def add_custom_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0.0"
        return response
    
    # 추가 엔드포인트
    @app.get("/api/info", tags=["기본"])
    async def api_info():
        """API 정보"""
        return {
            "name": "디아스포라 디지털 아카이브 API",
            "version": "1.0.0",
            "modules": {
                "archive": True,
                "story": True,
                "search": True,
                "graph": True
            },
            "endpoints": {
                "items": "/items",
                "search": "/search",
                "persons": "/persons",
                "locations": "/locations",
                "timeline": "/timeline",
                "graph": "/graph",
                "stories": "/stories"
            }
        }
    
    logger.info("Full API server created with story generation support")
    return app


# FastAPI 앱 인스턴스 (uvicorn 직접 실행용)
app = create_full_api({
    "storage": {
        "local_path": os.environ.get("DATA_PATH", "./data"),
        "use_local_fallback": True
    },
    "upload_dir": os.environ.get("UPLOAD_DIR", "./uploads"),
    "thumbnail_dir": os.environ.get("THUMBNAIL_DIR", "./thumbnails"),
    "story": {
        "llm": {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "model": os.environ.get("LLM_MODEL", "gpt-4")
        }
    }
})


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="디아스포라 디지털 아카이브 API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="호스트")
    parser.add_argument("--port", type=int, default=8000, help="포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드")
    parser.add_argument("--workers", type=int, default=1, help="워커 수")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     디아스포라 디지털 아카이브 API 서버                        ║
║     Korean Diaspora Digital Archive API Server               ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {args.host:<10}  Port: {args.port:<10}                       ║
║  Docs: http://{args.host}:{args.port}/docs                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.core.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
