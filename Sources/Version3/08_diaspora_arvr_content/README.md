# 08_diaspora_arvr_content — AR/VR 콘텐츠 생성 모듈

처리된 디아스포라 자료를 다양한 실감형 콘텐츠로 변환.

## 기능
- **웹 갤러리** — 반응형 HTML 갤러리 (즉시 브라우저로 확인)
- **AR 마커** — 실물 사진 인식용 매칭 데이터
- **Unity export** — Unity 2022.3 LTS / AR Foundation / XR Toolkit 연동 데이터
- **3D 공간 배치** — 자동 레이아웃 산출

## 콘텐츠 유형

| 타입 | 출력 | 활용 |
|------|------|------|
| `web` | HTML + JSON | 즉시 웹 브라우저 시연 |
| `unity` | Unity-호환 JSON | Unity AR/VR 프로젝트 import |
| `ar_marker` | AR 매칭 JSON | AR.js, ARFoundation 호환 |
| `all` | 위 모두 | 종합 |

## 의존성
표준 Python만 사용 (Pillow는 선택)

## 사용법

```powershell
# 웹 갤러리만 생성 (가장 빠름)
python main.py --input data/input/processed/ --type web

# Unity 데이터 생성
python main.py --input data/input/processed/ --type unity

# 모두
python main.py --input data/input/processed/ --type all --gallery-title "Howard S. Park 1921"
```

## 입력 형식
- 03 모듈의 batch_summary.json
- 또는 07 모듈의 consolidated_metadata.json
- 또는 이미지 파일이 들어있는 디렉토리

## 출력
```
data/output/
├── web_gallery/
│   ├── index.html       # 브라우저로 열기
│   └── gallery_data.json
├── unity_export/
│   ├── gallery_data.json   # Unity StreamingAssets/에 복사
│   └── README.md
└── ar_markers/
    └── markers.json
```

## Unity 통합 가이드
본 모듈은 Unity 프로젝트에 import할 데이터를 export합니다.
Unity 측 구현은 `Assets/StreamingAssets/gallery_data.json`을 로드하여
AR Foundation의 ImageTracking 또는 XR Interaction Toolkit으로 시각화합니다.
