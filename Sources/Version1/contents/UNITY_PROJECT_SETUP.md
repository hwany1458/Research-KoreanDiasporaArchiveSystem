# Unity 프로젝트 설정 가이드

## 한인 디아스포라 AR/VR 프로젝트 구축 가이드

---

## 목차

1. [개발 환경 설정](#1-개발-환경-설정)
2. [프로젝트 생성](#2-프로젝트-생성)
3. [필수 패키지 설치](#3-필수-패키지-설치)
4. [AR 설정 (모바일)](#4-ar-설정-모바일)
5. [VR 설정 (Quest/PC)](#5-vr-설정-questpc)
6. [스크립트 구성](#6-스크립트-구성)
7. [씬 구성](#7-씬-구성)
8. [빌드 설정](#8-빌드-설정)
9. [테스트 가이드](#9-테스트-가이드)

---

## 1. 개발 환경 설정

### 1.1 필수 소프트웨어

| 소프트웨어 | 버전 | 용도 |
|-----------|------|------|
| Unity Hub | 최신 | Unity 버전 관리 |
| Unity Editor | 2022.3 LTS | 게임 엔진 |
| Visual Studio 2022 | 최신 | C# IDE |
| Android Studio | 최신 | Android 빌드 (AR) |
| Xcode | 최신 (Mac) | iOS 빌드 (AR) |

### 1.2 Unity 설치 모듈

Unity Hub에서 설치 시 다음 모듈 포함:

```
☑ Android Build Support
  ☑ Android SDK & NDK Tools
  ☑ OpenJDK
☑ iOS Build Support (Mac only)
☑ Windows Build Support (IL2CPP)
☑ Documentation
```

### 1.3 하드웨어 (테스트용)

- **AR**: Android 폰 (ARCore 지원) 또는 iPhone (ARKit 지원)
- **VR**: Oculus Quest 2/3 또는 PC VR 헤드셋

---

## 2. 프로젝트 생성

### 2.1 새 프로젝트 생성

1. Unity Hub → New Project
2. **템플릿**: 3D (URP) 또는 3D Core
3. **프로젝트명**: `DiasporaAR_VR`
4. Create project

### 2.2 프로젝트 구조

```
DiasporaAR_VR/
├── Assets/
│   ├── Scenes/
│   │   ├── AR_Main.unity
│   │   ├── VR_Gallery.unity
│   │   └── VR_Tour.unity
│   ├── Scripts/
│   │   ├── Common/
│   │   ├── API/
│   │   ├── AR/
│   │   ├── VR/
│   │   ├── Audio/
│   │   └── UI/
│   ├── Prefabs/
│   │   ├── AR/
│   │   ├── VR/
│   │   └── UI/
│   ├── Materials/
│   ├── Textures/
│   ├── Audio/
│   └── Resources/
├── Packages/
└── ProjectSettings/
```

### 2.3 폴더 생성 스크립트

Unity 에디터에서 실행:

```csharp
// Editor 스크립트: CreateProjectStructure.cs
using UnityEditor;
using System.IO;

public class CreateProjectStructure
{
    [MenuItem("Tools/Create Project Structure")]
    public static void Create()
    {
        string[] folders = new string[]
        {
            "Assets/Scenes",
            "Assets/Scripts/Common",
            "Assets/Scripts/API",
            "Assets/Scripts/AR",
            "Assets/Scripts/VR",
            "Assets/Scripts/Audio",
            "Assets/Scripts/UI",
            "Assets/Prefabs/AR",
            "Assets/Prefabs/VR",
            "Assets/Prefabs/UI",
            "Assets/Materials",
            "Assets/Textures",
            "Assets/Audio",
            "Assets/Resources"
        };

        foreach (string folder in folders)
        {
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
        }

        AssetDatabase.Refresh();
    }
}
```

---

## 3. 필수 패키지 설치

### 3.1 Package Manager에서 설치

Window → Package Manager → Unity Registry:

| 패키지 | 용도 |
|-------|------|
| AR Foundation | AR 기본 기능 |
| ARCore XR Plugin | Android AR |
| ARKit XR Plugin | iOS AR |
| XR Interaction Toolkit | VR 인터랙션 |
| XR Plugin Management | XR 설정 관리 |
| OpenXR Plugin | VR 표준 |
| TextMeshPro | UI 텍스트 |

### 3.2 Git URL로 설치

Package Manager → + → Add package from git URL:

```
# Newtonsoft JSON (JSON 파싱)
com.unity.nuget.newtonsoft-json
```

### 3.3 Asset Store / 외부 패키지

| 패키지 | 용도 | 설치 방법 |
|-------|------|----------|
| Vuforia Engine | 이미지 인식 | vuforia.com 등록 후 |
| DOTween | 애니메이션 | Asset Store |
| UniTask | 비동기 처리 | GitHub |

### 3.4 packages/manifest.json 예시

```json
{
  "dependencies": {
    "com.unity.xr.arfoundation": "5.1.0",
    "com.unity.xr.arcore": "5.1.0",
    "com.unity.xr.arkit": "5.1.0",
    "com.unity.xr.interaction.toolkit": "2.5.2",
    "com.unity.xr.openxr": "1.9.1",
    "com.unity.xr.management": "4.4.0",
    "com.unity.textmeshpro": "3.0.6",
    "com.unity.nuget.newtonsoft-json": "3.2.1"
  }
}
```

---

## 4. AR 설정 (모바일)

### 4.1 AR Foundation 설정

#### XR Plugin Management 설정

1. Edit → Project Settings → XR Plug-in Management
2. Android 탭:
   - ☑ ARCore
3. iOS 탭:
   - ☑ ARKit

#### AR Session 씬 구성

```
AR_Main (Scene)
├── AR Session
├── AR Session Origin
│   ├── AR Camera
│   ├── AR Tracked Image Manager
│   └── AR Raycast Manager
├── Directional Light
├── Canvas (Screen Space - Overlay)
│   ├── ScanningUI
│   └── InfoPanel
└── Managers
    ├── ARSessionManager
    ├── ARInfoOverlay
    └── APIManager
```

### 4.2 AR Session 오브젝트 설정

**AR Session Origin에 추가:**

```csharp
// Inspector에서 컴포넌트 추가
[RequireComponent(typeof(ARTrackedImageManager))]
[RequireComponent(typeof(ARRaycastManager))]
```

**AR Tracked Image Manager 설정:**

1. AR Session Origin 선택
2. Add Component → AR Tracked Image Manager
3. Serialized Library → Reference Image Library 생성

### 4.3 Reference Image Library 생성

1. Assets → Create → XR → Reference Image Library
2. 이름: `DiasporaImageLibrary`
3. Add Image 클릭
4. 인식할 사진 추가 (각 이미지에 item_id와 동일한 이름 부여)

### 4.4 Android 빌드 설정

Edit → Project Settings → Player → Android:

```
Other Settings:
- Scripting Backend: IL2CPP
- Target Architectures: ☑ ARM64
- Minimum API Level: 26 (Android 8.0)
- Target API Level: 33+

Configuration:
- Graphics APIs: OpenGLES3, Vulkan
```

### 4.5 iOS 빌드 설정 (Mac only)

Edit → Project Settings → Player → iOS:

```
Other Settings:
- Camera Usage Description: "AR 기능을 위해 카메라 접근이 필요합니다"
- Target minimum iOS Version: 12.0
- Architecture: ARM64
```

---

## 5. VR 설정 (Quest/PC)

### 5.1 XR Plugin Management 설정

Edit → Project Settings → XR Plug-in Management:

**PC, Mac & Linux Standalone 탭:**
- ☑ OpenXR

**Android 탭 (Quest):**
- ☑ OpenXR

### 5.2 OpenXR 설정

Project Settings → XR Plug-in Management → OpenXR:

**Interaction Profiles:**
- Oculus Touch Controller Profile
- HTC Vive Controller Profile (선택)

**Features:**
- ☑ Hand Tracking
- ☑ Controller

### 5.3 VR 씬 구성

```
VR_Gallery (Scene)
├── XR Origin (XR Rig)
│   ├── Camera Offset
│   │   └── Main Camera
│   ├── Left Controller
│   │   ├── XR Controller
│   │   └── XR Ray Interactor
│   └── Right Controller
│       ├── XR Controller
│       └── XR Ray Interactor
├── Teleportation Provider
├── Locomotion System
├── XR Interaction Manager
├── Gallery Room
│   ├── Floor
│   ├── Walls
│   ├── Ceiling
│   └── Exhibits (Parent)
├── Directional Light
└── Managers
    ├── VRSessionManager
    ├── GalleryManager
    ├── GuidedTourManager
    ├── AudioManager
    └── APIManager
```

### 5.4 XR Origin 설정

**XR Origin 컴포넌트:**

```
XR Origin
├── Camera Y Offset: 1.7 (기립 모드) / 0 (좌식)
├── Tracking Origin Mode: Floor
```

**Controller 설정:**

```csharp
// Left/Right Controller에 추가
- XR Controller (Action-based)
- XR Ray Interactor
- XR Interactor Line Visual
```

### 5.5 Teleportation 설정

```
Locomotion System
├── Teleportation Provider
└── Snap Turn Provider (Action-based)
    ├── Turn Amount: 45
    └── Enable Turn Around: true
```

**바닥에 Teleportation Area 추가:**

```csharp
// Floor 오브젝트에 추가
- Teleportation Area
- Collider (Box 또는 Mesh)
```

### 5.6 Quest 빌드 설정

Edit → Project Settings → Player → Android:

```
Other Settings:
- Color Space: Linear
- Graphics APIs: Vulkan, OpenGLES3
- Minimum API Level: 29
- Target API Level: 32
- Scripting Backend: IL2CPP
- Target Architectures: ARM64

XR Settings:
- Stereo Rendering Mode: Single Pass Instanced
```

---

## 6. 스크립트 구성

### 6.1 스크립트 복사

제공된 스크립트를 해당 폴더에 복사:

```
Scripts/
├── Common/
│   ├── Singleton.cs
│   └── Constants.cs
├── API/
│   ├── APIManager.cs
│   └── APIModels.cs
├── AR/
│   ├── ARSessionManager.cs
│   └── ARInfoOverlay.cs
├── VR/
│   ├── VRSessionManager.cs
│   ├── GalleryManager.cs
│   └── ExhibitAndTour.cs
└── Audio/
    └── AudioManager.cs
```

### 6.2 Assembly Definition (선택)

대규모 프로젝트의 경우 Assembly Definition 생성:

```
Scripts/Common/Diaspora.Common.asmdef
Scripts/API/Diaspora.API.asmdef
Scripts/AR/Diaspora.AR.asmdef
Scripts/VR/Diaspora.VR.asmdef
```

---

## 7. 씬 구성

### 7.1 AR_Main 씬 구성 단계

1. **AR Foundation 설정**
   - GameObject → XR → AR Session
   - GameObject → XR → AR Session Origin

2. **이미지 트래킹 설정**
   - AR Session Origin → Add Component → AR Tracked Image Manager
   - Serialized Library에 Reference Image Library 할당

3. **매니저 추가**
   ```
   Create Empty "Managers"
   └── Add: ARSessionManager, ARInfoOverlay, APIManager
   ```

4. **UI Canvas 생성**
   - GameObject → UI → Canvas
   - Canvas: Screen Space - Overlay
   - 자식으로 ScanningUI, InfoPanel 추가

5. **프리팹 연결**
   - ARInfoOverlay에 InfoPanel, PersonTag 프리팹 할당

### 7.2 VR_Gallery 씬 구성 단계

1. **XR Origin 설정**
   - GameObject → XR → XR Origin (Action-based)
   - 기본 구성 확인 (Camera, Controllers)

2. **Locomotion 설정**
   - GameObject → XR → Locomotion System (Action-based)
   - Teleportation Provider 추가

3. **갤러리 공간 생성**
   ```
   Create Empty "Gallery Room"
   ├── Floor (Plane, Teleportation Area)
   ├── Walls (4개 Cube)
   ├── Ceiling (Plane)
   └── Exhibits (Empty)
   ```

4. **매니저 추가**
   ```
   Create Empty "Managers"
   └── Add: VRSessionManager, GalleryManager, GuidedTourManager, AudioManager, APIManager
   ```

5. **조명**
   - Directional Light 설정
   - 갤러리 내 Point Light 추가

---

## 8. 빌드 설정

### 8.1 Android (AR/VR Quest)

1. File → Build Settings
2. Platform: Android → Switch Platform
3. Add Open Scenes (AR_Main 또는 VR_Gallery)
4. Player Settings 확인
5. Build / Build And Run

### 8.2 iOS (AR)

1. File → Build Settings
2. Platform: iOS → Switch Platform
3. Build → Xcode 프로젝트 생성
4. Xcode에서 Signing 설정 후 빌드

### 8.3 Windows (PC VR)

1. File → Build Settings
2. Platform: PC, Mac & Linux Standalone
3. Target Platform: Windows
4. Architecture: x86_64
5. Build

### 8.4 빌드 자동화 스크립트

```csharp
// Editor/BuildScript.cs
using UnityEditor;
using UnityEditor.Build.Reporting;

public class BuildScript
{
    [MenuItem("Build/Android AR")]
    public static void BuildAndroidAR()
    {
        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { "Assets/Scenes/AR_Main.unity" },
            locationPathName = "Builds/Android/DiasporaAR.apk",
            target = BuildTarget.Android,
            options = BuildOptions.None
        };
        
        BuildPipeline.BuildPlayer(options);
    }
    
    [MenuItem("Build/Quest VR")]
    public static void BuildQuestVR()
    {
        // Quest용 설정
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        
        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { "Assets/Scenes/VR_Gallery.unity" },
            locationPathName = "Builds/Quest/DiasporaVR.apk",
            target = BuildTarget.Android,
            options = BuildOptions.None
        };
        
        BuildPipeline.BuildPlayer(options);
    }
}
```

---

## 9. 테스트 가이드

### 9.1 에디터 테스트

**AR 테스트 (XR Simulation):**
1. Window → Analysis → XR Interaction Debugger
2. XR Simulation을 이용한 테스트

**VR 테스트:**
1. Oculus Link 또는 Virtual Desktop 연결
2. Play 모드에서 VR 헤드셋으로 테스트

### 9.2 디바이스 테스트

**Android (AR):**
```bash
# ADB로 APK 설치
adb install -r DiasporaAR.apk

# 로그 확인
adb logcat -s Unity
```

**Quest (VR):**
```bash
# Quest 개발자 모드 활성화 필요
# Meta Quest Developer Hub 사용 권장
adb install -r DiasporaVR.apk
```

### 9.3 API 서버 연결 테스트

```csharp
// 에디터에서 테스트용 스크립트
[MenuItem("Debug/Test API Connection")]
public static void TestAPI()
{
    var manager = FindObjectOfType<APIManager>();
    manager.StartCoroutine(manager.CheckConnection((bool connected) =>
    {
        Debug.Log($"API 연결 상태: {connected}");
    }));
}
```

### 9.4 체크리스트

**AR 빌드 전:**
- [ ] ARCore/ARKit 지원 기기인가?
- [ ] Camera 권한 설정되었는가?
- [ ] Reference Image Library 설정되었는가?
- [ ] API Base URL이 올바른가?

**VR 빌드 전:**
- [ ] OpenXR 설정이 올바른가?
- [ ] Controller 바인딩이 설정되었는가?
- [ ] Teleportation Area가 설정되었는가?
- [ ] API Base URL이 올바른가?

---

## 부록: 문제 해결

### A. AR 인식 안됨

```
1. Reference Image Library 확인
2. 이미지 품질 확인 (고대비, 선명한 이미지)
3. AR Tracked Image Manager 활성화 확인
4. 조명 환경 확인
```

### B. VR 컨트롤러 인식 안됨

```
1. XR Plugin Management 설정 확인
2. OpenXR Interaction Profile 확인
3. Input System 패키지 설치 확인
4. XR Controller 컴포넌트 설정 확인
```

### C. API 연결 오류

```
1. 서버 실행 상태 확인
2. Base URL 확인 (http:// 포함)
3. CORS 설정 확인
4. 방화벽/네트워크 확인
```

### D. Quest 빌드 오류

```
1. Android SDK/NDK 버전 확인
2. Minimum API Level 29 이상
3. ARM64만 선택
4. IL2CPP Scripting Backend
```
