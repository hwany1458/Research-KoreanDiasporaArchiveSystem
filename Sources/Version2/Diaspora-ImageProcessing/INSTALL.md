# 설치 트러블슈팅 가이드

본 문서는 2026-04-18 실제 설치 과정에서 발생한 문제와 해결책을 기록합니다.

---

## 핵심 원칙

이 프로젝트는 여러 패키지가 **서로 다른 버전의 numpy, basicsr, torchvision**을 요구하여
의존성 충돌이 자주 발생합니다. 아래 순서와 방법을 반드시 지켜야 합니다.

---

## 올바른 설치 순서

```
1. torch + torchvision (CUDA 버전 지정)
2. numpy 고정 (1.26.4)
3. basicsr + realesrgan (--no-deps)
4. gfpgan
5. DDColor (git clone 방식)
6. numpy + basicsr + realesrgan 재고정
7. transformers + tokenizers + huggingface-hub (버전 지정, --no-deps)
8. face_recognition + 기타
9. 모델 다운로드
```

---

## 문제 1: torchvision.transforms.functional_tensor 없음

**증상**:
```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

**원인**: torchvision 0.17 이상에서 해당 모듈 삭제됨. basicsr 1.4.2가 이 모듈을 사용.

**해결**:
```powershell
pip uninstall torchvision -y
pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 문제 2: numpy 버전 충돌 (가장 빈번)

**증상**:
```
RuntimeError: Numpy is not available
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**원인**: basicsr, DDColor, torchvision 등이 설치 시 numpy를 2.x로 올려버림.

**해결**:
```powershell
pip install "numpy==1.26.4" --force-reinstall
pip install basicsr==1.4.2 --no-deps
pip install realesrgan==0.3.0 --no-deps
```

> ⚠️ 이 문제는 다른 패키지 설치 후 반복 발생합니다. 의심될 때마다 실행하세요.

**확인**:
```powershell
python -c "import numpy; print(numpy.__version__)"  # 1.26.4여야 함
```

---

## 문제 3: basicsr import 실패

**증상**:
```
ImportError: Real-ESRGAN not installed. Run: pip install realesrgan basicsr
```
(설치는 되어있는데 import 실패)

**원인**: numpy 버전 충돌 또는 basicsr 버전 문제.

**해결**:
```powershell
pip install "numpy==1.26.4" --force-reinstall
pip install basicsr==1.4.2 --no-deps
pip install realesrgan==0.3.0 --no-deps

# 캐시 삭제
Remove-Item -Recurse -Force src\__pycache__
Remove-Item -Recurse -Force src\modules\__pycache__
```

---

## 문제 4: DDColor 설치 실패 (ModuleNotFoundError: torch)

**증상**:
```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build ddcolor
```

**원인**: pip이 격리된 빌드 환경에서 torch를 못 찾음.

**해결**: `--no-build-isolation` 옵션 사용
```powershell
cd ddcolor_src
pip install -e . --no-build-isolation
cd ..
```

---

## 문제 5: DDColor basicsr 아키텍처 없음

**증상**:
```
No module named 'basicsr.archs.ddcolor_arch_utils'
```

**원인**: DDColor가 사용하는 커스텀 basicsr 아키텍처가 설치된 basicsr에 없음.

**해결**: ddcolor_src에서 파일 복사
```powershell
$src = "ddcolor_src\basicsr\archs"
$dst = "venv\Lib\site-packages\basicsr\archs"
Copy-Item "$src\ddcolor_arch.py" "$dst\ddcolor_arch.py" -Force
Copy-Item "$src\ddcolor_arch_utils" "$dst\ddcolor_arch_utils" -Recurse -Force
Copy-Item "$src\vgg_arch.py" "$dst\vgg_arch.py" -Force
```

---

## 문제 6: BLIP/CLIP torch 버전 오류

**증상**:
```
BlipForConditionalGeneration requires the PyTorch library but it was not found
Disabling PyTorch because PyTorch >= 2.4 is required but found 2.1.2
```

**원인**: transformers 5.x가 torch 2.4+ 를 요구하는데 2.1.2가 설치됨.

**해결**: transformers를 4.37.2로 다운그레이드
```powershell
pip install transformers==4.37.2 tokenizers==0.15.2 "huggingface-hub==0.20.3" --no-deps
```

**확인**:
```powershell
python -c "from transformers import BlipProcessor; print('BLIP OK')"
python -c "from transformers import CLIPModel; print('CLIP OK')"
```

---

## 문제 7: huggingface-hub 버전 충돌

**증상**:
```
ImportError: huggingface-hub>=0.19.3,<1.0 is required but found 1.11.0
```

**원인**: transformers 4.37.2가 huggingface-hub 1.x를 지원하지 않음.

**해결**:
```powershell
pip install "huggingface-hub==0.20.3" --no-deps
```

---

## 문제 8: DDColor HuggingFace 모델 다운로드 실패

**증상**:
```
401 Client Error: Repository Not Found for url: .../piddnad/ddcolor_models/...
```

**원인**: repo ID가 잘못됨. `piddnad/ddcolor_models`가 아니라 `piddnad/ddcolor_modelscope`.

**해결**: 코드에서 올바른 repo ID 사용
```python
model = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope")  # 올바름
# model = DDColorHF.from_pretrained("piddnad/ddcolor_models")   # 틀림
```

---

## 문제 9: 얼굴 감지 실패 (고해상도)

**증상**: 로그에 `감지된 얼굴: 0개` 인데 실제로 사람이 있는 경우

**원인**: Real-ESRGAN으로 4배 업스케일 후 이미지가 너무 커서 (6000px 이상) GFPGAN 내부 RetinaFace가 얼굴을 찾지 못함.

**해결**: `face_enhancement.py`의 `enhance()` 메서드가 자동으로 `max_size=2048`로 리사이즈 후 처리합니다. 별도 조치 불필요.

---

## 문제 10: Python __pycache__ 캐시 문제

**증상**: 파일을 수정했는데 이전 버전이 실행됨.

**해결**:
```powershell
Remove-Item -Recurse -Force src\__pycache__
Remove-Item -Recurse -Force src\modules\__pycache__
```

---

## 설치 완료 후 전체 검증

```powershell
python test_pipeline.py --all
```

모든 항목이 ✓로 나오면 정상입니다.

```
✓ PyTorch 2.1.2+cu121
✓ CUDA 사용 가능
✓ OpenCV
✓ Pillow
✓ Real-ESRGAN
✓ GFPGAN
✓ Transformers (BLIP, CLIP)
✓ face_recognition
```
