# Vision Agent

LFM2.5-VL 기반 화면 인식 AI 에이전트

## 설치

```bash
cd vision-agent
pip install -e .
```

## 사용법

```bash
# 화면 설명
opencode describe

# UI 요소 찾기  
opencode find "Submit button"

# 에러 분석
opencode error

# OCR
opencode ocr

# 인터랙티브 모드
opencode interactive
```

## Python API

```python
from opencode import create_agent

agent = create_agent()
print(agent.describe())  # 화면 설명
print(agent.find("Login"))  # 요소 찾기
print(agent.ocr())  # 텍스트 추출
```
