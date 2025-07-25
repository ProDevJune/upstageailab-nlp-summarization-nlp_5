# Dialogue Summarization Project - LYJ Branch

## 프로젝트 개요

이 프로젝트는 **AI 부트캠프 13기 NLP Advanced** 과정의 일상 대화 요약 대회를 위한 코드입니다. 최소 2명에서 최대 7명이 참여하는 대화를 자동으로 요약하는 모델을 개발합니다.

## 📁 프로젝트 구조

```
nlp-sum-lyj/
├── code/                           # 소스 코드
│   ├── baseline.ipynb             # BART 기반 베이스라인
│   ├── solar_api.ipynb            # Solar API 활용 코드
│   ├── config.yaml                # 설정 파일
│   └── requirements.txt           # 필요 패키지
├── data/                          # 데이터셋
│   ├── train.csv                  # 학습 데이터 (12,457개)
│   ├── dev.csv                    # 검증 데이터 (499개)
│   ├── test.csv                   # 테스트 데이터 (250개)
│   └── sample_submission.csv      # 제출 양식
└── docs/                          # 문서
    ├── competition_overview.md     # 대회 개요
    ├── baseline_code_analysis.md   # 베이스라인 상세 분석
    ├── solar_api_analysis.md       # Solar API 상세 분석
    ├── rouge_metrics_detail.md     # ROUGE 평가 지표 설명
    ├── project_structure_analysis.md # 프로젝트 구조 분석
    └── uv_package_manager_guide.md # uv 패키지 관리자 가이드
```

## 🚀 빠른 시작

### 1. 환경 설정

#### 방법 1: 기존 방식 (venv + pip)
```bash
# 가상환경 생성
python -m venv dialogue_sum_env
source dialogue_sum_env/bin/activate  # Windows: .\dialogue_sum_env\Scripts\activate

# 패키지 설치
pip install -r code/requirements.txt
```

#### 방법 2: uv 사용 (권장 - 10배 이상 빠름!)
```bash
# uv 설치 (처음 한 번만)
pip install uv

# 가상환경 생성 (0.1초!)
uv venv dialogue_sum_env
source dialogue_sum_env/bin/activate  # Windows: .\dialogue_sum_env\Scripts\activate

# 패키지 설치 (매우 빠름!)
uv pip install -r code/requirements.txt

# (선택) Lock 파일 생성으로 정확한 버전 관리
uv pip compile code/requirements.txt -o code/requirements.lock
```

> 💡 **uv를 사용하면**: 환경 설정 시간이 90초에서 7초로 단축됩니다!
> 자세한 내용은 [uv 패키지 관리자 가이드](docs/uv_package_manager_guide.md)를 참고하세요.

### 2. 데이터 확인

```python
import pandas as pd

# 데이터 로드
train_df = pd.read_csv('data/train.csv')
print(f"학습 데이터: {len(train_df)}개")
print(train_df.head())
```

### 3. 모델 학습 (Baseline)

```python
# config 파일 수정
config['general']['data_path'] = "./data/"
config['wandb']['entity'] = "your_wandb_account"

# 학습 실행
python -m baseline
```

### 4. Solar API 사용

```python
# API 키 설정
UPSTAGE_API_KEY = "your_api_key"

# 요약 실행
output = inference()
```

## 📊 접근 방법

### 1. BART 기반 Fine-tuning (baseline.ipynb)
- **모델**: KoBART (한국어 특화 BART)
- **장점**: 높은 성능, 커스터마이징 가능
- **단점**: GPU 필요, 학습 시간 소요
- **성능**: ROUGE-F1 47.12 (Public)

### 2. Solar API 활용 (solar_api.ipynb)
- **모델**: solar-1-mini-chat
- **장점**: 즉시 사용, GPU 불필요
- **단점**: API 비용, Rate Limit
- **최적화**: Few-shot 프롬프트 엔지니어링

## 📈 평가 지표

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram 기반 비교
- **ROUGE-2**: Bigram 기반 비교
- **ROUGE-L**: 최장 공통 부분 수열

### 최종 점수 계산
```
Final Score = max ROUGE-1-F1(pred, gold_i)
            + max ROUGE-2-F1(pred, gold_i)
            + max ROUGE-L-F1(pred, gold_i)
```

## 🔧 주요 설정 (config.yaml)

### 학습 설정
```yaml
training:
  num_train_epochs: 20
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  fp16: true  # Mixed precision
  early_stopping_patience: 3
```

### 추론 설정
```yaml
inference:
  batch_size: 32
  num_beams: 4  # Beam search
  no_repeat_ngram_size: 2
  generate_max_length: 100
```

## 💡 성능 향상 팁

### 1. 하이퍼파라미터 튜닝
- Learning rate: 3e-5, 5e-5 실험
- Batch size: 메모리 허용 범위 내 최대
- Beam size: 4-8 범위 실험

### 2. 데이터 전처리
- 특수 토큰 추가 (#Person1#, #Person2# 등)
- 노이즈 제거 (HTML 태그, 이스케이프 문자)

### 3. 프롬프트 엔지니어링 (Solar API)
- Few-shot 예시 활용
- 명확한 지시문 작성
- Temperature/Top-p 조정

### 4. 앙상블
- 다양한 모델 결과 조합
- 투표 또는 가중 평균

## 📝 제출 규칙

- **일일 제출 횟수**: 팀당 12회
- **최종 제출물**: 최대 2개 선택
- **평가 데이터**: Public 50%, Private 50%

## ⚠️ 주의사항

1. **외부 데이터셋**: DialogSum 사용 금지
2. **평가 데이터**: 학습에 사용 금지
3. **API 사용**: 무료 API만 허용 (Solar는 예외)
4. **파일 형식**: CSV (fname, summary 컬럼)

## 📚 참고 문서

- [대회 개요](docs/competition_overview.md)
- [베이스라인 코드 상세 분석](docs/baseline_code_analysis.md)
- [Solar API 상세 분석](docs/solar_api_analysis.md)
- [ROUGE 평가 지표 설명](docs/rouge_metrics_detail.md)
- [프로젝트 구조 분석](docs/project_structure_analysis.md)
- [uv 패키지 관리자 가이드](docs/uv_package_manager_guide.md) 🆕

## 🛠️ 트러블슈팅

### CUDA Out of Memory
```python
# 배치 크기 감소
config['training']['per_device_train_batch_size'] = 32

# Gradient Accumulation 사용
config['training']['gradient_accumulation_steps'] = 2
```

### Rate Limit (Solar API)
```python
# 요청 간격 조정
if (idx + 1) % 100 == 0:
    time.sleep(65)  # 1분 대기
```

### 토큰화 오류
```python
# 특수 토큰 확인
print(tokenizer.special_tokens_map)

# 최대 길이 조정
config['tokenizer']['encoder_max_len'] = 1024
```

## 🎯 개발 로드맵

- [x] 베이스라인 구현
- [x] Solar API 연동
- [ ] 데이터 증강
- [ ] 모델 앙상블
- [ ] 하이퍼파라미터 최적화
- [ ] 추가 모델 실험 (T5, GPT)

## 📞 연락처

문제가 있거나 질문이 있으시면 이슈를 등록해주세요.

---

**Last Updated**: 2025.01.14
**Author**: LYJ
**Branch**: lyj
