# 🛡️ DLthon: 대화형 위협 탐지 및 분류 프로젝트

본 프로젝트는 대화 데이터를 분석하여 **협박, 갈취, 직장 내 괴롭힘, 기타 괴롭힘** 및 **일반 대화**의 5개 클래스를 정밀 분류하는 자연어 처리(NLP) 모델을 개발하는 것을 목표로 합니다.

---

### 📂 Repository Structure

DLthon/
├── data/               # 데이터셋 (Raw, Processed, Synthetic)
│   ├── train.csv       # 원본 학습 데이터 (4,000건 미만)
│   ├── baseline.csv    # 증강(x1.3) 및 합성데이터(1k)가 포함된 최종 학습셋
│   └── submission.csv  # 추론 결과 제출 템플릿
├── docs/               # 프로젝트 가이드라인 및 전략 문서
│   ├── DLThon.md       # 대회 규칙 및 평가 지표 정의
│   ├── eda.md          # 6개 카테고리, 13개 세부 분석 지표 설계안
│   ├── strategy.md     # EDA 기반 전처리·증강·합성 데이터 생성 전략 보고서
│   ├── model_plan.md   # [NEW] 베이스라인 모델 아키텍처 및 학습 계획서
│   └── implementation_plan.md # 프로젝트 로드맵 및 구조 계획
├── notebooks/          # 분석 및 실험용 Interactive Notebook
│   ├── train_eda.ipynb # 13개 지표 시각화 및 통계 분석 메인 노트북
│   └── model.ipynb     # [NEW] 베이스라인 모델 학습, 평가 및 추론 노트북
├── reports/            # 분석 결과물 및 검증 리포트
│   ├── eda_results.txt # 텍스트 기반 상세 EDA 수치 결과
│   └── eda_code_check.md # 코드-문서 간 정합성 최종 검증 보고서
├── src/                # 모듈화된 소스 코드 및 유틸리티
│   └── eda_runner.py   # 헤드리스 환경용 EDA 분석 실행 스크립트
└── README.md           # 프로젝트 전체 개요 (현재 파일)

---

### 🚀 Current Progress (Project Status)

1. **데이터 분석 (EDA) 완료 ✅**
   - 6대 범주(구조, 어휘, 의미, 언어, 품질, 맥락) 13개 지표 정밀 분석.
   - `eda_results.txt`를 통해 클래스별 말투, 도메인 쏠림, 중복 데이터 등의 정량 수치 확보.

2. **데이터 중심(Data-Centric) 전략 수립 ✅**
   - **전처리**: 중복 제거 및 `\n` → `[SEP]` 치환 전략 수립.
   - **증강/합성**: 위협 4클래스 x1.3 증강 및 일반 대화 1,000건 합성 가이드라인 완비.
   - **Hard Negative**: 유사도가 높은 클래스 쌍(협박↔기타 괴롭힘)의 변별력을 높이기 위한 Safe-Context 합성 전략 적용.

3. **모델링 및 학습 파이프라인 구축 ✅**
   - **선정 모델**: `klue/roberta-base` (한국어 구어체 및 문맥 파악 최적화).
   - **파이프라인**: 전처리 → Dataset → Model → Training → Inference 통합 노트북(`model.ipynb`) 완성.
   - **학습 설정**: Macro F1 Score 극대화를 위한 Dropout(0.3) 및 Learning Rate(2e-5) 적용.

4. **실험 설계 및 로드맵 구축 ✅**
   - `model_plan.md`를 통해 6가지 핵심 변수(MAX_LEN, 손실함수, 모델 아키텍처 등)에 대한 **Ablation Study** 계획 수립.
   - 연구 표준에 부합하는 폴더 구조 재배치 및 문서화 완료.

---

### 🛠️ Data-Centric Strategy (Key Highlights)

- **Preprocessing**: 준-중복(유사도 0.95 이상) 데이터 117건 제거 및 편향된 조사 처리.
- **Augmentation**: 핵심 키워드(당장, 제발, 돈 등)를 보존하는 동의어 교체 및 역번역.
- **Synthesis**: 
    - **Hard Negative 전략**: 위협 빈출 단어를 포함하되 무해한 문맥의 일반 대화 생성.
    - **Domain Balancing**: 직장·학교 등 특정 장소의 키워드가 특정 위협으로 쏠리지 않도록 일상 대화 분배.

---

### 📊 Evaluation & Metrics

- **Primary Metric**: Macro F1 Score
- **Ablation Study**: 각 전략이 성능에 미치는 영향을 독립적으로 측정하여 논리적 개선 근거를 확보합니다.

---

### 🔗 Getting Started

- **Prerequisites**: Python 3.8+, PyTorch, Transformers, Scikit-learn, Konlpy.
- **Run Analysis**:
  ```bash
  python src/eda_runner.py

