# 🚨 DLThon: Threat Dialogue Classification Project

본 프로젝트는 대화의 성격을 **4가지 위협 세부 클래스(협박, 갈취, 직장 내 괴롭힘, 기타 괴롭힘)** 또는 **일반 대화**로 정밀 분류하는 자연어 처리(NLP) 과제입니다. 데이터 중심(Data-Centric) 접근법을 통해 모델의 성능을 극대화하고, 체계적인 실험(Ablation Study)을 통해 최적의 전략을 도출하는 것을 목표로 합니다.

---

## 📂 Repository Structure

```text
DLthon/
├── data/               # 데이터셋 (Raw, Processed, Synthetic)
│   ├── train.csv       # 원본 학습 데이터 (4,000건 미만)
│   └── submission.csv  # 추론 결과 제출 템플릿
├── docs/               # 프로젝트 가이드라인 및 전략 문서
│   ├── DLThon.md       # 대회 규칙 및 평가 지표 정의
│   ├── eda.md          # 6개 카테고리, 13개 세부 분석 지표 설계안
│   ├── strategy.md     # EDA 기반 전처리·증강·합성 데이터 생성 전략 보고서
│   └── implementation_plan.md # 프로젝트 로드맵 및 구조 계획
├── notebooks/          # 분석 및 실험용 Interactive Notebook
│   └── train_eda.ipynb # 13개 지표 시각화 및 통계 분석 메인 노트북
├── reports/            # 분석 결과물 및 검증 리포트
│   ├── eda_results.txt # 텍스트 기반 상세 EDA 수치 결과
│   └── eda_code_check.md # 코드-문서 간 정합성 최종 검증 보고서
├── src/                # 모듈화된 소스 코드 및 유틸리티
│   └── eda_runner.py   # 헤드리스 환경용 EDA 분석 실행 스크립트
└── README.md           # 프로젝트 전체 개요 (현재 파일)

### 🚀 Current Progress (Project Status)

1. **데이터 분석 (EDA) 완료 ✅**
   - 6대 범주 분석: 길이 및 구조, 어휘 및 키워드, 클래스 관계, 언어적 특징, 품질 및 노이즈, 상황 및 맥락.
   - 13개 지표 도출: TF-IDF 키워드, N-gram 패턴, TTR(어휘 다양성), 품사(POS) 분포, 변동계수(CV) 기반 조사 편향성 등 정밀 분석 완료.
   - 검증 완료: `eda_code_check.md`를 통해 분석 결과가 프로젝트 요구사항 및 실제 코드와 일치함을 확인.

2. **프로젝트 리포트 작성 ✅**
   - 데이터 전략 수립: 분석된 수치(중복 104건, 특정 조사 편향 등)를 바탕으로 구체적인 Action Item 도출.
   - 정의서 완비: 대회 룰(`DLThon.md`), 분석 계획(`eda.md`), 실행 전략(`strategy.md`) 문서화 완료.

3. **환경 구축 완료 ✅**
   - Git 기반 버전 관리 및 GitHub 원격 저장소 연동.
   - 연구 표준에 부합하는 폴더 구조 재배치 완료.

### 🛠️ Data-Centric Strategy (Key Highlights)

현재 수립된 `strategy.md`의 핵심 전략은 다음과 같습니다:

- **전처리 (Preprocessing)**: 준-중복(유사도 0.95 이상) 데이터 117건 제거 및 편향된 조사 처리.
- **증강 (Augmentation)**: 위협 4클래스에 대해 핵심 단어(예: "당장", "제발", "돈")를 보존하며 동의어 교체 및 역번역(BT) 수행 (목표 배율 x1.3).
- **합성 (Synthesis)**:
   - 일반 대화 1,000건을 LLM으로 생성.
   - **Hard Negative 전략**: 위협 클래스의 빈출 단어("돈", "진짜")가 포함되지만 무해한 문맥(Safe Context)의 대화 생성.
   - **도메인 쏠림 방지**: 직장·학교 일상 대화를 고루 분배하여 모델의 도메인 편향(예: 직장=무조건 괴롭힘) 제거.

### 📊 Evaluation & Metrics

- **Primary Metric**: Macro F1 Score
- **Evaluation Loop**: 6단계의 **Ablation Study**를 통해 각 전략(중복 제거, 증강 배율, 합성량 등)의 기여도를 독립적으로 측정합니다.

### 🔗 Getting Started

- **Prerequisites**: Python 3.8+, Konlpy(Okt), Scikit-learn, Pandas, Matplotlib, Seaborn.
- **Run Analysis**:
  ```bash
  python src/eda_runner.py
