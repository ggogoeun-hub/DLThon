# DKTC DLthon — 한국어 위협 대화 분류

아이펠 온라인 17기 DLthon | Dataset of Korean Threatening Conversations

## 과제 개요

한국어 대화를 5개 클래스로 분류하는 NLU 과제. Train에 없는 '일반 대화' 클래스를 합성 데이터로 생성해야 하는 것이 핵심 도전.

| 클래스 | Train | Test |
|---|---|---|
| 협박 대화 | 896건 | 100건 |
| 갈취 대화 | 981건 | 100건 |
| 직장 내 괴롭힘 대화 | 979건 | 100건 |
| 기타 괴롭힘 대화 | 1,094건 | 100건 |
| 일반 대화 | **0건 (없음)** | 100건 |

평가지표: F1 Score (Macro)

## Repository Structure

```
DLthon/
│
├── aiffel-d-lthon-dktc-online-17/        # 원본 데이터
│   ├── train.csv                          #   학습 데이터 (3,950건, 4클래스)
│   ├── test.csv                           #   테스트 데이터 (500건, 5클래스)
│   └── submission.csv                     #   제출 템플릿
│
├── notebooks/                             # Interactive Notebooks
│   ├── 01_EDA.ipynb                       #   EDA 시각화 (클래스 분포, 키워드, 포맷 분석)
│   ├── 02_synthetic_data_design.ipynb     #   합성 데이터 설계 (4축 구조 분석)
│   └── 03_training.ipynb                  #   모델 학습 및 추론 파이프라인
│
├── docs/                                  # 가이드라인 및 전략 문서
│   ├── strategy_v1.md                     #   데이터 전략 보고서 (전처리·증강·합성·모델링)
│   ├── review_seongyeon_DLThon.md         #   코드 리뷰
│   └── synthetic/                         #   합성 데이터 설계 (상세)
│       ├── README.md                      #     개요 & 4축 프레임워크
│       ├── profile_threat.md              #     위협 대화 구조 프로파일 (4축 수치)
│       ├── generation_guide_v1.md         #     생성 가이드라인 (프롬프트 설계)
│       └── validation.md                  #     품질 검증 기준 & 자동 검증 코드
│
├── figures/                               # EDA & 분석 시각화
│   ├── 01_class_distribution.png          #   클래스 분포
│   ├── 02_length_distribution.png         #   대화 길이 분포
│   ├── 03_newline_mismatch.png            #   Train/Test 포맷 불일치
│   ├── 04_tfidf_keywords.png              #   클래스별 TF-IDF 키워드
│   └── 05_wordclouds.png                  #   WordCloud
│
├── logs/                                  # 보고서 & 실험일지
│   ├── 실험일지_노션붙여넣기용.md             #   팀 공유용 실험일지 (Ablation Study)
│   └── *.html                             #   진행 보고서, 종합 분석 등
│
├── outputs/                               # 실험 결과
│   ├── ablation_study.csv                 #   Ablation Study 기록
│   └── submission_*.csv                   #   Kaggle 제출 파일
│
├── baseline.csv                           # 증강 포함 학습셋 (15,000건, 5×3,000)
├── hard_negative_normal.csv               # Hard Negative 합성 데이터 (200건)
└── synthetic_normal_combined.csv          # 합성 일반 대화 통합본 (1,200건)
```

## 핵심 발견

1. **Train/Test 포맷 불일치**: Train 100% `\n` 구분 vs Test 4.8%만 `\n` → 전처리 정규화 필수
2. **합성 데이터 도메인 갭**: 양/길이 확대로는 해결 안 됨 → 구조적 특성(권력 관계, 감정 흐름) 매칭 필요
3. **4축 분석 프레임워크**: 발화 대칭성, 존댓말 비대칭, 에스컬레이션, 굴복/대등 비율로 위협-일반 차이 정량화

## Data-Centric Strategy

- **전처리**: 중복 제거(104건) + `\n` → 공백 정규화
- **증강**: 핵심 키워드 보존 동의어 교체 + 패러프레이즈 (문장 셔플은 에스컬레이션 패턴 파괴로 제외)
- **합성**: 4축 구조 분석 기반 일반 대화 생성 (docs/synthetic/ 참고)
- **Hard Negative**: 위협 빈출 키워드를 무해한 문맥에 배치

## Ablation Study

| ID | 변경 변수 | Val F1 | Test 일반대화 |
|---|---|---|---|
| B01 | Baseline (KcELECTRA + 합성 694건) | 0.922 | 34건 |
| S01 | 합성 데이터 길이 개선 (avg 206자) | 0.922 | 35건 |
| S02 | 합성 데이터 양 확대 (1,200건) | 0.921 | 38건 |
| B02 | 대규모 증강 (15,000건 균형) | 0.990 | 23건 |

## 환경

- Python 3.9.6 / PyTorch 2.8.0 / Transformers 4.57
- 모델: beomi/KcELECTRA-base (109M params)
- 디바이스: Apple M5 CPU (MPS 속도 이슈로 CPU 사용)
