# 모델 비교 실험 계획서

> 목적: "맥락을 더 잘 이해하는 모델"을 단계적으로 검증
> 기준 성능: B04 (klue/roberta-base + Mean Pooling, Val F1 0.923, 일반대화 Precision 96.4%)
> 실험 환경: DLthon-model-exp/ (원본 리포 클론)

---

## 실험 원칙

- **한 번에 변수 하나만 변경** → 효과 귀인 명확화
- 동일 데이터(`train_final.csv`, 15,000건)로 모든 실험 수행
- 동일 시드(42), 동일 전처리, 동일 Val split

---

## 1단계: 모델 백본 비교

> 질문: "어떤 사전학습 모델이 위협/일반 대화의 맥락 차이를 가장 잘 구분하나?"

### 고정 조건

| 항목 | 설정 |
|---|---|
| 데이터 | train_final.csv (15,000건) |
| 풀링 | Mean Pooling |
| 분류 헤드 | FC(hidden→256) → GELU → Dropout(0.3) → FC(256→5) |
| 학습률 | 백본 4e-6 / 헤드 2e-5 (차등) |
| MAX_LEN | 256 |
| BATCH_SIZE | 8 |
| EPOCHS | 5 + EarlyStopping(patience=2) |
| 시드 | 42 |

### 비교 변수

| 실험 ID | 모델 | HuggingFace ID | 사전학습 특성 |
|---|---|---|---|
| M1-A | **KLUE-RoBERTa-base** | `klue/roberta-base` | MLM + Dynamic Masking. 문맥 파악 강점. B04 기준 |
| M1-B | **KcELECTRA-base** | `beomi/KcELECTRA-base` | RTD. 댓글/구어체 특화. B01~B03 사용 |
| M1-C | **KR-ELECTRA** | `snunlp/KR-ELECTRA-discriminator` | RTD. 다양한 한국어 웹 코퍼스 |

### 비교 지표

| 지표 | 의미 |
|---|---|
| Val F1 Macro | 전체 분류 성능 |
| Val 일반대화 F1 | 합성 데이터 학습 효과 |
| Test 일반대화 예측 수 | 실제 일반대화 탐지력 (100건 기대) |
| Test 예측 분포 균등성 | 5클래스 × 100건에 가까운지 |

### 판단 기준

- Val F1 가장 높은 모델 선정
- Val F1 차이 < 0.01이면, Test 일반대화 예측 수가 더 많은 모델 선정

### 1단계 결과 기록 템플릿

| 실험 | 모델 | Val F1 | 일반대화 Val F1 | Test 일반대화 | 학습시간 | 비고 |
|---|---|---|---|---|---|---|
| M1-A | klue/roberta-base | | | | | B04 기준 |
| M1-B | KcELECTRA-base | | | | | |
| M1-C | KR-ELECTRA | | | | | |

---

## 2단계: 학습 전략 비교

> 질문: "1단계에서 선정된 모델에서, 어떤 학습 설정이 최적인가?"
> 전제: 1단계 최적 모델 고정

### 비교 변수

| 실험 ID | 변경 변수 | 조건 | 가설 |
|---|---|---|---|
| M2-A | 학습률 전략 | 단일 2e-5 (전체 동일) | 차등 학습률이 정말 필요한가? |
| M2-B | 학습률 전략 | **차등 4e-6/2e-5** (B04 설정) | 기준선 |
| M2-C | 학습률 전략 | 차등 2e-6/2e-5 (백본 더 보수적) | Catastrophic Forgetting 추가 방지 |
| M2-D | Label Smoothing | 0.0 (없음) | Label Smoothing이 정말 효과 있나? |
| M2-E | Label Smoothing | **0.1** (B04 설정) | 기준선 |
| M2-F | 풀링 | **Mean Pooling** (B04 설정) | 기준선 |
| M2-G | 풀링 | [CLS] Pooling | Mean vs [CLS] 순수 비교 |

### 실험 순서

```
M2-A vs M2-B vs M2-C  → 최적 학습률 확정
M2-D vs M2-E          → Label Smoothing 효과 확인
M2-F vs M2-G          → 풀링 효과 확인 (Phase 1 재검증)
```

### 2단계 결과 기록 템플릿

| 실험 | 변경점 | Val F1 | Test 일반대화 | vs B04 | 비고 |
|---|---|---|---|---|---|
| M2-A | 단일 lr 2e-5 | | | | |
| M2-B | 차등 4e-6/2e-5 | | | | B04 기준 |
| M2-C | 차등 2e-6/2e-5 | | | | |
| M2-D | LS=0.0 | | | | |
| M2-E | LS=0.1 | | | | B04 기준 |
| M2-F | Mean Pooling | | | | B04 기준 |
| M2-G | [CLS] Pooling | | | | |

---

## 3단계: 인코딩 구조 비교

> 질문: "구조 개선으로 추가 향상 가능한가?"
> 전제: 1단계 최적 모델 + 2단계 최적 설정 고정

### 비교 변수

| 실험 ID | 구조 | 설명 | 타겟 4축 |
|---|---|---|---|
| M3-A | **Mean Pooling** (기준) | 1+2단계에서 확정된 설정 | — |
| M3-B | **Ending Pooling** | 마지막 64토큰(~3턴)만 풀링 | 에스칼레이션, 마무리 톤 |
| M3-C | **Speaker-aware** | 홀수턴/짝수턴 분리 인코딩 → 비대칭 감지 | 발화 대칭성, 존댓말 비대칭 |
| M3-D | **Multi-head** | 대등성 + 마무리톤 + 위협의도 별도 예측 → 결합 | 종합 |

### 구조별 구현 복잡도

| 구조 | 코드 변경량 | 학습 시간 영향 | 과적합 위험 |
|---|---|---|---|
| Ending Pooling | ~10줄 | 동일 | 낮음 |
| Speaker-aware | ~50줄 | +20% | 중간 |
| Multi-head | ~100줄 | +30% | 높음 |

### 실험 순서 (간단한 것부터)

```
M3-A (기준) → M3-B (Ending) → M3-C (Speaker) → M3-D (Multi-head)
                  │                   │                  │
                  └── 효과 없으면 ──→  시도            시도
                      효과 있으면 ──→  M3-B + M3-C 결합 시도
```

### 3단계 결과 기록 템플릿

| 실험 | 구조 | Val F1 | Test 일반대화 | vs 기준 | 비고 |
|---|---|---|---|---|---|
| M3-A | Mean Pooling | | | | 기준 |
| M3-B | Ending Pooling | | | | |
| M3-C | Speaker-aware | | | | |
| M3-D | Multi-head | | | | |

---

## 최종 선정 로직

```
1단계 결과: [최적 백본] 확정
    ↓
2단계 결과: [최적 학습률 + LS + 풀링] 확정
    ↓
3단계 결과: [최적 인코딩 구조] 확정
    ↓
최종 모델 = [1단계 백본] + [2단계 설정] + [3단계 구조]
```

### 발표 시 활용

| 슬라이드 | 내용 |
|---|---|
| "왜 이 모델인가" | 1단계 3종 비교 결과 (백본 선정 근거) |
| "왜 이 설정인가" | 2단계 학습 전략 비교 (차등 학습률/LS 효과) |
| "구조 개선 시도" | 3단계 인코딩 실험 (시도와 결론) |

각 단계에서 "가설 → 실험 → 결과 → 판단"이 명확하므로 Ablation Study 평가 항목에 직접 대응.

---

## 파일 구조

```
DLthon-model-exp/
├── docs/
│   └── model_comparison_plan.md    ← 이 문서
├── notebooks/
│   ├── model.ipynb                 # 원본 (B04 기준)
│   └── model_comparison.ipynb      # 🆕 비교 실험 노트북
├── data/
│   └── train_final.csv             # 동일 데이터
└── results/
    └── model_comparison.csv        # 전 단계 결과 기록
```
