# 베이스라인 모델 구현 계획서

> **목적**: `baseline.csv`(위협 4클래스 + 합성 일반 대화)를 사용한 5클래스 분류 모델 구축  
> **참조**: `DLThon.md` (과제 규칙), `strategy.md` (데이터 전략), `eda_results.txt` (EDA 수치)

---

## 0. 프로젝트 전제사항

| 항목 | 내용 |
|---|---|
| 입력 데이터 | `baseline.csv` — 팀원이 증강(4클래스) + 합성(일반 대화)을 완료한 학습 데이터 |
| 데이터 컬럼 | `idx`, `class`, `conversation` |
| 분류 클래스 | `협박 대화`, `갈취 대화`, `직장 내 괴롭힘 대화`, `기타 괴롭힘 대화`, `일반 대화` (**5종**) |
| 평가 지표 | **Macro F1 Score** |
| 추론 대상 | `test.csv` (5클래스 × 100개 = 500건) — `submission.csv` 형식으로 제출 |

---

## 1. 모델 선정

### 1-1. 선정 모델: `klue/roberta-base`

| 비교 항목 | KLUE-BERT-base | **KLUE-RoBERTa-base** (선정) | KLUE-RoBERTa-large |
|---|---|---|---|
| 한국어 사전학습 | ✅ | ✅ | ✅ |
| 구어체/대화 이해 | 보통 | **우수** | 우수 |
| 학습 속도 | 빠름 | 빠름 | **느림** |
| 메모리 사용 | 110M params | 110M params | 330M params |
| Ablation 반복 가능성 | ✅ | ✅ | ❌ (GPU 부담) |

**선정 근거**:
1. RoBERTa는 NSP 제거 + Dynamic Masking으로 BERT 대비 **문맥 파악 능력이 우수**
2. `base` 모델(110M)은 Ablation Study에서 실험을 6회 이상 반복하기에 충분히 가벼움
3. KLUE 벤치마크에서 한국어 NLU 태스크 최상위권 성능

### 1-2. 대안 모델 (Ablation용)
- **`monologg/koelectra-base-v3-discriminator`**: ELECTRA 기반, 효율적 사전학습으로 소규모 데이터에 강점
- 베이스라인 F1이 기대 이하일 경우 비교 실험 대상

---

## 2. 데이터 전처리 파이프라인

### 2-1. 데이터 로드 및 검증

```python
# Cell 1: 데이터 로드
import pandas as pd

train_df = pd.read_csv('data/baseline.csv')  # 팀원이 만든 최종 학습 데이터

# 클래스 분포 확인
print(train_df['class'].value_counts())
print(f"총 데이터: {len(train_df)}건")
```

### 2-2. 레이블 인코딩

```python
# Cell 2: 레이블 매핑
label2id = {
    '갈취 대화': 0,
    '기타 괴롭힘 대화': 1,
    '직장 내 괴롭힘 대화': 2,
    '협박 대화': 3,
    '일반 대화': 4,
}
id2label = {v: k for k, v in label2id.items()}

train_df['label'] = train_df['class'].map(label2id)
```

### 2-3. 전처리 — 최소한의 정규화

> `strategy.md` 1-2에 따라:
> - `!`, `?` → **유지** (클래스 힌트)
> - 이모티콘, `...` → 학습 데이터에 없으므로 별도 처리 불필요
> - 조사 → **제거하지 않음** (옵션 A: BERT 계열이 조사를 문맥 정보로 활용)

```python
# Cell 3: 텍스트 전처리
import re

def preprocess(text):
    """최소한의 전처리 — 모델이 원문 문맥을 최대한 활용하도록"""
    text = re.sub(r'\n+', ' [SEP] ', text)  # 발화 구분을 [SEP]으로 치환
    text = re.sub(r'\s+', ' ', text).strip()  # 연속 공백 정리
    return text

train_df['text'] = train_df['conversation'].apply(preprocess)
```

**발화 구분 전략**: `\n`을 `[SEP]` 토큰으로 치환하면, RoBERTa가 대화의 턴 경계를 인식할 수 있습니다.

### 2-4. Train/Validation 분할

```python
# Cell 4: 데이터 분할
from sklearn.model_selection import StratifiedKFold, train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(),
    train_df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df['label']
)
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
```

---

## 3. 모델 아키텍처

### 3-1. 토큰화

```python
# Cell 5: 토크나이저 설정
from transformers import AutoTokenizer

MODEL_NAME = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 256  # EDA: median=203자, 75%=270자 → 256 토큰이면 대부분 커버
```

**`MAX_LEN = 256` 근거**:
- EDA `[1-1]`: 글자 수 중간값 203자, 75% → 270자
- RoBERTa 토크나이저는 한국어 1글자당 약 1~2 토큰 → 256 토큰으로 충분
- 512보다 메모리 50% 절약 → 배치 사이즈를 키울 수 있음

### 3-2. Dataset 클래스

```python
# Cell 6: PyTorch Dataset
import torch
from torch.utils.data import Dataset, DataLoader

class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

### 3-3. 분류 모델

```python
# Cell 7: 모델 정의
from transformers import AutoModel
import torch.nn as nn

class ConversationClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
```

**아키텍처 설계 결정**:

| 결정 | 선택 | 근거 |
|---|---|---|
| 풀링 방식 | `[CLS]` 토큰 | 대화 전체의 의미를 응축한 벡터. Mean pooling 대비 분류 태스크에 안정적 |
| Dropout | 0.3 | 데이터가 ~5,000건으로 적음. 과적합 방지를 위해 기본(0.1)보다 높게 설정 |
| 분류 헤드 | 단일 Linear | 베이스라인에서는 단순 구조로 시작. 성능 분석 후 MLP 헤드 추가 검토 |

---

## 4. 학습 설정

### 4-1. 하이퍼파라미터

```python
# Cell 8: 학습 설정
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
```

| 파라미터 | 값 | 근거 |
|---|---|---|
| Batch Size | 32 | MAX_LEN=256 기준 GPU 메모리 최적. 16도 가능 (GPU 환경에 따라) |
| Learning Rate | 2e-5 | BERT/RoBERTa fine-tuning 표준 범위(1e-5 ~ 5e-5) |
| Epochs | 5 | 데이터 ~5,000건 → 3~5 에폭에서 수렴 예상. Early Stopping으로 보완 |
| Warmup | 10% | 학습 초기 불안정 방지 |
| Weight Decay | 0.01 | L2 정규화로 과적합 억제 |

### 4-2. 옵티마이저 & 스케줄러

```python
# Cell 9: 옵티마이저
from transformers import get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps,
)
```

### 4-3. 손실 함수

```python
# Cell 10: 손실 함수
criterion = nn.CrossEntropyLoss()

# [선택] 클래스 불균형이 심할 경우 가중치 적용
# EDA 기준 클래스당 ~1,000건이지만, 합성 데이터 품질에 따라 달라질 수 있음
# class_weights = torch.tensor([...], dtype=torch.float).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 5. 학습 루프

### 5-1. 학습 함수

```python
# Cell 11: 학습 루프
from sklearn.metrics import f1_score
import numpy as np

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    preds, trues = [], []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(trues, preds, average='macro')
    return avg_loss, f1
```

### 5-2. 검증 함수

```python
# Cell 12: 검증 루프
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(trues, preds, average='macro')
    return avg_loss, f1, preds, trues
```

### 5-3. 메인 학습 실행

```python
# Cell 13: 메인 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConversationClassifier(MODEL_NAME).to(device)

best_f1 = 0
history = []

for epoch in range(EPOCHS):
    train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    val_loss, val_f1, val_preds, val_trues = evaluate(model, val_loader, criterion, device)

    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_f1': val_f1,
    })

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # Early Stopping / Best Model 저장
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  → Best model saved! (F1: {best_f1:.4f})")
```

---

## 6. 평가 및 분석

### 6-1. 혼동 행렬

```python
# Cell 14: 혼동 행렬 시각화
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 최적 모델 로드
model.load_state_dict(torch.load('best_model.pt'))
_, _, final_preds, final_trues = evaluate(model, val_loader, criterion, device)

# Classification Report
class_names = [id2label[i] for i in range(5)]
print(classification_report(final_trues, final_preds, target_names=class_names))

# 혼동 행렬
cm = confusion_matrix(final_trues, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
```

**혼동 행렬 분석 포인트** (`strategy.md` 기반):
- `협박 ↔ 기타 괴롭힘` 혼동 여부 확인 (코사인 유사도 0.87로 최고)
- `협박 ↔ 갈취` 혼동 여부 확인 (코사인 유사도 0.83)
- `일반 대화`가 위협 클래스로 오분류되는 비율 확인

### 6-2. 학습 곡선

```python
# Cell 15: 학습 곡선 시각화
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hist_df['epoch'], hist_df['train_loss'], label='Train')
axes[0].plot(hist_df['epoch'], hist_df['val_loss'], label='Val')
axes[0].set_title('Loss Curve')
axes[0].legend()

axes[1].plot(hist_df['epoch'], hist_df['train_f1'], label='Train')
axes[1].plot(hist_df['epoch'], hist_df['val_f1'], label='Val')
axes[1].set_title('F1 Score Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 7. 추론 & 제출

```python
# Cell 16: 테스트 데이터 추론 & submission.csv 생성
test_df = pd.read_csv('data/test.csv')
test_df['text'] = test_df['conversation'].apply(preprocess)

test_dataset = ConversationDataset(
    test_df['text'].tolist(),
    [0] * len(test_df),  # 더미 레이블
    tokenizer, MAX_LEN
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

# 레이블 디코딩 & 제출 파일 생성
test_df['class'] = [id2label[p] for p in all_preds]
submission = test_df[['idx', 'class']]
submission.to_csv('submission.csv', index=False)
print("submission.csv 저장 완료!")
print(submission['class'].value_counts())
```

---

## 8. Ablation Study 실험 로그

### 완료된 실험 (B01~B02)

| ID | 변수 | 조건 | Val F1 | Test 일반대화 | 교훈 |
|---|---|---|---|---|---|
| B01 | Baseline | KcELECTRA, 합성 694건, max_len=128 | 0.922 | 34건 | 기준선 |
| S01 | 합성 길이 | 합성 700건 (avg 206자) | 0.922 | 35건 | 길이 개선 효과 없음 |
| S02 | 합성 양 | 합성 1,200건 | 0.921 | 38건 | 양 확대 미미한 효과 |
| S02b | MAX_LEN | 128 → 256 | 0.937 | — | +1.6%p, max_len=256 확정 |
| B02 | 대규모 증강 | 15k 균형, 경량 EDA 증강 | 0.990 | 23건 | 문체 숏컷 과적합 |

### 예정 실험

| ID | 변수 | 조건 | 비고 |
|---|---|---|---|
| **B04** | v2 전략 데이터 | baseline_B04.csv (gem_prompt_v2 기반 15k) | strategy_v2.md |
| Exp-A1 | MAX_LEN | 256 / 384 / 512 | S02b에서 256 효과 확인. 추가 실험 |
| Exp-A2 | Dropout | 0.1 / 0.3 / 0.5 | — |
| Exp-A3 | 전처리 | `\n` → 공백 (확정) vs `[TURN]` special token | v1에서 공백 확정. [TURN] 추가 실험 |
| Exp-A4 | 손실 함수 | CE vs Focal Loss vs Label Smoothing | 협박↔기타괴롭힘 유사도 0.87 |
| Exp-A5 | 모델 비교 | KcELECTRA vs KLUE-RoBERTa vs KcBERT | 앙상블 후보 확보 |
| Exp-A6 | 풀링 | [CLS] vs Mean Pooling vs Attention Pooling | 후반부 위협 패턴 포착 |
| E01 | 앙상블 | 상위 2~3 모델 soft voting | — |

### 실험 우선순위 (시간 제한 시)

```
필수: B04 (v2 데이터), Exp-A5 (모델 비교)
권장: Exp-A4 (손실 함수), Exp-A1 (MAX_LEN 확장)
선택: Exp-A6 (풀링), Exp-A3 ([TURN] 토큰), E01 (앙상블)
```

### 핵심 교훈 (실험에서 확인됨)

1. **Val F1을 맹신하지 말 것** — B02: Val 0.990이지만 Test 일반대화 23건
2. **max_len=256 이상 사용** — S02b에서 +1.6%p 확인
3. **과잉 증강 주의** — ×3.4 증강이 오히려 역효과 (B02 vs S02)
4. **MPS 대신 CPU 사용** — Apple Silicon MPS 속도 이슈 (스텝당 30초 vs CPU 1.1초)
5. **save_total_limit 설정 필수** — 체크포인트 17GB 누적으로 디스크 풀 사고

## 9. Cross-Validation 전략

| 방식 | 설정 | 용도 |
|---|---|---|
| **단순 분할** | train_test_split(test_size=0.15, stratify) | 빠른 반복 실험 |
| **Stratified 5-Fold** | StratifiedKFold(n_splits=5, shuffle=True) | 최종 성능 평가 |

> 검증셋에 증강 데이터가 섞이면 안 됨 (strategy_v2.md 5-0 참고)

---

## 10. 노트북 셀 구성 (model.ipynb)

| 셀 번호 | 내용 | 비고 |
|---|---|---|
| **Cell 1** | 환경 설정 & 라이브러리 임포트 | transformers, torch, sklearn |
| **Cell 2** | 데이터 로드 & 전처리 | `baseline.csv` → `preprocess()` → label encoding |
| **Cell 3** | Train/Val 분할 | `StratifiedKFold` or `train_test_split` |
| **Cell 4** | Tokenizer & Dataset 정의 | MAX_LEN=256 |
| **Cell 5** | 모델 정의 | `ConversationClassifier` |
| **Cell 6** | 학습 설정 | Optimizer, Scheduler, Loss |
| **Cell 7** | 학습 실행 | 에폭별 Loss/F1 출력, best model 저장 |
| **Cell 8** | 평가 & 분석 | 혼동 행렬, Classification Report, 학습 곡선 |
| **Cell 9** | 추론 & 제출 생성 | `submission.csv` 출력 |
| **Cell 10** | Ablation Study 기록 | 실험 결과 테이블 |

---

## 11. 체크리스트 (DLThon.md 평가항목 매핑)

| 평가 항목 | 이 계획서에서의 대응 |
|---|---|
| ✅ 모델 선정 근거가 타당한가? | §1. KLUE-RoBERTa 선정 + 비교 모델(KoELECTRA) 언급 |
| ✅ 모델의 성능/학습 방향을 판단하고 개선을 시도한 기준이 논리적인가? | §6. 혼동 행렬 기반 분석 → §8. Ablation Study |
| ✅ 결과 도출을 위해 다양한 시도를 했는가? | §8. 5개 실험 변수 (MAX_LEN, Dropout, 전처리, Loss, 모델) |
| ✅ 도출된 결론에 충분한 설득력이 있는가? | §6. Classification Report + 혼동 행렬 시각화 |
