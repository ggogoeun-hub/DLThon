"""
DKTC 모델 학습 스크립트
======================
baseline.csv (15,000건) → KcELECTRA-base → submission 생성
"""

import os
import re
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── 시드 고정 ───
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)

# ─── 디바이스 ───
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ═══════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════
DATA_DIR = "aiffel-d-lthon-dktc-online-17"

train_full = pd.read_csv("baseline.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/submission.csv")

print(f"Train: {train_full.shape}")
print(f"Test:  {test_df.shape}")
print(train_full["class"].value_counts().to_string())

# ─── 전처리 (train/test 동일) ───
def preprocess_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_full["text"] = train_full["conversation"].apply(preprocess_text)
test_df["text"] = test_df["conversation"].apply(preprocess_text)

# ─── 레이블 인코딩 ───
label2id = {
    "갈취 대화": 0,
    "기타 괴롭힘 대화": 1,
    "일반 대화": 2,
    "직장 내 괴롭힘 대화": 3,
    "협박 대화": 4,
}
id2label = {v: k for k, v in label2id.items()}
train_full["label"] = train_full["class"].map(label2id)

# ═══════════════════════════════════════════════════
# 2. 토크나이저 & 데이터셋
# ═══════════════════════════════════════════════════
MODEL_NAME = "beomi/KcELECTRA-base"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 토큰 길이 분포
sample_lengths = [len(tokenizer.encode(t)) for t in train_full["text"].sample(500, random_state=42)]
print(f"\n토큰 길이: mean={np.mean(sample_lengths):.0f}, 95th={np.percentile(sample_lengths, 95):.0f}, "
      f"커버율(≤{MAX_LENGTH}): {sum(1 for l in sample_lengths if l <= MAX_LENGTH)/len(sample_lengths)*100:.1f}%")


class DKTCDataset(Dataset):
    """Dynamic padding — 패딩은 DataCollator가 배치 단위로 처리"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
        )
        enc["labels"] = self.labels[idx]
        return enc


# ─── Train/Val 분할 ───
X_train, X_val, y_train, y_val = train_test_split(
    train_full["text"].tolist(),
    train_full["label"].tolist(),
    test_size=0.15,
    random_state=42,
    stratify=train_full["label"].tolist(),
)

train_dataset = DKTCDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = DKTCDataset(X_val, y_val, tokenizer, MAX_LENGTH)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ═══════════════════════════════════════════════════
# 3. 모델 학습
# ═══════════════════════════════════════════════════
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy": accuracy_score(labels, preds),
    }


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5, id2label=id2label, label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./outputs/kcelectra-baseline-v2",
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=100,
    seed=42,
    report_to="none",
    fp16=False,
    use_cpu=True,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"\nModel: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {training_args.device}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print("=" * 50)
print("학습 시작...")

start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time
print(f"\n학습 완료 — steps: {train_result.global_step}, loss: {train_result.training_loss:.4f}, 소요: {elapsed/60:.1f}분")

# ═══════════════════════════════════════════════════
# 4. 평가
# ═══════════════════════════════════════════════════
eval_results = trainer.evaluate()
print("\n=== Validation 결과 ===")
for k, v in eval_results.items():
    print(f"  {k}: {v:.4f}")

val_preds = trainer.predict(val_dataset)
y_pred = np.argmax(val_preds.predictions, axis=-1)

target_names = [id2label[i] for i in range(5)]
print(f"\n{classification_report(y_val, y_pred, target_names=target_names, digits=4)}")

# Confusion Matrix 저장
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# ═══════════════════════════════════════════════════
# 5. Test 예측 & Submission
# ═══════════════════════════════════════════════════
test_texts = test_df["text"].tolist()
test_dataset = DKTCDataset(test_texts, [0] * len(test_texts), tokenizer, MAX_LENGTH)

test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_pred_classes = [id2label[p] for p in test_pred_labels]

print("\nTest 예측 분포:")
print(pd.Series(test_pred_classes).value_counts().to_string())

submission["class"] = test_pred_classes
os.makedirs("outputs", exist_ok=True)
submission_path = "outputs/submission_baseline_v2.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission 저장: {submission_path}")

# ═══════════════════════════════════════════════════
# 6. Ablation 기록
# ═══════════════════════════════════════════════════
experiment = {
    "exp_id": "B02",
    "model": MODEL_NAME,
    "data": "baseline.csv (v2, 15000건)",
    "augmentation": "경량EDA(삭제+스왑+구두점)",
    "lr": 2e-5,
    "epochs": training_args.num_train_epochs,
    "max_length": MAX_LENGTH,
    "val_f1_macro": eval_results.get("eval_f1_macro", 0),
    "val_accuracy": eval_results.get("eval_accuracy", 0),
    "notes": "baseline.csv v2 - 이모티콘 제거, 경량 증강",
}

ablation_path = "outputs/ablation_study.csv"
if os.path.exists(ablation_path):
    abl = pd.read_csv(ablation_path)
    abl = pd.concat([abl, pd.DataFrame([experiment])], ignore_index=True)
else:
    abl = pd.DataFrame([experiment])
abl.to_csv(ablation_path, index=False)
print("\nAblation 기록 저장 완료")
