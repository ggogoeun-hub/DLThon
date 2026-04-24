"""
Exp B04 — v2 전략 + KLUE-RoBERTa
=================================
변경점 (vs B03):
  - 모델: KcELECTRA → beomi/KcELECTRA-base (맥락 파악 강점)
  - 데이터: baseline_B04.csv (33도메인 Gemini 합성 + 경량 증강)
  - Label Smoothing 0.1 유지
"""

import os, re, random, time
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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(42)

BASE = "/Users/goeunlee/aiffel/DLthon"

# ═══════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════
train_full = pd.read_csv(f"{BASE}/data/baseline/baseline_B04.csv")
test_df = pd.read_csv(f"{BASE}/data/test.csv")
submission = pd.read_csv(f"{BASE}/data/submission.csv")

print(f"Train: {train_full.shape}")
print(train_full["class"].value_counts().to_string())

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

train_full["text"] = train_full["conversation"].apply(preprocess)
test_df["text"] = test_df["conversation"].apply(preprocess)

label2id = {"갈취 대화":0,"기타 괴롭힘 대화":1,"일반 대화":2,"직장 내 괴롭힘 대화":3,"협박 대화":4}
id2label = {v:k for k,v in label2id.items()}
train_full["label"] = train_full["class"].map(label2id)

# ═══════════════════════════════════════════════════
# 2. 토크나이저 & 데이터셋
# ═══════════════════════════════════════════════════
MODEL_NAME = "beomi/KcELECTRA-base"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_lengths = [len(tokenizer.encode(t)) for t in train_full["text"].sample(500, random_state=42)]
print(f"\n토큰 길이: mean={np.mean(sample_lengths):.0f}, 95th={np.percentile(sample_lengths, 95):.0f}, "
      f"커버율(≤{MAX_LENGTH}): {sum(1 for l in sample_lengths if l <= MAX_LENGTH)/len(sample_lengths)*100:.1f}%")

class DKTCDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        return enc

X_train, X_val, y_train, y_val = train_test_split(
    train_full["text"].tolist(), train_full["label"].tolist(),
    test_size=0.15, random_state=42, stratify=train_full["label"].tolist(),
)
train_dataset = DKTCDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = DKTCDataset(X_val, y_val, tokenizer, MAX_LENGTH)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ═══════════════════════════════════════════════════
# 3. 학습
# ═══════════════════════════════════════════════════
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {
        "f1_macro": f1_score(eval_pred.label_ids, preds, average="macro"),
        "f1_weighted": f1_score(eval_pred.label_ids, preds, average="weighted"),
        "accuracy": accuracy_score(eval_pred.label_ids, preds),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5, id2label=id2label, label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=f"{BASE}/outputs/kcelectra-B04b",
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing_factor=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
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
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=data_collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"\nModel: {MODEL_NAME}")
print(f"Data: baseline_B04.csv ({len(train_full)}건, v2 전략)")
print(f"Label Smoothing: 0.1")
print("=" * 50)

start = time.time()
train_result = trainer.train()
elapsed = time.time() - start
print(f"\n학습 완료 — steps: {train_result.global_step}, loss: {train_result.training_loss:.4f}, 소요: {elapsed/60:.1f}분")

# ═══════════════════════════════════════════════════
# 4. 평가
# ═══════════════════════════════════════════════════
eval_results = trainer.evaluate()
print("\n=== Validation ===")
for k, v in eval_results.items():
    if isinstance(v, float): print(f"  {k}: {v:.4f}")

val_preds = trainer.predict(val_dataset)
y_pred = np.argmax(val_preds.predictions, axis=-1)
target_names = [id2label[i] for i in range(5)]
print(f"\n{classification_report(y_val, y_pred, target_names=target_names, digits=4)}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ═══════════════════════════════════════════════════
# 5. Test 예측
# ═══════════════════════════════════════════════════
test_texts = test_df["text"].tolist()
test_dataset = DKTCDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LENGTH)
test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_pred_classes = [id2label[p] for p in test_pred_labels]

print("\nTest 예측 분포:")
print(pd.Series(test_pred_classes).value_counts().to_string())

os.makedirs(f"{BASE}/outputs", exist_ok=True)

sub_str = pd.DataFrame({"idx": test_df["idx"], "class": test_pred_classes})
sub_str.to_csv(f"{BASE}/outputs/submission_B04_str.csv", index=False)

train_order = {"협박 대화":0,"기타 괴롭힘 대화":1,"갈취 대화":2,"직장 내 괴롭힘 대화":3,"일반 대화":4}
sub_num = pd.DataFrame({"idx": test_df["idx"], "class": [train_order[c] for c in test_pred_classes]})
sub_num.to_csv(f"{BASE}/outputs/submission_B04_num.csv", index=False)
print("Submission 저장 완료 (str + num)")

# ═══════════════════════════════════════════════════
# 6. Ablation 기록
# ═══════════════════════════════════════════════════
experiment = {
    "exp_id": "B04b", "model": MODEL_NAME,
    "data": "baseline_B04.csv (v2 전략, 33도메인 Gemini 합성, 15k)",
    "normal_count": 3000,
    "augmentation": "경량EDA + label_smoothing=0.1",
    "lr": 2e-5, "epochs": 7, "max_length": MAX_LENGTH,
    "val_f1_macro": eval_results.get("eval_f1_macro", 0),
    "val_accuracy": eval_results.get("eval_accuracy", 0),
    "notes": "v2 전략 + KcELECTRA (모델 효과 분리). 33도메인 합성, 마침표/존댓말 test 일치",
}
ablation_path = f"{BASE}/outputs/ablation_study.csv"
existing = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else pd.DataFrame()
pd.concat([existing, pd.DataFrame([experiment])], ignore_index=True).to_csv(ablation_path, index=False)
print("Ablation 기록 완료")
