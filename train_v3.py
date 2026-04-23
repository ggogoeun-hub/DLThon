"""
Exp B03 — 숏컷 해소 전략
========================
핵심 변경:
  1. 위협 증강 전면 제거 (원본만 사용) → 증강 과적합 제거
  2. 합성 일반 최소화 (~1,000건) → 문체 숏컷 노출 줄이기
  3. Label Smoothing 0.1 → 과신 방지
  4. MAX_LEN=256 유지 (S02b에서 효과 확인)

근거:
  - 누적 인사이트 #10: 증강 ×3.4가 오히려 일반 예측 감소 (38→23)
  - 누적 인사이트 #8: 합성 일반의 마침표/존댓말 극단적 차이가 숏컷 유발
  - 누적 인사이트 #9: Val 99%는 문체 과적합, 낮은 Val이 오히려 나을 수 있음
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

# ═══════════════════════════════════════════════════
# 1. 데이터 구축 — 증강 없이 원본 + 정제 합성만
# ═══════════════════════════════════════════════════
DATA_DIR = "aiffel-d-lthon-dktc-online-17"

train_orig = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/submission.csv")

# 원본 중복 제거 (증강 없음!)
train_dedup = train_orig.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"원본 (증강 없음): {len(train_dedup)}건")
print(train_dedup["class"].value_counts().to_string())

# 합성 일반 대화 — 이모티콘 제거 + 품질 필터
EMO_RE = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
normal_files = [
    "synthetic_normal_conversations.csv",
    "hard_negative_normal.csv",
    "normal_conversations_500.csv",
    "normal_conversations_2.csv",
    "normal_v2_batch1.csv",
]
raw_normals = []
for f in normal_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        raw_normals.extend(df["conversation"].tolist())

# 정제
cleaned_normals = []
seen = set()
for text in raw_normals:
    t = str(text)
    t = EMO_RE.sub("", t)
    t = re.sub(r"ㅋ|ㅎ", "", t)
    t = re.sub(r" +", " ", t)
    turns = [turn.strip() for turn in t.split("\n") if turn.strip()]
    t = "\n".join(turns)
    if len(t) >= 150 and t not in seen:
        seen.add(t)
        cleaned_normals.append(t)

# 1,000건만 사용 (길이 기준 상위 — test 분포에 가깝게)
scored = [(t, abs(len(t) - 215)) for t in cleaned_normals]
scored.sort(key=lambda x: x[1])
selected_normals = [t for t, _ in scored[:1000]]

print(f"\n합성 일반: {len(raw_normals)}건 원본 → {len(cleaned_normals)}건 정제 → {len(selected_normals)}건 선택")
normal_lens = [len(t) for t in selected_normals]
print(f"  길이: {np.mean(normal_lens):.0f}±{np.std(normal_lens):.0f}자")

# 병합
normal_df = pd.DataFrame({"class": "일반 대화", "conversation": selected_normals})
train_full = pd.concat([train_dedup[["class", "conversation"]], normal_df], ignore_index=True)
print(f"\n최종 학습 데이터: {len(train_full)}건")
print(train_full["class"].value_counts().to_string())

# ═══════════════════════════════════════════════════
# 2. 전처리 & 토크나이저
# ═══════════════════════════════════════════════════
def preprocess(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

train_full["text"] = train_full["conversation"].apply(preprocess)
test_df["text"] = test_df["conversation"].apply(preprocess)

label2id = {"갈취 대화": 0, "기타 괴롭힘 대화": 1, "일반 대화": 2, "직장 내 괴롭힘 대화": 3, "협박 대화": 4}
id2label = {v: k for k, v in label2id.items()}
train_full["label"] = train_full["class"].map(label2id)

MODEL_NAME = "beomi/KcELECTRA-base"
MAX_LENGTH = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

# ═══════════════════════════════════════════════════
# 3. 학습 — Label Smoothing 적용
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
    output_dir="./outputs/kcelectra-B03",
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing_factor=0.1,  # ← 과신 방지
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
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
print(f"Label Smoothing: {training_args.label_smoothing_factor}")
print(f"Data: {len(train_full)}건 (증강 없음)")
print("=" * 50)
print("학습 시작...")

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

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# ═══════════════════════════════════════════════════
# 5. Test 예측 — 문자열 + 숫자 둘 다 저장
# ═══════════════════════════════════════════════════
test_texts = test_df["text"].tolist()
test_dataset = DKTCDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LENGTH)
test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_pred_classes = [id2label[p] for p in test_pred_labels]

print("\nTest 예측 분포:")
print(pd.Series(test_pred_classes).value_counts().to_string())

# softmax 확률 분석
import torch.nn.functional as F
probs = F.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()
max_probs = probs.max(axis=1)
print(f"\nTest 예측 확률: mean={max_probs.mean():.3f}, min={max_probs.min():.3f}, <0.5: {(max_probs<0.5).sum()}건")

os.makedirs("outputs", exist_ok=True)

# 문자열 버전
sub_str = pd.DataFrame({"idx": test_df["idx"], "class": test_pred_classes})
sub_str.to_csv("outputs/submission_B03_str.csv", index=False)

# 숫자 버전 (train 등장순: 협박=0, 기타=1, 갈취=2, 직장=3, 일반=4)
train_order = {"협박 대화": 0, "기타 괴롭힘 대화": 1, "갈취 대화": 2, "직장 내 괴롭힘 대화": 3, "일반 대화": 4}
sub_num = pd.DataFrame({"idx": test_df["idx"], "class": [train_order[c] for c in test_pred_classes]})
sub_num.to_csv("outputs/submission_B03_num.csv", index=False)

print("\nSubmission 저장: submission_B03_str.csv, submission_B03_num.csv")

# ═══════════════════════════════════════════════════
# 6. Ablation 기록
# ═══════════════════════════════════════════════════
experiment = {
    "exp_id": "B03", "model": MODEL_NAME,
    "data": f"원본 {len(train_dedup)} + 합성 {len(selected_normals)} = {len(train_full)}건 (증강 없음)",
    "normal_count": len(selected_normals),
    "augmentation": "없음 + label_smoothing=0.1",
    "lr": 2e-5, "epochs": training_args.num_train_epochs, "max_length": MAX_LENGTH,
    "val_f1_macro": eval_results.get("eval_f1_macro", 0),
    "val_accuracy": eval_results.get("eval_accuracy", 0),
    "notes": "숏컷 해소: 증강 제거, 합성 최소화, label smoothing",
}
ablation_path = "outputs/ablation_study.csv"
existing = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else pd.DataFrame()
pd.concat([existing, pd.DataFrame([experiment])], ignore_index=True).to_csv(ablation_path, index=False)
print("Ablation 기록 완료")
