"""
Exp B03 — 숏컷 해소 + 클래스 균형 유지 (5 × 3,000 = 15,000건)
==============================================================
B02 대비 변경:
  1. 합성 일반 대화 문체 보정 — 마침표/존댓말 분포를 위협과 유사하게
  2. Label Smoothing 0.1 — 과신 방지
  3. 위협 증강은 B02와 동일 (경량 EDA)

숏컷 해소 핵심: 합성 일반의 마침표(0.38→5.0), 존댓말(0.83→3.5)을 test 분포에 맞춤
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

DATA_DIR = "aiffel-d-lthon-dktc-online-17"
TARGET = 3000

# ═══════════════════════════════════════════════════
# 1. 원본 위협 데이터 + 경량 증강 (B02와 동일)
# ═══════════════════════════════════════════════════
train_orig = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/submission.csv")

train_dedup = train_orig.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"원본: {len(train_dedup)}건")

PRESERVE = {
    "협박 대화": ["제발", "니가", "지금", "죽여", "죽고", "살려", "시키는"],
    "갈취 대화": ["돈이", "안돼", "없어", "뒤져", "맞고", "줄래", "내놔"],
    "직장 내 괴롭힘 대화": ["부장님", "죄송합니다", "아닙니다", "제가", "과장님", "팀장님"],
    "기타 괴롭힘 대화": ["아니야", "그렇게", "그만해", "무슨", "소리야"],
}

def is_preserve(word, cls):
    for kw in PRESERVE.get(cls, []):
        if kw in word: return True
    return False

def aug_delete(text, cls, p=0.08):
    turns = text.split("\n")
    out = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 3: out.append(turn); continue
        kept = [w for w in words if is_preserve(w, cls) or random.random() > p]
        out.append(" ".join(kept) if kept else turn)
    return "\n".join(out)

def aug_swap(text, cls, n=1):
    turns = text.split("\n")
    out = []
    for turn in turns:
        words = list(turn.split())
        if len(words) <= 2: out.append(turn); continue
        indices = list(range(len(words)-1))
        random.shuffle(indices)
        done = 0
        for i in indices:
            if done >= n: break
            if not is_preserve(words[i], cls) and not is_preserve(words[i+1], cls):
                words[i], words[i+1] = words[i+1], words[i]; done += 1
        out.append(" ".join(words))
    return "\n".join(out)

def aug_punct(text, cls):
    turns = text.split("\n")
    out = []
    for turn in turns:
        r = random.random()
        if r < 0.2 and turn.endswith("?"): turn = turn[:-1]
        elif r < 0.3 and not turn.endswith(("?","!",".")): turn += "."
        elif r < 0.4 and turn.endswith("."): turn = turn[:-1]
        out.append(turn)
    return "\n".join(out)

def augment(text, cls):
    method = random.choice(["del_light","del_med","swap","punct","combo_ds","combo_dp"])
    if method == "del_light": return aug_delete(text, cls, p=0.06)
    elif method == "del_med": return aug_delete(text, cls, p=0.12)
    elif method == "swap": return aug_swap(text, cls, n=1)
    elif method == "punct": return aug_punct(text, cls)
    elif method == "combo_ds": return aug_swap(aug_delete(text, cls, p=0.06), cls, n=1)
    elif method == "combo_dp": return aug_punct(aug_delete(text, cls, p=0.08), cls)
    return text

# 위협 4클래스 증강
threat_rows = []
for cls in ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"]:
    originals = train_dedup[train_dedup["class"]==cls]["conversation"].tolist()
    for c in originals:
        threat_rows.append({"class": cls, "conversation": c})

    n_need = TARGET - len(originals)
    orig_set = set(originals)
    generated = set()
    attempts = 0
    while len(generated) < n_need and attempts < n_need * 8:
        src = random.choice(originals)
        aug = augment(src, cls)
        if aug not in orig_set and aug not in generated:
            generated.add(aug)
            threat_rows.append({"class": cls, "conversation": aug})
        attempts += 1
    print(f"  {cls}: {len(originals)} + {len(generated)} = {len(originals)+len(generated)}")

# ═══════════════════════════════════════════════════
# 2. 합성 일반 대화 — 문체 보정 (핵심 변경)
# ═══════════════════════════════════════════════════
print("\n=== 합성 일반 대화 문체 보정 ===")

EMO_RE = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
normal_files = [
    "synthetic_normal_conversations.csv", "hard_negative_normal.csv",
    "normal_conversations_500.csv", "normal_conversations_2.csv", "normal_v2_batch1.csv",
]
raw_normals = []
for f in normal_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        raw_normals.extend(df["conversation"].tolist())

# 정제 + 문체 보정
POLITE_ENDINGS = ["요", "습니다", "세요", "겠습니다", "드려요", "하세요"]

def fix_style(text):
    """합성 일반 대화의 마침표/존댓말을 test 분포에 맞게 보정"""
    t = str(text)
    # 이모티콘 제거
    t = EMO_RE.sub("", t)
    t = re.sub(r"ㅋ|ㅎ", "", t)

    turns = [turn.strip() for turn in t.split("\n") if turn.strip()]
    new_turns = []

    for i, turn in enumerate(turns):
        # 마침표 보정: 30% 확률로 턴 끝에 마침표 추가 (현재 0.38 → 목표 ~4.0)
        if not turn.endswith((".", "?", "!", "요", "다")) and random.random() < 0.35:
            turn = turn + "."

        # 존댓말 보정: 20% 확률로 반말 → 존댓말 변환 (현재 0.83 → 목표 ~3.0)
        if random.random() < 0.25:
            # 간단한 반말→존댓말 변환
            if turn.endswith("어"): turn = turn[:-1] + "어요"
            elif turn.endswith("아"): turn = turn[:-1] + "아요"
            elif turn.endswith("지"): turn = turn[:-1] + "지요"
            elif turn.endswith("야"): turn = turn[:-1] + "에요"
            elif turn.endswith("거든"): turn = turn[:-2] + "거든요"
            elif turn.endswith("는데"): turn = turn[:-2] + "는데요"

        new_turns.append(turn)

    return "\n".join(new_turns)

cleaned_normals = []
seen = set()
for text in raw_normals:
    t = fix_style(text)
    t = re.sub(r" +", " ", t)
    if len(t) >= 130 and t not in seen:
        seen.add(t)
        cleaned_normals.append(t)

# 3,000건 맞추기
print(f"정제 후: {len(cleaned_normals)}건")

normal_rows = []
if len(cleaned_normals) >= TARGET:
    scored = [(t, abs(len(t)-215)) for t in cleaned_normals]
    scored.sort(key=lambda x: x[1])
    for t, _ in scored[:TARGET]:
        normal_rows.append({"class": "일반 대화", "conversation": t})
else:
    for t in cleaned_normals:
        normal_rows.append({"class": "일반 대화", "conversation": t})
    # 부족분 증강
    n_need = TARGET - len(cleaned_normals)
    gen = set()
    attempts = 0
    while len(gen) < n_need and attempts < n_need * 8:
        src = random.choice(cleaned_normals)
        aug = augment(src, "일반 대화")
        aug = fix_style(aug)  # 증강분도 문체 보정
        if aug not in gen and aug not in seen:
            gen.add(aug)
            normal_rows.append({"class": "일반 대화", "conversation": aug})
        attempts += 1
    print(f"부족분 {len(gen)}건 증강 추가")

if len(normal_rows) > TARGET:
    normal_rows = normal_rows[:TARGET]

# 보정 효과 확인
def preprocess(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

normal_texts = [preprocess(r["conversation"]) for r in normal_rows]
avg_period = np.mean([t.count(".") for t in normal_texts])
avg_polite = np.mean([sum(t.count(m) for m in ["요","습니다","세요","겠습니다"]) for t in normal_texts])
avg_excl = np.mean([t.count("!") for t in normal_texts])
avg_ques = np.mean([t.count("?") for t in normal_texts])
avg_len = np.mean([len(t) for t in normal_texts])
print(f"\n보정 후 합성 일반 특성:")
print(f"  마침표: 0.38 → {avg_period:.1f} (목표 ~4.0)")
print(f"  존댓말: 0.83 → {avg_polite:.1f} (목표 ~3.0)")
print(f"  느낌표: 0.04 → {avg_excl:.2f}")
print(f"  물음표: 2.22 → {avg_ques:.1f}")
print(f"  길이:   174  → {avg_len:.0f}")

# ═══════════════════════════════════════════════════
# 3. 병합 & 학습
# ═══════════════════════════════════════════════════
all_rows = threat_rows + normal_rows
train_full = pd.DataFrame(all_rows)
train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n최종 데이터: {len(train_full)}건")
print(train_full["class"].value_counts().to_string())

train_full["text"] = train_full["conversation"].apply(preprocess)
test_df["text"] = test_df["conversation"].apply(preprocess)

label2id = {"갈취 대화":0,"기타 괴롭힘 대화":1,"일반 대화":2,"직장 내 괴롭힘 대화":3,"협박 대화":4}
id2label = {v:k for k,v in label2id.items()}
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
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

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

print(f"\nModel: {MODEL_NAME}, Label Smoothing: 0.1")
print(f"Data: {len(train_full)}건 (5 × 3,000, 문체 보정)")
print("=" * 50)

start = time.time()
train_result = trainer.train()
elapsed = time.time() - start
print(f"\n학습 완료 — steps: {train_result.global_step}, loss: {train_result.training_loss:.4f}, 소요: {elapsed/60:.1f}분")

# 평가
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

# Test 예측
test_texts = test_df["text"].tolist()
test_dataset = DKTCDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LENGTH)
test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_pred_classes = [id2label[p] for p in test_pred_labels]

print("\nTest 예측 분포:")
print(pd.Series(test_pred_classes).value_counts().to_string())

os.makedirs("outputs", exist_ok=True)

# 문자열 버전
sub_str = pd.DataFrame({"idx": test_df["idx"], "class": test_pred_classes})
sub_str.to_csv("outputs/submission_B03_str.csv", index=False)

# 숫자 버전
train_order = {"협박 대화":0,"기타 괴롭힘 대화":1,"갈취 대화":2,"직장 내 괴롭힘 대화":3,"일반 대화":4}
sub_num = pd.DataFrame({"idx": test_df["idx"], "class": [train_order[c] for c in test_pred_classes]})
sub_num.to_csv("outputs/submission_B03_num.csv", index=False)
print("\nSubmission 저장 완료 (str + num)")

# Ablation
experiment = {
    "exp_id": "B03", "model": MODEL_NAME,
    "data": f"15,000건 (5×3,000, 문체 보정 + label smoothing)",
    "normal_count": len(normal_rows),
    "augmentation": "경량EDA + 문체보정 + label_smoothing=0.1",
    "lr": 2e-5, "epochs": 7, "max_length": MAX_LENGTH,
    "val_f1_macro": eval_results.get("eval_f1_macro", 0),
    "val_accuracy": eval_results.get("eval_accuracy", 0),
    "notes": "숏컷 해소: 합성 일반 마침표/존댓말 보정, label smoothing 0.1",
}
ablation_path = "outputs/ablation_study.csv"
existing = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else pd.DataFrame()
pd.concat([existing, pd.DataFrame([experiment])], ignore_index=True).to_csv(ablation_path, index=False)
print("Ablation 기록 완료")
