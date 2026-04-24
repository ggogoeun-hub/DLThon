"""
baseline_B04.csv 생성
=====================
strategy_v2.md 증강 규칙 적용:
  - 위협 4클래스: 원본 → 랜덤 삭제 + 랜덤 스왑 + 구두점 변형 (동의어 교체/삽입 금지)
  - 일반 대화: v2 합성 975건 → 동일 경량 증강으로 3,000건
  - 문장 셔플 금지 (에스칬레이션 패턴 보존)
  - 보존 키워드 보호
  - 5 × 3,000 = 15,000건
"""

import pandas as pd
import numpy as np
import random
import re
import os

random.seed(42)
np.random.seed(42)

TARGET = 3000

# ═══════════════════════════════════════════════════
# 보존 키워드 (strategy_v2 3-2)
# ═══════════════════════════════════════════════════
PRESERVE = {
    "협박 대화": ["제발", "니가", "지금", "죽여", "죽고", "살려", "시키는"],
    "갈취 대화": ["돈이", "안돼", "없어", "뒤져", "맞고", "줄래", "내놔"],
    "직장 내 괴롭힘 대화": ["부장님", "죄송합니다", "아닙니다", "제가", "과장님", "팀장님"],
    "기타 괴롭힘 대화": ["아니야", "그렇게", "그만해", "무슨", "소리야"],
    "일반 대화": [],  # 일반 대화는 보존 키워드 없음
}


def is_preserve(word, cls):
    for kw in PRESERVE.get(cls, []):
        if kw in word:
            return True
    return False


# ═══════════════════════════════════════════════════
# 증강 함수 (strategy_v2 3-1: 삭제 + 스왑 + 구두점만)
# ═══════════════════════════════════════════════════

def aug_delete(text, cls, p=0.08):
    """비핵심 단어 소량 삭제"""
    turns = text.split("\n") if "\n" in text else [text]
    out = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 3:
            out.append(turn)
            continue
        kept = [w for w in words if is_preserve(w, cls) or random.random() > p]
        out.append(" ".join(kept) if kept else turn)
    sep = "\n" if "\n" in text else " "
    return sep.join(out)


def aug_swap(text, cls, n=1):
    """턴 내 인접 비핵심 단어 스왑"""
    turns = text.split("\n") if "\n" in text else [text]
    out = []
    for turn in turns:
        words = list(turn.split())
        if len(words) <= 2:
            out.append(turn)
            continue
        indices = list(range(len(words) - 1))
        random.shuffle(indices)
        done = 0
        for i in indices:
            if done >= n:
                break
            if not is_preserve(words[i], cls) and not is_preserve(words[i + 1], cls):
                words[i], words[i + 1] = words[i + 1], words[i]
                done += 1
        out.append(" ".join(words))
    sep = "\n" if "\n" in text else " "
    return sep.join(out)


def aug_punct(text, cls):
    """구두점 미세 변형"""
    turns = text.split("\n") if "\n" in text else text.split(". ")
    out = []
    for turn in turns:
        r = random.random()
        if r < 0.15 and turn.endswith("?"):
            turn = turn[:-1]
        elif r < 0.25 and not turn.endswith(("?", "!", ".")):
            turn = turn + "."
        elif r < 0.35 and turn.endswith("."):
            turn = turn[:-1]
        out.append(turn)
    sep = "\n" if "\n" in text else ". "
    return sep.join(out)


def augment(text, cls):
    method = random.choice(["del_light", "del_med", "swap", "punct", "combo_ds", "combo_dp"])
    if method == "del_light":
        return aug_delete(text, cls, p=0.06)
    elif method == "del_med":
        return aug_delete(text, cls, p=0.12)
    elif method == "swap":
        return aug_swap(text, cls, n=1)
    elif method == "punct":
        return aug_punct(text, cls)
    elif method == "combo_ds":
        return aug_swap(aug_delete(text, cls, p=0.06), cls, n=1)
    elif method == "combo_dp":
        return aug_punct(aug_delete(text, cls, p=0.08), cls)
    return text


def generate_augmented(originals, cls, n_needed):
    generated = set()
    orig_set = set(originals)
    attempts = 0
    max_att = n_needed * 10
    while len(generated) < n_needed and attempts < max_att:
        src = random.choice(originals)
        aug = augment(src, cls)
        if aug not in orig_set and aug not in generated:
            generated.add(aug)
        attempts += 1
    return list(generated)


# ═══════════════════════════════════════════════════
# 1. 원본 위협 데이터
# ═══════════════════════════════════════════════════
print("=" * 60)
print("1. 원본 위협 데이터 로드")
print("=" * 60)

BASE = "/Users/goeunlee/aiffel/DLthon"
train_orig = pd.read_csv(f"{BASE}/data/train.csv")
train_dedup = train_orig.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"원본: {len(train_dedup)}건")
print(train_dedup["class"].value_counts().to_string())

# ═══════════════════════════════════════════════════
# 2. 위협 4클래스 증강
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. 위협 4클래스 증강 (경량 EDA: 삭제+스왑+구두점)")
print("=" * 60)

threat_rows = []
for cls in ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"]:
    originals = train_dedup[train_dedup["class"] == cls]["conversation"].tolist()
    for c in originals:
        threat_rows.append({"class": cls, "conversation": c})

    n_need = TARGET - len(originals)
    aug_list = generate_augmented(originals, cls, n_need)
    for c in aug_list:
        threat_rows.append({"class": cls, "conversation": c})

    total = len(originals) + len(aug_list)
    print(f"  {cls}: {len(originals)} + {len(aug_list)} = {total}")

# ═══════════════════════════════════════════════════
# 3. 일반 대화 — v2 합성 975건 → 3,000건
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. 일반 대화 (v2 합성 975건 → 3,000건)")
print("=" * 60)

normal_df = pd.read_csv(f"{BASE}/data/synthetic_normal_v2_clean.csv")
normal_originals = normal_df["conversation"].tolist()
print(f"  v2 합성 정제본: {len(normal_originals)}건")

normal_rows = []
for c in normal_originals:
    normal_rows.append({"class": "일반 대화", "conversation": c})

n_need = TARGET - len(normal_originals)
print(f"  증강 필요: {n_need}건")

aug_normals = generate_augmented(normal_originals, "일반 대화", n_need)
for c in aug_normals:
    normal_rows.append({"class": "일반 대화", "conversation": c})

print(f"  일반 대화 최종: {len(normal_rows)}건")

# ═══════════════════════════════════════════════════
# 4. 병합 & 저장
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. 병합 & 저장")
print("=" * 60)

all_rows = threat_rows + normal_rows
baseline = pd.DataFrame(all_rows)
baseline = baseline.sample(frac=1, random_state=42).reset_index(drop=True)
baseline.insert(0, "idx", range(len(baseline)))

output = f"{BASE}/data/baseline/baseline_B04.csv"
os.makedirs(f"{BASE}/data/baseline", exist_ok=True)
baseline.to_csv(output, index=False)

print(f"총 {len(baseline)}건")
print(baseline["class"].value_counts().to_string())

# ═══════════════════════════════════════════════════
# 5. 품질 검증
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. 품질 검증")
print("=" * 60)

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

# 중복
dup = baseline.duplicated(subset="conversation").sum()
cross = (baseline.groupby("conversation")["class"].nunique() > 1).sum()
print(f"  중복: {dup}건, 교차: {cross}건")

# 원본 보존
for cls in train_dedup["class"].unique():
    orig_c = set(train_dedup[train_dedup["class"] == cls]["conversation"])
    base_c = set(baseline[baseline["class"] == cls]["conversation"])
    missing = len(orig_c - base_c)
    print(f"  보존 {cls}: {'✅' if missing == 0 else f'❌ {missing}건 누락'}")

# 이모티콘
EMO_RE = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"]
    emo = texts.apply(lambda x: bool(EMO_RE.search(str(x)))).sum()
    print(f"  이모티콘 {cls}: {emo}건")

# 클래스별 통계 (전처리 후)
print("\n  [전처리 후 통계]")
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"].apply(preprocess)
    lens = texts.apply(len)
    periods = texts.apply(lambda x: x.count("."))
    polite_cnt = texts.apply(lambda x: sum(x.count(m) for m in ["요", "습니다", "세요", "겠습니다", "에요", "해요"]))
    print(f"  {cls}: 길이 {lens.mean():.0f}±{lens.std():.0f}, 마침표 {periods.mean():.1f}, 존댓말 {polite_cnt.mean():.1f}")

print(f"\n✅ 저장 완료: {output}")
