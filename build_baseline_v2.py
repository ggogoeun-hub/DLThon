"""
baseline.csv v2 — 정밀 수정 버전
==================================
수정 사항:
  1. 일반 대화: 이모티콘 전면 제거 + 150자 미만 필터 + 부족분 보충
  2. 위협 증강: 랜덤 삽입/동의어 교체 제거 → 경량 증강(삭제+스왑)만 사용
  3. 품질 검증 통합
"""

import pandas as pd
import random
import re
import os
import numpy as np
from collections import Counter

random.seed(42)
np.random.seed(42)

DATA_DIR = "aiffel-d-lthon-dktc-online-17"
TARGET = 3000
MIN_NORMAL_LEN = 130  # 일반 대화 최소 길이 (char)

# 보존 키워드
PRESERVE = {
    "협박 대화": ["제발", "니가", "지금", "죽여", "죽고", "살려", "시키는"],
    "갈취 대화": ["돈이", "안돼", "없어", "뒤져", "맞고", "줄래", "내놔"],
    "직장 내 괴롭힘 대화": ["부장님", "죄송합니다", "아닙니다", "제가", "과장님", "팀장님"],
    "기타 괴롭힘 대화": ["아니야", "그렇게", "그만해", "무슨", "소리야"],
}

BANNED = ["죽여", "죽어", "맞을래", "시키는 대로", "내놔", "닥쳐", "패버릴"]
EMO_RE = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
SPACE_RE = re.compile(r" {2,}")


def is_preserve(word, cls):
    for kw in PRESERVE.get(cls, []):
        if kw in word:
            return True
    return False


# ═══════════════════════════════════════════════════
# 경량 증강 함수 (삽입/동의어 교체 제거)
# ═══════════════════════════════════════════════════

def aug_delete(text, cls, p=0.08):
    """턴 내 비핵심 단어 소량 삭제"""
    turns = text.split("\n")
    out = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 3:
            out.append(turn)
            continue
        kept = [w for w in words if is_preserve(w, cls) or random.random() > p]
        out.append(" ".join(kept) if kept else turn)
    return "\n".join(out)


def aug_swap(text, cls, n=1):
    """턴 내 인접 비핵심 단어 1회 스왑"""
    turns = text.split("\n")
    out = []
    for turn in turns:
        words = list(turn.split())
        if len(words) <= 2:
            out.append(turn)
            continue
        done = 0
        indices = list(range(len(words) - 1))
        random.shuffle(indices)
        for i in indices:
            if done >= n:
                break
            if not is_preserve(words[i], cls) and not is_preserve(words[i + 1], cls):
                words[i], words[i + 1] = words[i + 1], words[i]
                done += 1
        out.append(" ".join(words))
    return "\n".join(out)


def aug_punct(text, cls):
    """구두점 미세 변형 (? ↔ 제거, . 추가/제거)"""
    turns = text.split("\n")
    out = []
    for turn in turns:
        r = random.random()
        if r < 0.2 and turn.endswith("?"):
            turn = turn[:-1]
        elif r < 0.3 and not turn.endswith(("?", "!", ".")):
            turn = turn + "."
        elif r < 0.4 and turn.endswith("."):
            turn = turn[:-1]
        out.append(turn)
    return "\n".join(out)


def augment(text, cls, method=None):
    if method is None:
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


def generate_augmented(originals, cls, n_needed, max_attempts_factor=8):
    """원본 리스트에서 n_needed개 증강 데이터 생성"""
    generated = set()
    orig_set = set(originals)
    attempts = 0
    max_att = n_needed * max_attempts_factor

    while len(generated) < n_needed and attempts < max_att:
        src = random.choice(originals)
        aug = augment(src, cls)
        if aug not in orig_set and aug not in generated:
            generated.add(aug)
        attempts += 1

    return list(generated)


# ═══════════════════════════════════════════════════
# 1. 원본 로드 & 중복 제거
# ═══════════════════════════════════════════════════
print("=" * 60)
print("1. 원본 로드 & 중복 제거")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_dedup = train_df.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"원본 {len(train_df)} → 중복제거 {len(train_dedup)}")
print(train_dedup["class"].value_counts().to_string())


# ═══════════════════════════════════════════════════
# 2. 위협 4클래스 증강 (경량 방식)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. 위협 4클래스 증강")
print("=" * 60)

threat_rows = []
for cls in ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"]:
    originals = train_dedup[train_dedup["class"] == cls]["conversation"].tolist()
    n_orig = len(originals)
    n_need = TARGET - n_orig

    # 원본 추가
    for c in originals:
        threat_rows.append({"class": cls, "conversation": c})

    # 증강
    aug_list = generate_augmented(originals, cls, n_need)
    for c in aug_list:
        threat_rows.append({"class": cls, "conversation": c})

    total = n_orig + len(aug_list)
    print(f"  {cls}: {n_orig} + {len(aug_list)} = {total}건", "✅" if total >= TARGET else f"⚠️ {TARGET - total}건 부족")


# ═══════════════════════════════════════════════════
# 3. 일반 대화 — 이모티콘 제거 + 품질 필터
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. 일반 대화 정제")
print("=" * 60)

normal_files = [
    "synthetic_normal_conversations.csv",
    "hard_negative_normal.csv",
    "normal_conversations_500.csv",
    "normal_conversations_2.csv",
    "normal_v2_batch1.csv",
]

raw_convs = []
for f in normal_files:
    path = os.path.join("/Users/goeunlee/aiffel/DLthon", f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        raw_convs.extend(df["conversation"].tolist())

raw_convs = list(set(raw_convs))  # 중복 제거
print(f"  원본 합성 데이터: {len(raw_convs)}건 (고유)")

# 이모티콘 제거 + 정리
cleaned = []
for text in raw_convs:
    t = str(text)
    t = EMO_RE.sub("", t)           # ㅋㅋ, ㅎㅎ, ㅠㅠ 제거
    t = re.sub(r"ㅋ|ㅎ", "", t)      # 단일 ㅋ, ㅎ도 제거
    t = SPACE_RE.sub(" ", t)         # 다중 공백 정리
    # 각 턴 앞뒤 공백 정리
    turns = [turn.strip() for turn in t.split("\n") if turn.strip()]
    t = "\n".join(turns)
    if len(t) >= MIN_NORMAL_LEN:
        # 금지 단어 체크
        safe = all(bw not in t for bw in BANNED)
        if safe:
            cleaned.append(t)

print(f"  이모티콘 제거 + 길이 필터({MIN_NORMAL_LEN}자↑) 후: {len(cleaned)}건")

# 이모티콘 재검증
emo_remain = sum(1 for t in cleaned if EMO_RE.search(t))
print(f"  이모티콘 잔존: {emo_remain}건")

# 길이 분포
lens = [len(t) for t in cleaned]
print(f"  길이: {np.mean(lens):.0f}±{np.std(lens):.0f}자, 범위 {min(lens)}~{max(lens)}")

# 3,000건 맞추기
normal_rows = []
if len(cleaned) >= TARGET:
    # 길이 기준 상위 3,000건 (180~350 범위 우선)
    scored = [(t, abs(len(t) - 215)) for t in cleaned]  # 215자(목표 중앙)에서의 거리
    scored.sort(key=lambda x: x[1])
    selected = [t for t, _ in scored[:TARGET]]
    for t in selected:
        normal_rows.append({"class": "일반 대화", "conversation": t})
    print(f"  품질 기준 상위 {TARGET}건 선택")
else:
    for t in cleaned:
        normal_rows.append({"class": "일반 대화", "conversation": t})

    # 부족분 증강 (경량)
    n_need = TARGET - len(cleaned)
    print(f"  부족분: {n_need}건 → 경량 증강")

    aug_normals = generate_augmented(cleaned, "일반 대화", n_need)
    # 증강분도 이모티콘/금지어 재검증
    for t in aug_normals:
        t = EMO_RE.sub("", t)
        t = re.sub(r"ㅋ|ㅎ", "", t)
        t = SPACE_RE.sub(" ", t)
        if all(bw not in t for bw in BANNED):
            normal_rows.append({"class": "일반 대화", "conversation": t})

    print(f"  일반 대화 최종: {len(normal_rows)}건")

# 3,000 초과 시 트림
if len(normal_rows) > TARGET:
    normal_rows = normal_rows[:TARGET]


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

output = "/Users/goeunlee/aiffel/DLthon/baseline.csv"
baseline.to_csv(output, index=False)

print(f"총 {len(baseline)}건")
print(baseline["class"].value_counts().to_string())


# ═══════════════════════════════════════════════════
# 5. 전체 품질 검증
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. 품질 검증")
print("=" * 60)

# 5-1. 중복
dup = baseline.duplicated(subset="conversation").sum()
cross = (baseline.groupby("conversation")["class"].nunique() > 1).sum()
print(f"  [중복] 완전: {dup}, 교차: {cross}")

# 5-2. 원본 보존
orig_dedup = train_dedup
for cls in orig_dedup["class"].unique():
    orig_c = set(orig_dedup[orig_dedup["class"] == cls]["conversation"])
    base_c = set(baseline[baseline["class"] == cls]["conversation"])
    missing = len(orig_c - base_c)
    print(f"  [보존] {cls}: {missing}건 누락" if missing > 0 else f"  [보존] {cls}: ✅ 전체 보존")

# 5-3. 이모티콘
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"]
    emo_n = texts.apply(lambda x: bool(EMO_RE.search(str(x)))).sum()
    pct = emo_n / len(texts) * 100
    flag = "🔴" if pct > 5 else "✅"
    print(f"  [이모티콘] {flag} {cls}: {emo_n}건 ({pct:.1f}%)")

# 5-4. 길이
print("  [길이 분포]")
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"]
    lens = texts.apply(len)
    turns = texts.apply(lambda x: len(str(x).split("\n")))
    print(f"    {cls}: {lens.mean():.0f}±{lens.std():.0f}자, 턴 {turns.mean():.1f}±{turns.std():.1f}")

# 5-5. 금지 단어 (일반 대화)
normal_t = baseline[baseline["class"] == "일반 대화"]["conversation"]
ban_n = sum(1 for t in normal_t if any(bw in str(t) for bw in BANNED))
print(f"  [금지어] 일반 대화: {ban_n}건")

# 5-6. 말투
polite = ["요", "습니다", "세요", "겠습니다"]
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"].tolist()
    p_sum = sum(sum(str(t).count(m) for m in polite) for t in texts) / len(texts)
    print(f"  [존댓말] {cls}: {p_sum:.1f}회/건")

# 5-7. 특수문자
for cls in baseline["class"].unique():
    texts = baseline[baseline["class"] == cls]["conversation"].tolist()
    excl = sum(str(t).count("!") for t in texts) / len(texts)
    ques = sum(str(t).count("?") for t in texts) / len(texts)
    print(f"  [특수문자] {cls}: ! {excl:.2f}/건, ? {ques:.2f}/건")

print(f"\n✅ baseline.csv v2 저장 완료: {output}")
