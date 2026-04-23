"""
baseline.csv 생성 스크립트
==========================
strategy_v1.md 기반:
- 원본 4클래스: 중복 제거 → 각 3,000건 증강
- 일반 대화: 합성 CSV 병합 → 3,000건
- 총 15,000건 (5 × 3,000)

증강 기법: 동의어 교체(클래스 내 어휘), 랜덤 삭제, 랜덤 스왑, 복합 증강
※ 문장 셔플은 에스칼레이션 패턴 보존을 위해 제외
※ 핵심 키워드는 증강 시 보존
"""

import pandas as pd
import random
import re
import os
from collections import Counter

random.seed(42)

# ──────────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────────
DATA_DIR = "aiffel-d-lthon-dktc-online-17"
TARGET_PER_CLASS = 3000

# 클래스별 보존 필수 키워드 (strategy_v1.md 2-4절)
PRESERVE_KEYWORDS = {
    "협박 대화": ["제발", "니가", "지금", "죽여", "죽고", "살려", "시키는"],
    "갈취 대화": ["돈이", "안돼", "없어", "뒤져", "맞고", "줄래", "내놔"],
    "직장 내 괴롭힘 대화": ["부장님", "죄송합니다", "아닙니다", "제가", "과장님", "팀장님"],
    "기타 괴롭힘 대화": ["아니야", "그렇게", "그만해", "무슨", "소리야"],
}

# 금지 단어 (일반 대화 합성 시)
BANNED_WORDS = ["죽여", "죽어", "맞을래", "시키는 대로", "내놔", "닥쳐", "패버릴"]


# ──────────────────────────────────────────────────
# 1. 데이터 로드 & 중복 제거
# ──────────────────────────────────────────────────
print("=" * 60)
print("1. 데이터 로드 & 중복 제거")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"원본: {len(train_df)}건")

train_dedup = train_df.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"중복 제거 후: {len(train_dedup)}건")
print(train_dedup["class"].value_counts().to_string())
print()


# ──────────────────────────────────────────────────
# 2. 일반 대화 합성 데이터 병합
# ──────────────────────────────────────────────────
print("=" * 60)
print("2. 일반 대화 합성 데이터 병합")
print("=" * 60)

normal_files = [
    "synthetic_normal_conversations.csv",
    "hard_negative_normal.csv",
    "normal_conversations_500.csv",
    "normal_conversations_2.csv",
    "normal_v2_batch1.csv",
]

normal_dfs = []
for f in normal_files:
    path = os.path.join("/Users/goeunlee/aiffel/DLthon", f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        normal_dfs.append(df)
        print(f"  {f}: {len(df)}건")

normal_all = pd.concat(normal_dfs, ignore_index=True)
normal_all = normal_all.drop_duplicates(subset="conversation").reset_index(drop=True)
print(f"일반 대화 합계 (중복 제거): {len(normal_all)}건")
print()


# ──────────────────────────────────────────────────
# 3. 텍스트 증강 함수들 (EDA 기반)
# ──────────────────────────────────────────────────

def get_words(text):
    """텍스트를 단어 리스트로 분리 (턴 구조 보존)"""
    return text.split()


def is_preserve_word(word, class_name):
    """보존 필수 키워드인지 확인"""
    keywords = PRESERVE_KEYWORDS.get(class_name, [])
    for kw in keywords:
        if kw in word:
            return True
    return False


def random_deletion(text, class_name, p=0.15):
    """턴 단위로 랜덤 단어 삭제 (보존 키워드 제외)"""
    turns = text.split("\n")
    new_turns = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 3:
            new_turns.append(turn)
            continue
        new_words = []
        for w in words:
            if is_preserve_word(w, class_name):
                new_words.append(w)
            elif random.random() > p:
                new_words.append(w)
        if not new_words:
            new_words = [random.choice(words)]
        new_turns.append(" ".join(new_words))
    return "\n".join(new_turns)


def random_swap(text, class_name, n=2):
    """턴 내에서 인접 단어 스왑 (보존 키워드 제외)"""
    turns = text.split("\n")
    new_turns = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 2:
            new_turns.append(turn)
            continue
        words = list(words)
        for _ in range(min(n, len(words) - 1)):
            idx = random.randint(0, len(words) - 2)
            if not is_preserve_word(words[idx], class_name) and not is_preserve_word(words[idx + 1], class_name):
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        new_turns.append(" ".join(words))
    return "\n".join(new_turns)


def random_insertion(text, class_name, class_vocab, n=2):
    """턴에 같은 클래스 어휘에서 랜덤 단어 삽입"""
    turns = text.split("\n")
    new_turns = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 1:
            new_turns.append(turn)
            continue
        words = list(words)
        for _ in range(min(n, 2)):
            insert_word = random.choice(class_vocab)
            pos = random.randint(0, len(words))
            words.insert(pos, insert_word)
        new_turns.append(" ".join(words))
    return "\n".join(new_turns)


def synonym_replace(text, class_name, class_vocab, n=3):
    """보존 키워드가 아닌 단어를 같은 클래스 어휘로 교체"""
    turns = text.split("\n")
    new_turns = []
    for turn in turns:
        words = turn.split()
        if len(words) <= 2:
            new_turns.append(turn)
            continue
        words = list(words)
        replaceable = [i for i, w in enumerate(words)
                       if not is_preserve_word(w, class_name) and len(w) > 1]
        random.shuffle(replaceable)
        for i in replaceable[:n]:
            words[i] = random.choice(class_vocab)
        new_turns.append(" ".join(words))
    return "\n".join(new_turns)


def augment_text(text, class_name, class_vocab, method=None):
    """단일 텍스트 증강 (방법 지정 또는 랜덤 선택)"""
    if method is None:
        method = random.choice(["delete", "swap", "insert", "synonym", "combo"])

    if method == "delete":
        return random_deletion(text, class_name, p=random.uniform(0.1, 0.2))
    elif method == "swap":
        return random_swap(text, class_name, n=random.randint(1, 3))
    elif method == "insert":
        return random_insertion(text, class_name, class_vocab, n=random.randint(1, 3))
    elif method == "synonym":
        return synonym_replace(text, class_name, class_vocab, n=random.randint(2, 4))
    elif method == "combo":
        # 복합 증강: 2~3개 기법 순차 적용
        t = text
        methods = random.sample(["delete", "swap", "insert", "synonym"], k=random.randint(2, 3))
        for m in methods:
            t = augment_text(t, class_name, class_vocab, method=m)
        return t
    return text


# ──────────────────────────────────────────────────
# 4. 클래스별 어휘 사전 구축
# ──────────────────────────────────────────────────
print("=" * 60)
print("3. 클래스별 어휘 사전 구축")
print("=" * 60)

class_vocabs = {}
for cls in train_dedup["class"].unique():
    texts = train_dedup[train_dedup["class"] == cls]["conversation"].tolist()
    words = []
    for t in texts:
        for w in t.split():
            # 2글자 이상, 보존 키워드 아닌 일반 단어
            if len(w) >= 2 and not is_preserve_word(w, cls):
                words.append(w)
    # 빈도 상위 500 단어를 어휘 사전으로
    common = [w for w, _ in Counter(words).most_common(500)]
    class_vocabs[cls] = common
    print(f"  {cls}: 어휘 {len(common)}개")
print()


# ──────────────────────────────────────────────────
# 5. 위협 4클래스 증강
# ──────────────────────────────────────────────────
print("=" * 60)
print("4. 위협 4클래스 증강 (클래스당 3,000건)")
print("=" * 60)

augmented_rows = []

for cls in ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"]:
    originals = train_dedup[train_dedup["class"] == cls]["conversation"].tolist()
    n_orig = len(originals)
    n_needed = TARGET_PER_CLASS - n_orig

    print(f"\n  [{cls}] 원본: {n_orig}건 → 증강 필요: {n_needed}건")

    # 원본 먼저 추가
    for conv in originals:
        augmented_rows.append({"class": cls, "conversation": conv})

    # 증강 생성
    generated = set()
    attempts = 0
    max_attempts = n_needed * 5

    while len(generated) < n_needed and attempts < max_attempts:
        src = random.choice(originals)
        aug = augment_text(src, cls, class_vocabs[cls])

        # 원본과 동일하거나 이미 생성된 것은 스킵
        if aug not in generated and aug not in originals:
            generated.add(aug)
            augmented_rows.append({"class": cls, "conversation": aug})

        attempts += 1

    actual = len(generated)
    print(f"  [{cls}] 증강 완료: {actual}건 생성 (총 {n_orig + actual}건)")

    if actual < n_needed:
        print(f"  ⚠️ 목표 미달: {n_needed - actual}건 부족")


# ──────────────────────────────────────────────────
# 6. 일반 대화 3,000건 맞추기
# ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 일반 대화 3,000건 구성")
print("=" * 60)

normal_convs = normal_all["conversation"].tolist()
n_normal = len(normal_convs)

# 일반 대화용 어휘 사전
normal_words = []
for t in normal_convs:
    for w in t.split():
        if len(w) >= 2:
            normal_words.append(w)
normal_vocab = [w for w, _ in Counter(normal_words).most_common(500)]

normal_rows = []
for conv in normal_convs:
    normal_rows.append({"class": "일반 대화", "conversation": conv})

if n_normal < TARGET_PER_CLASS:
    n_needed = TARGET_PER_CLASS - n_normal
    print(f"  현재: {n_normal}건, 추가 필요: {n_needed}건")

    generated = set()
    attempts = 0
    while len(generated) < n_needed and attempts < n_needed * 5:
        src = random.choice(normal_convs)
        aug = augment_text(src, "일반 대화", normal_vocab)
        if aug not in generated and aug not in normal_convs:
            generated.add(aug)
            normal_rows.append({"class": "일반 대화", "conversation": aug})
        attempts += 1

    print(f"  증강 완료: {len(generated)}건 추가 (총 {n_normal + len(generated)}건)")
else:
    # 3,000건 초과 시 랜덤 샘플
    normal_rows = normal_rows[:TARGET_PER_CLASS]
    print(f"  {n_normal}건 → {TARGET_PER_CLASS}건으로 샘플링")


# ──────────────────────────────────────────────────
# 7. 병합 & 저장
# ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 병합 & 저장")
print("=" * 60)

all_rows = augmented_rows + normal_rows
baseline_df = pd.DataFrame(all_rows)

# idx 부여
baseline_df = baseline_df.reset_index(drop=True)
baseline_df.insert(0, "idx", range(len(baseline_df)))

# 셔플
baseline_df = baseline_df.sample(frac=1, random_state=42).reset_index(drop=True)
baseline_df["idx"] = range(len(baseline_df))

# 저장
output_path = "/Users/goeunlee/aiffel/DLthon/baseline.csv"
baseline_df.to_csv(output_path, index=False)

print(f"\n최종 baseline.csv: {len(baseline_df)}건")
print(baseline_df["class"].value_counts().to_string())
print(f"\n저장 완료: {output_path}")


# ──────────────────────────────────────────────────
# 8. 품질 검증
# ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 품질 검증")
print("=" * 60)

# 길이 통계
for cls in baseline_df["class"].unique():
    texts = baseline_df[baseline_df["class"] == cls]["conversation"]
    lens = texts.apply(len)
    turns = texts.apply(lambda x: len(x.split("\n")))
    print(f"  [{cls}] 길이: {lens.mean():.0f}자(평균), 턴: {turns.mean():.1f}(평균)")

# 중복 체크
n_dup = baseline_df.duplicated(subset="conversation").sum()
print(f"\n  중복 건수: {n_dup}건")

# 일반 대화 금지 단어 체크
normal_texts = baseline_df[baseline_df["class"] == "일반 대화"]["conversation"]
ban_issues = 0
for text in normal_texts:
    for bw in BANNED_WORDS:
        if bw in text:
            ban_issues += 1
            break
print(f"  일반 대화 금지 단어 포함: {ban_issues}건")

print("\n✅ baseline.csv 생성 완료!")
