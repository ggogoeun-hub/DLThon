"""
B03 데이터셋 품질 검증 — baseline_B03.csv 단일 대상
====================================================
검증 항목:
  1. B03 일반대화 vs 테스트 — 도메인(내용) 유사도
  2. 테스트 데이터 일반대화 후보 분석
  3. B03 일반대화 중 위협 오분류 가능 경계 사례
  4. B03 위협대화 중 일반 혼동 가능 패턴
  5. B03 보정 충분성 + 숏컷 분리 가능성 + 개선 제안
"""

import os, re, random
import numpy as np
import pandas as pd
from collections import Counter

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════
# 0. 데이터 로딩
# ══════════════════════════════════════════════════════
DATA_DIR = "aiffel-d-lthon-dktc-online-17"

baseline = pd.read_csv("baseline_B03.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

# B03 클래스별 분리
b03_normal_raw = baseline[baseline["class"] == "일반 대화"]
b03_threat_raw = baseline[baseline["class"] != "일반 대화"]

b03_normal = b03_normal_raw["conversation"].apply(preprocess).tolist()
b03_threat = b03_threat_raw["conversation"].apply(preprocess).tolist()
b03_threat_cls = b03_threat_raw["class"].tolist()
test_texts = test_df["conversation"].apply(preprocess).tolist()

# 위협 클래스별
threat_classes = ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"]
b03_by_class = {}
for cls in threat_classes:
    b03_by_class[cls] = baseline[baseline["class"] == cls]["conversation"].apply(preprocess).tolist()

print(f"baseline_B03.csv 로딩 완료:")
print(f"  전체: {len(baseline)}건")
for cls in threat_classes + ["일반 대화"]:
    n = len(baseline[baseline["class"] == cls])
    print(f"    {cls}: {n}건")
print(f"  테스트: {len(test_texts)}건")

# ══════════════════════════════════════════════════════
# 공통 도구
# ══════════════════════════════════════════════════════
POLITE_MARKERS = ["요", "습니다", "세요", "겠습니다"]
CASUAL_CHARS = list("ㅋㅎㅠㅜ")

THREAT_KEYWORDS = [
    "죽여", "죽어", "죽이", "죽을", "죽고", "때려", "때릴", "맞을래",
    "내놔", "닥쳐", "뒤져", "뒤질", "패버", "칼", "찔러", "살려",
    "시키는 대로", "돈 내", "돈줘", "고소", "경찰", "신고",
    "불태워", "가만 안", "가만두지", "협박",
]

DOMAINS = {
    "직장/회사": ["과장", "부장", "팀장", "대리", "사원", "회의", "출근", "퇴근", "야근", "업무", "프로젝트", "보고서", "직장"],
    "학교/학업": ["학교", "수업", "과제", "시험", "선생", "교수", "학원", "공부", "성적"],
    "가족/관계": ["엄마", "아빠", "부모", "가족", "동생", "형", "누나", "오빠", "언니", "할머니", "할아버지"],
    "서비스/소비": ["주문", "배달", "고객", "식당", "가게", "알바", "계산", "결제", "환불", "택배", "쇼핑"],
    "금전": ["돈", "만원", "천원", "계좌", "빌려", "월급", "이자", "갚", "송금", "카드"],
    "친구/일상": ["친구", "놀자", "만나", "카페", "영화", "게임", "여행", "피시방", "노래방"],
    "음식/맛집": ["맛있", "식당", "라면", "치킨", "피자", "밥", "점심", "저녁", "아침"],
    "날씨/계절": ["날씨", "비", "봄", "여름", "가을", "겨울", "눈", "더워", "추워", "벚꽃"],
    "연애/이별": ["사귀", "헤어", "사랑", "남자친구", "여자친구", "좋아해", "고백", "데이트", "남친", "여친"],
    "감정/갈등": ["미안", "사과", "화나", "짜증", "속상", "힘들", "스트레스", "열받"],
}


def compute_features(text):
    return {
        "period": text.count("."),
        "excl": text.count("!"),
        "ques": text.count("?"),
        "polite": sum(text.count(m) for m in POLITE_MARKERS),
        "casual": sum(text.count(c) for c in CASUAL_CHARS),
        "length": len(text),
        "words": len(text.split()),
        "threat_kw": sum(1 for kw in THREAT_KEYWORDS if kw in text),
    }


def classify_domains(text):
    found = []
    for domain, keywords in DOMAINS.items():
        if any(kw in text for kw in keywords):
            found.append(domain)
    return found if found else ["기타"]


def feat_stats(texts):
    if not texts:
        return {}
    feats = [compute_features(t) for t in texts]
    result = {}
    for key in feats[0]:
        vals = [f[key] for f in feats]
        result[key] = {"mean": np.mean(vals), "std": np.std(vals)}
    return result


def divider(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════
# 검증 1: B03 일반대화 vs 테스트 — 도메인 유사도
# ══════════════════════════════════════════════════════
divider("검증 1: B03 일반대화 vs 테스트 — 도메인(내용) 유사도")

def domain_dist(texts):
    all_d = []
    for t in texts:
        all_d.extend(classify_domains(t))
    return Counter(all_d), len(texts)

test_dom, test_n = domain_dist(test_texts)
b03n_dom, b03n_n = domain_dist(b03_normal)

print(f"\n  {'도메인':12s} | {'테스트':>12s} | {'B03 일반':>12s} | 갭")
print(f"  {'-'*55}")
all_domains = sorted(set(list(test_dom.keys()) + list(b03n_dom.keys())))
for d in all_domains:
    t_pct = test_dom.get(d, 0) / test_n * 100
    b_pct = b03n_dom.get(d, 0) / b03n_n * 100
    gap = t_pct - b_pct
    flag = " !!!" if abs(gap) > 10 else (" !" if abs(gap) > 5 else "")
    print(f"  {d:12s} | {t_pct:10.1f}% | {b_pct:10.1f}% | {gap:+.1f}%p{flag}")

# 어휘 겹침
test_vocab = set(w for t in test_texts for w in t.split())
b03n_vocab = set(w for t in b03_normal for w in t.split())
overlap = test_vocab & b03n_vocab
print(f"\n  어휘 겹침: {len(overlap):,} / {len(test_vocab):,} 테스트 어휘 ({len(overlap)/len(test_vocab)*100:.1f}%)")

# 테스트에만 있는 빈출 어휘
test_wf = Counter(w for t in test_texts for w in t.split())
b03_all_vocab = set(w for t in (b03_normal + b03_threat) for w in t.split())
test_only_freq = sorted(
    [(w, c) for w, c in test_wf.items() if w not in b03_all_vocab and c >= 3],
    key=lambda x: -x[1]
)[:20]
if test_only_freq:
    print(f"\n  테스트에만 등장하는 빈출 어휘 (B03 전체에 없음):")
    for w, c in test_only_freq:
        print(f"    '{w}' ({c}회)")


# ══════════════════════════════════════════════════════
# 검증 2: 테스트 데이터 분석
# ══════════════════════════════════════════════════════
divider("검증 2: 테스트 데이터 일반대화 후보 분석")

test_ft = feat_stats(test_texts)
print(f"\n  [테스트 전체 문체 통계 (500건)]")
print(f"  {'특성':10s} {'평균':>8s} {'표준편차':>8s}")
print(f"  {'-'*30}")
for key in ["period", "excl", "ques", "polite", "casual", "length", "words", "threat_kw"]:
    print(f"  {key:10s} {test_ft[key]['mean']:8.2f} {test_ft[key]['std']:8.2f}")

# 위협 키워드 유무로 분류
test_no_kw = [t for t in test_texts if compute_features(t)["threat_kw"] == 0]
test_with_kw = [t for t in test_texts if compute_features(t)["threat_kw"] > 0]
print(f"\n  위협 키워드 없는 텍스트 (일반 추정): {len(test_no_kw)}건 ({len(test_no_kw)/500*100:.0f}%)")
print(f"  위협 키워드 있는 텍스트: {len(test_with_kw)}건 ({len(test_with_kw)/500*100:.0f}%)")

# 각 그룹의 문체 비교
if test_no_kw:
    ft_no = feat_stats(test_no_kw)
    ft_with = feat_stats(test_with_kw) if test_with_kw else {}
    print(f"\n  {'특성':10s} | {'kw없음(일반추정)':>14s} | {'kw있음(위협추정)':>14s}")
    print(f"  {'-'*45}")
    for key in ["period", "excl", "ques", "polite", "length"]:
        v1 = f"{ft_no[key]['mean']:.2f}"
        v2 = f"{ft_with[key]['mean']:.2f}" if ft_with else "N/A"
        print(f"  {key:10s} | {v1:>14s} | {v2:>14s}")

# 테스트 샘플 10건
print(f"\n  [테스트 랜덤 샘플 10건]")
sample_idx = random.sample(range(len(test_texts)), min(10, len(test_texts)))
for i in sample_idx:
    t = test_texts[i]
    f = compute_features(t)
    domains = classify_domains(t)
    snippet = t[:120] + ("..." if len(t) > 120 else "")
    kw_flag = f" [위협kw={f['threat_kw']}]" if f['threat_kw'] > 0 else ""
    print(f"    [{i:3d}] 도메인={domains} .={f['period']} 존댓={f['polite']} len={f['length']}{kw_flag}")
    print(f"          {snippet}")


# ══════════════════════════════════════════════════════
# 검증 3: B03 일반대화 중 경계 사례 식별
# ══════════════════════════════════════════════════════
divider("검증 3: B03 일반대화 중 위협 오분류 가능 경계 사례")

# 3a: 위협 키워드 포함
print(f"\n  [위협 키워드 포함 일반대화]")
edge_kw = []
for i, t in enumerate(b03_normal):
    found = [kw for kw in THREAT_KEYWORDS if kw in t]
    if found:
        edge_kw.append((i, t, found))
print(f"  발견: {len(edge_kw)}건 / {len(b03_normal)}건")
for idx, text, kws in edge_kw[:15]:
    snippet = text[:100] + ("..." if len(text) > 100 else "")
    print(f"    [{idx:4d}] 키워드={kws}")
    print(f"           {snippet}")

# 3b: 문체가 위협과 유사한 일반대화
threat_ft = feat_stats(b03_threat)
print(f"\n  [문체가 위협 데이터와 유사한 일반대화 (상위 10)]")

def threat_style_score(text):
    f = compute_features(text)
    tm = threat_ft
    score = 0
    for key in ["period", "polite", "excl"]:
        if tm[key]["mean"] > 0:
            score += min(f[key] / tm[key]["mean"], 1.5)
    if tm["length"]["mean"] > 0:
        score += min(f["length"] / tm["length"]["mean"], 1.2)
    if f["casual"] == 0:
        score += 0.5
    return score

scored = sorted(
    [(i, t, threat_style_score(t)) for i, t in enumerate(b03_normal)],
    key=lambda x: -x[2]
)
for idx, text, sim in scored[:10]:
    f = compute_features(text)
    snippet = text[:100] + ("..." if len(text) > 100 else "")
    print(f"    [{idx:4d}] 유사도={sim:.2f} .={f['period']} 존댓={f['polite']} !={f['excl']} len={f['length']}")
    print(f"           {snippet}")


# ══════════════════════════════════════════════════════
# 검증 4: B03 위협 중 일반 혼동 가능 패턴
# ══════════════════════════════════════════════════════
divider("검증 4: B03 위협 중 일반대화 혼동 가능 패턴")

# 위협 키워드 밀도가 낮은 위협 텍스트
low_density = []
for cls in threat_classes:
    for t in b03_by_class[cls]:
        f = compute_features(t)
        low_density.append((cls, t, f["threat_kw"], f["length"]))

zero_kw = [x for x in low_density if x[2] == 0]
one_kw = [x for x in low_density if x[2] <= 1]
print(f"\n  위협 키워드 0개인 위협 텍스트: {len(zero_kw)}건 / {len(low_density)}건 ({len(zero_kw)/len(low_density)*100:.1f}%)")
print(f"  위협 키워드 0~1개: {len(one_kw)}건 ({len(one_kw)/len(low_density)*100:.1f}%)")

print(f"\n  [위협 키워드 0개 — 일반대화와 혼동 가능 (상위 20)]")
for cls, text, kw_cnt, length in zero_kw[:20]:
    snippet = text[:120] + ("..." if len(text) > 120 else "")
    print(f"    [{cls[:6]}] len={length:3d} | {snippet}")

# 일상 주제 포함 위협
daily_kw = ["밥", "점심", "저녁", "커피", "카페", "영화", "날씨", "여행", "쇼핑", "게임"]
daily_threats = sorted(
    [(cls, t, sum(1 for k in daily_kw if k in t))
     for cls in threat_classes for t in b03_by_class[cls]
     if sum(1 for k in daily_kw if k in t) >= 2],
    key=lambda x: -x[2]
)
print(f"\n  일상 키워드 2개 이상 포함 위협: {len(daily_threats)}건")
for cls, text, cnt in daily_threats[:10]:
    snippet = text[:120] + ("..." if len(text) > 120 else "")
    print(f"    [{cls[:6]}] 일상kw={cnt} | {snippet}")


# ══════════════════════════════════════════════════════
# 검증 5: B03 보정 충분성 평가
# ══════════════════════════════════════════════════════
divider("검증 5: B03 보정 충분성 + 숏컷 분리 가능성")

# 5a: 전체 비교 테이블
groups = {
    "테스트 (500)": test_texts,
    "B03 일반 (3k)": b03_normal,
    "B03 위협 (12k)": b03_threat,
}
group_ft = {name: feat_stats(texts) for name, texts in groups.items()}

print(f"\n  [문체 특성 비교]")
print(f"  {'':16s} | {'마침표':>7s} | {'느낌표':>7s} | {'물음표':>7s} | {'존댓말':>7s} | {'캐주얼':>7s} | {'길이':>7s}")
print(f"  {'-'*76}")
for name, ft in group_ft.items():
    print(f"  {name:16s} | {ft['period']['mean']:7.2f} | {ft['excl']['mean']:7.2f} | "
          f"{ft['ques']['mean']:7.2f} | {ft['polite']['mean']:7.2f} | "
          f"{ft['casual']['mean']:7.2f} | {ft['length']['mean']:7.1f}")

# 5b: 보정 달성도
b03n_ft = group_ft["B03 일반 (3k)"]
test_ft_g = group_ft["테스트 (500)"]
threat_ft_g = group_ft["B03 위협 (12k)"]

checks = [
    ("마침표", "period"),
    ("느낌표", "excl"),
    ("물음표", "ques"),
    ("존댓말", "polite"),
    ("캐주얼", "casual"),
    ("길이",   "length"),
]

print(f"\n  [보정 달성도 — 테스트 목표 대비]")
print(f"  {'특성':8s} | {'B03일반':>8s} | {'테스트':>8s} | {'위협':>8s} | {'갭':>8s} | 판정")
print(f"  {'-'*60}")

for label, key in checks:
    b_val = b03n_ft[key]["mean"]
    t_val = test_ft_g[key]["mean"]
    th_val = threat_ft_g[key]["mean"]

    if key == "casual":
        gap_pct = b_val  # 0에 가까울수록 좋음
        if b_val < 0.1: verdict = "PASS"
        elif b_val < 0.5: verdict = "WARN"
        else: verdict = "FAIL"
        gap_str = f"{b_val:.2f}"
    else:
        gap_pct = (t_val - b_val) / t_val * 100 if t_val > 0 else 0
        if abs(gap_pct) < 20: verdict = "PASS"
        elif abs(gap_pct) < 40: verdict = "WARN"
        else: verdict = "FAIL"
        gap_str = f"{gap_pct:+.0f}%"

    print(f"  {label:8s} | {b_val:8.2f} | {t_val:8.2f} | {th_val:8.2f} | {gap_str:>8s} | {verdict}")

# 5c: 숏컷 분리 가능성 (Cohen's d)
print(f"\n  [숏컷 위험도 — Cohen's d (B03 일반 vs 위협)]")
print(f"  {'특성':8s} | {'일반 평균':>10s} | {'위협 평균':>10s} | {'Cohen d':>8s} | 위험도")
print(f"  {'-'*58}")

for label, key in checks:
    if key == "casual":
        continue
    n_vals = [compute_features(t)[key] for t in b03_normal]
    t_vals = [compute_features(t)[key] for t in b03_threat]
    n_m, n_s = np.mean(n_vals), np.std(n_vals)
    t_m, t_s = np.mean(t_vals), np.std(t_vals)
    pooled = np.sqrt((n_s**2 + t_s**2) / 2) if (n_s + t_s) > 0 else 1
    d = abs(t_m - n_m) / pooled

    if d < 0.2: risk = "LOW (Good)"
    elif d < 0.5: risk = "MEDIUM"
    elif d < 0.8: risk = "HIGH"
    else: risk = "CRITICAL!"
    print(f"  {label:8s} | {n_m:10.2f} | {t_m:10.2f} | {d:8.2f} | {risk}")


# ══════════════════════════════════════════════════════
# 종합 개선 제안
# ══════════════════════════════════════════════════════
divider("종합 개선 제안")

print("""
  [우선순위 1 — CRITICAL] 마침표 보정 강화
  현 fix_style(): 35% 확률 턴 단위 주입 → 실제 달성치 확인 필요
  제안: 결정적 주입 — 텍스트당 목표를 random.gauss(5.0, 1.5)로 잡고
        턴별 순회하며 부족분 채우기. '요', '다' 종결 턴도 대상 포함.

  [우선순위 2 — CRITICAL] 존댓말 변환 패턴 확장
  현재 6개 어미만 → 추가 필요:
    잖아→잖아요, 네→네요, 래→래요, 인데→인데요,
    거야→거예요, 좋아→좋아요, 그래→그래요
  확률 25% → 40%로 상향

  [우선순위 3 — HIGH] 느낌표 직접 주입
  현재 미대응. 감탄사(진짜, 와, 대박) 뒤 느낌표 추가 또는
  텍스트당 1개 랜덤 삽입.

  [우선순위 4 — MEDIUM] 물음표 보완
  의문사(뭐, 어디, 언제, 왜, 어떻게) 포함 턴에 물음표 없으면 추가.

  [우선순위 5 — MEDIUM] 도메인 다양성
  테스트 대비 부족한 도메인(연애/이별, 감정/갈등, 서비스/소비) 보충.

  [참고] 캐주얼 마커 제거는 잘 작동 중. 유지.
""")

print("검증 완료.")
