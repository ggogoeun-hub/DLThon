import sys
import io

# stdout을 UTF-8 파일로 직접 리다이렉트 (한국어 깨짐 방지)
sys.stdout = open('eda_results.txt', 'w', encoding='utf-8')

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 시각화 비활성화 (headless 실행)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
train_df = pd.read_csv('train.csv')

# ============================================================
# 1. 길이 및 구조 (Structure) 분석
# ============================================================
def analyze_structure(df):
    df['char_len'] = df['conversation'].apply(len)
    df['word_len'] = df['conversation'].apply(lambda x: len(x.split()))
    df['turn_count'] = df['conversation'].apply(
        lambda x: len([s for s in x.split('\n') if s.strip()])
    )
    return df

df = analyze_structure(train_df)

print("=" * 60)
print("[1-1] 문장/단어 길이 분포 분석")
print("=" * 60)
len_stats = df.groupby('class')[['char_len', 'word_len']].agg(
    ['mean', 'median', 'min', 'max']
).round(1)
print(len_stats.to_string())

print("\n### [1-1 인사이트 및 Action Item] ###")
print("1. 특정 클래스만 글자 수/단어 수가 유독 길거나 짧다면, 길이에 의한 편향(Bias)이 발생할 수 있습니다.")
print("2. 글자 수 대비 단어 수가 적다면(어휘 밀도↑), 특정 말투/단어가 반복되는지 점검이 필요합니다.")
print("3. 합성 데이터(일반 대화) 생성 시, 이 통계 범위 안에서 길이를 조절하세요.")

print("\n" + "=" * 60)
print("[1-2] 대화 발화 횟수 (Turns) 분석")
print("=" * 60)
turn_stats = df.groupby('class')['turn_count'].agg(
    ['mean', 'median', 'min', 'max']
).round(1)
print(turn_stats.to_string())

print("\n### [1-2 인사이트 및 Action Item] ###")
print("1. 특정 클래스만 발화 횟수가 유독 많거나 적은지 확인하세요.")
print("2. 합성 데이터 생성 시, 모델이 학습할 적절한 '대화의 호흡'을 이 중간값 수준으로 설정하는 것이 안전합니다.")

# ============================================================
# 2. 어휘 및 키워드 (Vocabulary) 분석
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_top_tfidf_words(df, n_top=10):
    results = {}
    classes = df['class'].unique()
    class_corpus = [" ".join(df[df['class'] == c]['conversation']) for c in classes]
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(class_corpus)
    words = vectorizer.get_feature_names_out()
    for i, class_name in enumerate(classes):
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-n_top:][::-1]
        results[class_name] = [words[idx] for idx in top_indices]
    return pd.DataFrame(results)

print("\n" + "=" * 60)
print("[2-1] TF-IDF 상위 키워드 (클래스 대표 단어)")
print("=" * 60)
top_keywords = get_top_tfidf_words(df)
print(top_keywords.to_string())

print("\n### [2-1 인사이트 및 Action Item] ###")
print("1. 각 클래스의 TF-IDF 상위 키워드를 합성 데이터 생성 시 '반드시 포함할 단어'로 활용하세요.")
print("2. 특정 클래스에만 등장하는 고유 키워드를 다른 클래스의 합성 데이터에서 '배제할 단어'로 관리하세요.")

def get_top_ngrams(df, n_gram=2, top_k=10):
    results = {}
    classes = df['class'].unique()
    for class_name in classes:
        corpus = df[df['class'] == class_name]['conversation']
        vec = CountVectorizer(ngram_range=(n_gram, n_gram)).fit(corpus)
        bag = vec.transform(corpus)
        sum_words = bag.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
        results[class_name] = words_freq
    return results

print("\n" + "=" * 60)
print("[2-2] N-gram 분석 (Bi-gram & Tri-gram)")
print("=" * 60)

for n in [2, 3]:
    print(f"\n--- {n}-gram (상위 10개) ---")
    ngrams = get_top_ngrams(df, n_gram=n)
    for cls, items in ngrams.items():
        print(f"  [{cls}]: {', '.join(f'{w}({c})' for w,c in items)}")

print("\n### [2-2 인사이트 및 Action Item] ###")
print("1. '협박' 클래스에서 유독 자주 쓰이는 '도구/행위' 조합이 보이나요?")
print("2. '직장 내 괴롭힘'에서 '상급자 호칭 + 명령조'의 조합이 어떻게 나타나는지 확인하세요.")
print("3. 이 N-gram들을 '일반 대화' 합성 생성 시 '배제 키워드'로 활용하여 모델 변별력을 높이세요.")

def calculate_ttr(text):
    tokens = text.split()
    if len(tokens) == 0: return 0
    return len(set(tokens)) / len(tokens)

print("\n" + "=" * 60)
print("[2-3] 어휘 다양성 (TTR) 분석")
print("=" * 60)
df['ttr'] = df['conversation'].apply(calculate_ttr)
ttr_summary = df.groupby('class')['ttr'].agg(['mean', 'median', 'std']).round(3)
print(ttr_summary.to_string())

print("\n### [2-3 인사이트 및 Action Item] ###")
print("1. TTR이 낮은 클래스는 뻔한 단어만 반복 사용합니다. 증강 시 동의어 교체 폭을 넓히세요.")
print("2. TTR이 높은 클래스는 다양한 표현이 사용됩니다. 합성 데이터도 어휘를 다양하게 구성하세요.")

# ============================================================
# 3. 클래스 관계 (Semantic) 분석
# ============================================================
from sklearn.metrics.pairwise import cosine_similarity

classes = df['class'].unique()
class_documents = [" ".join(df[df['class'] == cls]['conversation']) for cls in classes]

vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = vectorizer.fit_transform(class_documents)
feature_names = vectorizer.get_feature_names_out()

print("\n" + "=" * 60)
print("[3-1] 클래스 간 코사인 유사도 분석")
print("=" * 60)

similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=classes, columns=classes).round(4)
print(similarity_df.to_string())

print("\n### 유사도 높은 클래스 쌍 순위 (헷갈림 위험 순) ###")
pairs = []
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        pairs.append((classes[i], classes[j], similarity_matrix[i, j]))
pairs.sort(key=lambda x: -x[2])
for rank, (c1, c2, score) in enumerate(pairs, 1):
    flag = " ⚠️ 최우선 경계 강화 필요" if rank == 1 else ""
    print(f"  {rank}위: [{c1}] ↔ [{c2}] = {score:.4f}{flag}")

print("\n### [3-1 인사이트 및 Action Item] ###")
print("1. 유사도가 가장 높은 클래스 쌍은 모델이 가장 많이 혼동합니다.")
print("2. 해당 쌍 사이의 경계를 강화하는 전략(Loss 가중치, Hard Negative 학습)을 수립하세요.")

print("\n" + "=" * 60)
print("[3-2] 중복 단어 (Overlap / Hard Negatives) 분석")
print("=" * 60)

top_n = 50
class_top_words = {}
for i, cls in enumerate(classes):
    row = tfidf_matrix.getrow(i).toarray()[0]
    top_indices = row.argsort()[-top_n:][::-1]
    class_top_words[cls] = set(feature_names[idx] for idx in top_indices)

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

overlap_matrix = np.zeros((len(classes), len(classes)))
for i, c1 in enumerate(classes):
    for j, c2 in enumerate(classes):
        overlap_matrix[i, j] = jaccard_similarity(class_top_words[c1], class_top_words[c2])
overlap_df = pd.DataFrame(overlap_matrix, index=classes, columns=classes).round(4)
print(overlap_df.to_string())

print(f"\n### 클래스 쌍별 공통 빈출 단어 (상위 {top_n}개 기준) ###")
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        c1, c2 = classes[i], classes[j]
        common = sorted(class_top_words[c1] & class_top_words[c2])
        if common:
            print(f"  [{c1}] ↔ [{c2}] ({len(common)}개): {', '.join(common)}")

print("\n### [3-2 인사이트 및 Action Item] ###")
print("1. 공통 단어가 '일반 대화'에 포함되었을 때 모델이 '위협'으로 오해하지 않도록,")
print("   공통 단어가 포함되었으나 무해한(Safe) 일반 대화를 합성해야 합니다. (핵심!)")
print("2. 예: '돈'이 '갈취'와 '일반 대화'에 모두 등장한다면,")
print("   '돈 빌려줘서 고마워' 같은 정상적 문맥의 합성 데이터가 필요합니다.")

# ============================================================
# 4. 언어적 특징 (Linguistic) 분석
# ============================================================
try:
    from konlpy.tag import Okt
    okt = Okt()
    USE_KONLPY = True
except ImportError:
    USE_KONLPY = False

def extract_pos_distribution(text):
    if not USE_KONLPY:
        return {'Noun': 0, 'Verb': 0, 'Adjective': 0, 'Exclamation': 0}
    pos_tags = okt.pos(text)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total = len(pos_tags)
    if total == 0:
        return {'Noun': 0, 'Verb': 0, 'Adjective': 0, 'Exclamation': 0}
    return {
        'Noun': pos_counts.get('Noun', 0) / total,
        'Verb': pos_counts.get('Verb', 0) / total,
        'Adjective': pos_counts.get('Adjective', 0) / total,
        'Exclamation': pos_counts.get('Exclamation', 0) / total,
    }

print("\n" + "=" * 60)
print("[4-1] 품사(POS) 분포 분석")
print("=" * 60)

if USE_KONLPY:
    print("형태소 분석 중... (수 분 소요 가능)")
    pos_df = df['conversation'].apply(extract_pos_distribution).apply(pd.Series)
    df_pos = pd.concat([df[['class']], pos_df], axis=1)
    pos_summary = df_pos.groupby('class')[['Noun', 'Verb', 'Adjective', 'Exclamation']].mean().round(4)
    print(pos_summary.to_string())
else:
    print("konlpy 미설치 — 품사 분석을 건너뜁니다.")

print("\n### [4-1 인사이트 및 Action Item] ###")
print("1. '협박' 클래스에 동사 비율이 높다면 → 명령조('죽여', '내놔') 말투가 특징적입니다.")
print("2. '괴롭힘' 클래스에 형용사 비율이 높다면 → 부정적 평가('못난', '쓸모없는') 말투가 특징적입니다.")
print("3. 감탄사 비율이 특정 클래스에 쏠려 있다면 → 합성 데이터 생성 시 감탄사 사용 빈도를 조절하세요.")

def analyze_tone_and_profanity(text):
    polite_patterns = ['습니다', '합니다', '세요', '에요', '해요', '네요', '군요', '나요', '죠']
    polite_count = sum(text.count(p) for p in polite_patterns)
    casual_patterns = ['해', '야', '냐', '라', '거든', '잖아', '는데', '인데']
    casual_count = sum(text.count(p) for p in casual_patterns)
    profanity_words = ['새끼', '미친', '죽여', '시발', '씨발', '병신', '지랄', '닥쳐', '패버릴']
    profanity_count = sum(1 for w in profanity_words if w in text)
    return pd.Series({
        'polite_count': polite_count,
        'casual_count': casual_count,
        'profanity_count': profanity_count,
    })

print("\n" + "=" * 60)
print("[4-2] 존칭/비속어 사용 빈도 분석")
print("=" * 60)
tone_df = df['conversation'].apply(analyze_tone_and_profanity)
df_tone = pd.concat([df[['class']], tone_df], axis=1)
tone_summary = df_tone.groupby('class')[['polite_count', 'casual_count', 'profanity_count']].mean().round(2)
print(tone_summary.to_string())

print("\n### [4-2 인사이트 및 Action Item] ###")
print("1. '직장 내 괴롭힘'에 존댓말 비율이 높다면 → 수직적 관계의 괴롭힘 특징입니다.")
print("2. 욕설 빈도가 높은 클래스 확인 → 합성 데이터에 '욕설 없는 갈등 상황'을 추가하여 모델의 욕설 의존도를 낮추세요.")
print("3. '일반 대화' 합성 시 존댓말/반말 비중을 트레인 셋의 비율과 일치시켜 이질감을 제거하세요.")

# ============================================================
# 5. 품질 및 노이즈 (Quality) 분석
# ============================================================
def analyze_noise_features(text):
    exclamation_count = text.count('!')
    question_count = text.count('?')
    emoticon_pattern = re.compile(r'[ㅋㅎㅠㅜ]+')
    emoticons = emoticon_pattern.findall(text)
    emoticon_len = sum(len(e) for e in emoticons)
    ellipsis_count = len(re.findall(r'\.{2,}', text))
    return pd.Series({
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'emoticon_len': emoticon_len,
        'ellipsis_count': ellipsis_count,
    })

print("\n" + "=" * 60)
print("[5-1] 특수문자/문장부호 빈도 분석")
print("=" * 60)
noise_df = df['conversation'].apply(analyze_noise_features)
df_noise = pd.concat([df[['class']], noise_df], axis=1)
noise_summary = df_noise.groupby('class')[['exclamation_count', 'question_count', 'emoticon_len', 'ellipsis_count']].mean().round(2)
print(noise_summary.to_string())

print("\n### [5-1 인사이트 및 Action Item] ###")
print("1. 특정 클래스에만 느낌표나 말끝 흐림이 과도하게 몰려 있다면, 모델이 내용이 아닌 부호에 의존할 위험이 있습니다.")
print("2. 학습 데이터에 'ㅋㅋ', 'ㅠㅠ'가 아예 없다면, 합성 데이터 생성 시에도 넣지 않아야 합니다.")

print("\n" + "=" * 60)
print("[5-2] 데이터 중복 및 유사도 분석")
print("=" * 60)

duplicated_rows = df.duplicated(subset=['conversation'])
num_duplicates = duplicated_rows.sum()
print(f"\n[a] 완전 동일 중복 건수: {num_duplicates} 건")
if num_duplicates > 0:
    print("   -> 학습 시 중복 제거(deduplication)를 고려하세요.")
else:
    print("   -> 완전 중복은 없습니다.")

print("\n[b] 준-중복(Near-Duplicate) 탐지 (코사인 유사도 >= 0.95)...")
tfidf_dup = TfidfVectorizer(max_features=5000)
tfidf_dup_matrix = tfidf_dup.fit_transform(df['conversation'])
sim_matrix = cosine_similarity(tfidf_dup_matrix)
np.fill_diagonal(sim_matrix, 0)
near_dup_pairs = []
threshold = 0.95
rows_idx, cols_idx = np.where(np.triu(sim_matrix) >= threshold)
for r, c in zip(rows_idx, cols_idx):
    near_dup_pairs.append((r, c, sim_matrix[r, c]))
print(f"   유사도 {threshold} 이상인 준-중복 쌍: {len(near_dup_pairs)} 건")
if len(near_dup_pairs) > 0:
    print("   (상위 5개 예시)")
    for idx, (r, c, score) in enumerate(sorted(near_dup_pairs, key=lambda x: -x[2])[:5]):
        print(f"   [{idx+1}] idx {r} <-> idx {c} | 유사도: {score:.4f} | 클래스: {df.iloc[r]['class']} <-> {df.iloc[c]['class']}")
    print("   -> '사실상 동일한' 데이터가 발견되었습니다. 과적합 방지를 위해 제거/통합을 고려하세요.")
else:
    print("   -> 준-중복 데이터가 없습니다.")

print("\n" + "=" * 60)
print("[5-3] 불용어(조사/기능어) 클래스별 편향 분석")
print("=" * 60)

stopwords_list = [
    '은', '는', '이', '가', '을', '를', '에', '에서', '의', '도',
    '로', '으로', '와', '과', '한', '하고', '라고', '이라고',
    '인데', '는데', '거든', '니까', '잖아', '다고', '래', '데',
]

def count_stopwords(text, stopwords):
    tokens = text.split()
    counts = {}
    for sw in stopwords:
        cnt = sum(1 for t in tokens if t.endswith(sw) and len(t) > len(sw))
        counts[sw] = cnt
    return pd.Series(counts)

sw_counts = df['conversation'].apply(lambda x: count_stopwords(x, stopwords_list))
df_sw = pd.concat([df[['class']], sw_counts], axis=1)
sw_summary = df_sw.groupby('class')[stopwords_list].mean()
print(sw_summary.round(1).to_string())

print("\n### 조사/기능어 클래스별 편향도 (변동계수 CV = std/mean) ###")
print("(CV가 높을수록 특정 클래스에 쏠려 있어 모델이 의존할 위험이 큼)\n")
cv_scores = sw_summary.std() / sw_summary.mean()
cv_scores = cv_scores.sort_values(ascending=False)
for sw, cv in cv_scores.items():
    flag = " ⚠️ 편향 주의" if cv > 0.3 else ""
    print(f"  '{sw}': CV = {cv:.3f}{flag}")

print("\n### [5-3 인사이트 및 Action Item] ###")
print("1. CV가 높은 조사/기능어는 모델이 해당 단어에 의존해 분류할 위험이 있습니다.")
print("2. 전처리 시 해당 기능어를 불용어 리스트에 추가하여 제거할지, 그대로 둘지 결정하세요.")
print("3. 합성 데이터 생성 시, 클래스별 조사 빈도를 트레인 셋과 유사하게 맞추면 이질감을 줄일 수 있습니다.")

# ============================================================
# 6. 상황 및 맥락 (Context & Flow) 분석
# ============================================================
def analyze_domain_context(text):
    company_words = ['회사', '팀장', '과장', '대리', '사원', '업무', '결재', '출근', '퇴근', '회의']
    school_words = ['학교', '선생', '학생', '교실', '공부', '성적', '숙제', '학원', '수업', '점수']
    street_words = ['골목', '길거리', '거리', '공원', '편의점', '술집', '노래방', '클럽', '카페', '지하철']
    return pd.Series({
        'company_context': sum(1 for w in company_words if w in text),
        'school_context': sum(1 for w in school_words if w in text),
        'street_context': sum(1 for w in street_words if w in text),
    })

print("\n" + "=" * 60)
print("[6-1] 도메인 별 키워드 쏠림 분석")
print("=" * 60)

domain_df = df['conversation'].apply(analyze_domain_context)
df_domain = pd.concat([df[['class']], domain_df], axis=1)
domain_summary = df_domain.groupby('class')[['company_context', 'school_context', 'street_context']].mean().round(3)
print(domain_summary.to_string())

print("\n### [6-1 인사이트 및 Action Item] ###")
print("1. '직장 내 괴롭힘'에만 회사 단어가 쏠려 있다면, 모델이 '대리님'만 보고 괴롭힘으로 분류할 수 있습니다.")
print("2. 이를 방지하기 위해 '대리님, 이번 결재 건 관련해서...' 같은 정상적인 직장 일상 대화를 합성 데이터로 반드시 생성하세요.")
print("3. 학교/길거리 도메인도 동일한 원리로, 해당 키워드가 포함된 무해한 일반 대화를 설계해야 합니다.")

def analyze_conversation_flow(text):
    threat_words = ['죽여', '맞을래', '돈', '내놔', '닥쳐', '미친', '새끼', '씨발', '경찰', '신고', '죽어']
    defense_words = ['살려', '제발', '그만', '왜요', '미안', '죄송', '잘못', '용서', '하지마', '싫어']
    turns = [t.strip() for t in text.split('\n') if t.strip()]
    total_turns = len(turns)
    if total_turns < 3:
        return pd.Series({
            'threat_early': 0, 'threat_mid': 0, 'threat_late': 0,
            'defense_early': 0, 'defense_mid': 0, 'defense_late': 0,
        })
    chunk = total_turns // 3
    sections = [
        " ".join(turns[:chunk]),
        " ".join(turns[chunk:chunk * 2]),
        " ".join(turns[chunk * 2:]),
    ]
    result = {}
    for idx, label in enumerate(['early', 'mid', 'late']):
        result[f'threat_{label}'] = sum(1 for w in threat_words if w in sections[idx])
        result[f'defense_{label}'] = sum(1 for w in defense_words if w in sections[idx])
    return pd.Series(result)

print("\n" + "=" * 60)
print("[6-2] 발화 순서별 감정/의도 변화 분석")
print("=" * 60)

flow_df = df['conversation'].apply(analyze_conversation_flow)
df_flow = pd.concat([df[['class']], flow_df], axis=1)

threat_summary = df_flow.groupby('class')[['threat_early', 'threat_mid', 'threat_late']].mean().round(3)
defense_summary = df_flow.groupby('class')[['defense_early', 'defense_mid', 'defense_late']].mean().round(3)

print("\n--- 공격(위협) 어조 변화 (초반 → 중반 → 후반) ---")
print(threat_summary.to_string())
print("\n--- 방어(구원) 어조 변화 (초반 → 중반 → 후반) ---")
print(defense_summary.to_string())

print("\n### [6-2 인사이트 및 Action Item] ###")
print("1. 위협이 초반부터 강하다면 → '직접 협박' 패턴입니다.")
print("2. 후반으로 갈수록 위협이 짙어진다면 → '간접 협박' 또는 '갈취' 패턴입니다.")
print("3. 방어 단어가 후반에 급증한다면 → 피해자의 반응 패턴으로, 합성 데이터에 이 흐름을 반영하세요.")
print("4. 처음엔 친절하다가 마지막에 위협적으로 변하는 '간접 협박' 시나리오를 증강 전략에 포함하세요.")

print("\n" + "=" * 60)
print("===  EDA 분석 완료  ===")
print("=" * 60)
