"""EXP-C: 슬라이딩 윈도우 코히런스 스코어링"""
import torch
from transformers import AutoTokenizer, AutoModel
import warnings; warnings.filterwarnings('ignore')
from exp_turn_common import *

print('='*60)
print('EXP-C: 슬라이딩 윈도우 코히런스')
print('='*60)

MODEL_NAME = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME); model.eval()

WINDOW = 5  # 토큰 윈도우 크기
rows = load_data()
all_p, all_r, all_f = [], [], []

for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual_boundaries(row['conversation'])
    words = flat.split()

    enc = tokenizer(flat, return_tensors='pt', max_length=256, truncation=True)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]  # (seq_len, 768)

    seq_len = hidden.size(0)

    # 1) 슬라이딩 윈도우 평균 벡터
    window_vecs = []
    for i in range(1, seq_len - WINDOW):
        vec = hidden[i:i+WINDOW].mean(0)
        window_vecs.append(vec)

    if len(window_vecs) < 2:
        print_result(row['idx'], row['class'], flat, [], actual)
        continue

    # 2) 인접 윈도우 간 코사인 유사도
    cos = torch.nn.functional.cosine_similarity
    sims = []
    for i in range(len(window_vecs) - 1):
        sim = cos(window_vecs[i], window_vecs[i+1], dim=0).item()
        sims.append((i + 1 + WINDOW//2, sim))  # 중앙 토큰 위치

    # 3) 토큰 위치 → 단어 위치 매핑
    word_to_token = []
    token_pos = 1
    for w in words:
        word_to_token.append(token_pos)
        w_tokens = tokenizer.encode(w, add_special_tokens=False)
        token_pos += len(w_tokens)
        if token_pos >= seq_len - 1:
            break
    while len(word_to_token) < len(words):
        word_to_token.append(min(token_pos, seq_len - 2))

    # 역매핑: 토큰 위치 → 가장 가까운 단어 인덱스
    def token_to_word(t_pos):
        best = 0
        for wi, tp in enumerate(word_to_token):
            if tp <= t_pos:
                best = wi
        return best + 1  # 다음 단어가 턴 시작

    # 4) 유사도 하위 N개 선택 (예상 턴 수에 맞춤)
    expected_turns = len(actual) if actual else max(3, len(words) // 8)
    sims.sort(key=lambda x: x[1])
    top_candidates = sims[:expected_turns * 2]

    # 단어 인덱스로 변환 + 중복 제거
    word_boundaries = set()
    for t_pos, sim in top_candidates:
        w_idx = token_to_word(t_pos)
        if 0 < w_idx < len(words):
            word_boundaries.add(w_idx)

    # 너무 가까운 경계 제거 (3단어 이내)
    predicted = sorted(word_boundaries)
    filtered = []
    for b in predicted:
        if not filtered or b - filtered[-1] >= 3:
            filtered.append(b)

    # 예상 턴 수에 맞춰 트림
    filtered = filtered[:expected_turns]

    p, r, f = print_result(row['idx'], row['class'], flat, filtered, actual)
    if actual:
        all_p.append(p); all_r.append(r); all_f.append(f)

print(f'\n{"="*60}')
print(f'EXP-C 위협 데이터 평균: P={sum(all_p)/len(all_p):.3f} R={sum(all_r)/len(all_r):.3f} F1={sum(all_f)/len(all_f):.3f}')
print(f'{"="*60}')
