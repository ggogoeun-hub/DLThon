"""EXP-B: 종결어미 + 임베딩 유사도 기반 턴 탐지"""
import torch
from transformers import AutoTokenizer, AutoModel
import warnings; warnings.filterwarnings('ignore')
from exp_turn_common import *

print('='*60)
print('EXP-B: 종결어미 + 임베딩 유사도')
print('='*60)

MODEL_NAME = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME); model.eval()

ENDINGS = ['다', '요', '지', '냐', '래', '자', '네', '거든', '잖아', '야', '죠', '까',
           '세요', '습니다', '합니다', '해요', '니까', '는데', '었어', '았어', '겠어',
           '할게', '줘', '해줘', '인데', '는걸', '걸요', '데요']

rows = load_data()
all_p, all_r, all_f = [], [], []

for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual_boundaries(row['conversation'])
    words = flat.split()

    # 1) 종결어미 위치 = 후보 분할점
    candidates = []
    for i, w in enumerate(words[:-1]):
        if any(w.endswith(e) for e in ENDINGS):
            candidates.append(i + 1)  # 다음 단어부터 새 턴

    if not candidates:
        print_result(row['idx'], row['class'], flat, [], actual)
        continue

    # 2) 각 후보에서 앞/뒤 문장의 임베딩 유사도 계산
    enc = tokenizer(flat, return_tensors='pt', max_length=256, truncation=True)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]  # (seq_len, 768)

    # 단어 → 토큰 매핑
    word_to_token_start = []
    token_pos = 1  # [CLS] 다음
    for w in words:
        word_to_token_start.append(token_pos)
        w_tokens = tokenizer.encode(w, add_special_tokens=False)
        token_pos += len(w_tokens)
        if token_pos >= hidden.size(0) - 1:
            break
    # 부족한 단어에 대해 마지막 토큰 위치 채우기
    while len(word_to_token_start) < len(words):
        word_to_token_start.append(min(token_pos, hidden.size(0) - 2))

    # 3) 각 후보에서 앞 5토큰 평균 vs 뒤 5토큰 평균 코사인 유사도
    similarities = []
    window = 5
    for cand in candidates:
        t_pos = word_to_token_start[min(cand, len(word_to_token_start)-1)]
        t_pos = min(t_pos, hidden.size(0) - 2)

        before_start = max(1, t_pos - window)
        after_end = min(hidden.size(0) - 1, t_pos + window)

        before_vec = hidden[before_start:t_pos].mean(0)
        after_vec = hidden[t_pos:after_end].mean(0)

        sim = torch.nn.functional.cosine_similarity(before_vec, after_vec, dim=0).item()
        similarities.append((cand, sim))

    # 4) 유사도 기준으로 턴 경계 선택 (하위 40%)
    similarities.sort(key=lambda x: x[1])
    n_select = max(1, int(len(similarities) * 0.4))
    predicted = sorted([s[0] for s in similarities[:n_select]])

    p, r, f = print_result(row['idx'], row['class'], flat, predicted, actual)
    if actual:
        all_p.append(p); all_r.append(r); all_f.append(f)

print(f'\n{"="*60}')
print(f'EXP-B 위협 데이터 평균: P={sum(all_p)/len(all_p):.3f} R={sum(all_r)/len(all_r):.3f} F1={sum(all_f)/len(all_f):.3f}')
print(f'{"="*60}')
