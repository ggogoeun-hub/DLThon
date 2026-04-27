"""턴 탐지 실험 결과를 CSV로 저장"""
import csv, re, torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import warnings; warnings.filterwarnings('ignore')

MODEL_NAME = 'klue/roberta-base'
DATA_PATH = 'data/data-exp/trun-split-exp/turn_split_test_25.csv'
OUT_DIR = 'data/data-exp/trun-split-exp'

def strip_punct(text):
    return re.sub(r'\s+', ' ', re.sub(r'[.?!]', '', text)).strip()

def flatten(conv):
    return strip_punct(conv.replace('\n', ' '))

def get_actual(conv):
    turns = [t.strip() for t in conv.split('\n') if t.strip()]
    if len(turns) <= 1: return []
    boundaries = []; idx = 0
    for t in turns[:-1]:
        idx += len(strip_punct(t).split())
        boundaries.append(idx)
    return boundaries

def insert_turns(flat_text, boundaries):
    """flat text에 boundary 위치마다 \\n 삽입"""
    words = flat_text.split()
    result = []
    for i, w in enumerate(words):
        if i in set(boundaries) and i > 0:
            result.append('\n')
        result.append(w)
    return ' '.join(result).replace(' \n ', '\n')

# 데이터 로드
with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME); model.eval()

ENDINGS = ['다', '요', '지', '냐', '래', '자', '네', '거든', '잖아', '야', '죠', '까',
           '세요', '습니다', '합니다', '해요', '니까', '는데', '었어', '았어', '겠어',
           '할게', '줘', '해줘', '인데', '는걸', '걸요', '데요']

# ===== EXP-B: 종결어미 + 임베딩 유사도 =====
print('EXP-B 처리 중...')
exp_b_rows = []
for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual(row['conversation'])
    words = flat.split()

    candidates = []
    for i, w in enumerate(words[:-1]):
        if any(w.endswith(e) for e in ENDINGS):
            candidates.append(i + 1)

    predicted = []
    if candidates:
        enc = tokenizer(flat, return_tensors='pt', max_length=256, truncation=True)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[0]
        word_to_token = []
        tp = 1
        for w in words:
            word_to_token.append(tp)
            tp += len(tokenizer.encode(w, add_special_tokens=False))
            if tp >= hidden.size(0) - 1: break
        while len(word_to_token) < len(words):
            word_to_token.append(min(tp, hidden.size(0) - 2))

        sims = []
        window = 5
        for c in candidates:
            t_pos = word_to_token[min(c, len(word_to_token)-1)]
            t_pos = min(t_pos, hidden.size(0) - 2)
            bs = max(1, t_pos - window)
            ae = min(hidden.size(0) - 1, t_pos + window)
            bv = hidden[bs:t_pos].mean(0)
            av = hidden[t_pos:ae].mean(0)
            sim = torch.nn.functional.cosine_similarity(bv, av, dim=0).item()
            sims.append((c, sim))
        sims.sort(key=lambda x: x[1])
        n_sel = max(1, int(len(sims) * 0.4))
        predicted = sorted([s[0] for s in sims[:n_sel]])

    conv_with_turns = insert_turns(flat, predicted)
    exp_b_rows.append({
        'idx': row['idx'], 'class': row['class'],
        'conversation_original': row['conversation'],
        'conversation_flat': flat,
        'conversation_predicted_turns': conv_with_turns,
        'actual_boundaries': str(actual),
        'predicted_boundaries': str(predicted)
    })

with open(f'{OUT_DIR}/result_exp_b_nsp.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=exp_b_rows[0].keys())
    w.writeheader(); w.writerows(exp_b_rows)
print(f'  저장: {OUT_DIR}/result_exp_b_nsp.csv')

# ===== EXP-C: 코히런스 =====
print('EXP-C 처리 중...')
WINDOW = 5
exp_c_rows = []
for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual(row['conversation'])
    words = flat.split()

    enc = tokenizer(flat, return_tensors='pt', max_length=256, truncation=True)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]
    seq_len = hidden.size(0)

    window_vecs = [hidden[i:i+WINDOW].mean(0) for i in range(1, seq_len - WINDOW)]
    cos = torch.nn.functional.cosine_similarity
    sims = [(i+1+WINDOW//2, cos(window_vecs[i], window_vecs[i+1], dim=0).item()) for i in range(len(window_vecs)-1)]

    word_to_token = []
    tp = 1
    for w in words:
        word_to_token.append(tp)
        tp += len(tokenizer.encode(w, add_special_tokens=False))
        if tp >= seq_len - 1: break
    while len(word_to_token) < len(words):
        word_to_token.append(min(tp, seq_len - 2))

    def token_to_word(t):
        best = 0
        for wi, tp in enumerate(word_to_token):
            if tp <= t: best = wi
        return best + 1

    expected = len(actual) if actual else max(3, len(words) // 8)
    sims.sort(key=lambda x: x[1])
    wb = set(token_to_word(t) for t, _ in sims[:expected*2] if 0 < token_to_word(t) < len(words))
    predicted = sorted(wb)
    filtered = []
    for b in predicted:
        if not filtered or b - filtered[-1] >= 3: filtered.append(b)
    filtered = filtered[:expected]

    conv_with_turns = insert_turns(flat, filtered)
    exp_c_rows.append({
        'idx': row['idx'], 'class': row['class'],
        'conversation_original': row['conversation'],
        'conversation_flat': flat,
        'conversation_predicted_turns': conv_with_turns,
        'actual_boundaries': str(actual),
        'predicted_boundaries': str(filtered)
    })

with open(f'{OUT_DIR}/result_exp_c_coherence.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=exp_c_rows[0].keys())
    w.writeheader(); w.writerows(exp_c_rows)
print(f'  저장: {OUT_DIR}/result_exp_c_coherence.csv')

# ===== EXP-D: Token Classification =====
print('EXP-D 처리 중...')

class TurnBoundaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(hidden)

# 학습된 모델이 없으면 빠른 학습
import os
model_path = 'outputs/best_model_turn_cls.pt'
turn_model = TurnBoundaryClassifier(MODEL_NAME)

if not os.path.exists(model_path):
    print('  턴 분류기 학습 중...')
    from torch.utils.data import Dataset, DataLoader
    import random, numpy as np
    random.seed(42); torch.manual_seed(42)

    with open('data/train.csv', 'r', encoding='utf-8-sig') as f:
        train_data = list(csv.DictReader(f))

    samples = []
    for r in train_data:
        turns = [t.strip() for t in r['conversation'].split('\n') if t.strip()]
        if len(turns) < 2: continue
        clean_turns = [strip_punct(t) for t in turns]
        flat = ' '.join(clean_turns)
        enc = tokenizer(flat, max_length=256, truncation=True, return_offsets_mapping=True)
        turn_starts = set(); cp = 0
        for i, ct in enumerate(clean_turns):
            if i > 0: turn_starts.add(cp)
            cp += len(ct) + 1
        labels = []
        for s, e in enc['offset_mapping']:
            if s == 0 and e == 0: labels.append(-100)
            elif s in turn_starts or (s > 0 and s-1 in turn_starts): labels.append(1)
            else: labels.append(0)
        samples.append({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': labels})

    class TD(Dataset):
        def __init__(s, samples):
            s.samples = samples
        def __len__(s): return len(s.samples)
        def __getitem__(s, i):
            s2 = s.samples[i]
            pl = 256 - len(s2['input_ids'])
            return {
                'input_ids': torch.tensor((s2['input_ids'] + [1]*pl)[:256]),
                'attention_mask': torch.tensor((s2['attention_mask'] + [0]*pl)[:256]),
                'labels': torch.tensor((s2['labels'] + [-100]*pl)[:256])
            }

    for p in turn_model.backbone.embeddings.parameters(): p.requires_grad = False
    for i in range(9):
        for p in turn_model.backbone.encoder.layer[i].parameters(): p.requires_grad = False

    loader = DataLoader(TD(samples), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, turn_model.parameters()), lr=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=-100, weight=torch.tensor([1.0, 5.0]))

    for ep in range(5):
        turn_model.train(); tl = 0
        for b in loader:
            opt.zero_grad()
            logits = turn_model(b['input_ids'], b['attention_mask'])
            loss = crit(logits.view(-1,2), b['labels'].view(-1))
            loss.backward(); opt.step(); tl += loss.item()
        print(f'    Epoch {ep+1} | Loss: {tl/len(loader):.4f}')
    torch.save(turn_model.state_dict(), model_path)
else:
    turn_model.load_state_dict(torch.load(model_path, weights_only=True))

turn_model.eval()
exp_d_rows = []
for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual(row['conversation'])
    words = flat.split()

    enc = tokenizer(flat, return_tensors='pt', max_length=256, truncation=True)
    with torch.no_grad():
        preds = turn_model(enc['input_ids'], enc['attention_mask']).argmax(-1)[0].cpu()

    word_to_token = []
    tp = 1
    for w in words:
        word_to_token.append(tp)
        tp += len(tokenizer.encode(w, add_special_tokens=False))
        if tp >= preds.size(0) - 1: break

    predicted = [wi for wi, tp in enumerate(word_to_token) if tp < preds.size(0) and preds[tp].item() == 1 and wi > 0]

    conv_with_turns = insert_turns(flat, predicted)
    exp_d_rows.append({
        'idx': row['idx'], 'class': row['class'],
        'conversation_original': row['conversation'],
        'conversation_flat': flat,
        'conversation_predicted_turns': conv_with_turns,
        'actual_boundaries': str(actual),
        'predicted_boundaries': str(predicted)
    })

with open(f'{OUT_DIR}/result_exp_d_token_cls.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=exp_d_rows[0].keys())
    w.writeheader(); w.writerows(exp_d_rows)
print(f'  저장: {OUT_DIR}/result_exp_d_token_cls.csv')

# ===== 비교 요약 =====
print('\n전체 결과 저장 중...')
summary = []
for name, rr in [('EXP-B_NSP', exp_b_rows), ('EXP-C_Coherence', exp_c_rows), ('EXP-D_TokenCls', exp_d_rows)]:
    for r in rr:
        summary.append({
            'experiment': name, 'idx': r['idx'], 'class': r['class'],
            'actual_boundaries': r['actual_boundaries'],
            'predicted_boundaries': r['predicted_boundaries'],
            'conversation_predicted_turns': r['conversation_predicted_turns']
        })

with open(f'{OUT_DIR}/comparison_all.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader(); w.writerows(summary)
print(f'  저장: {OUT_DIR}/comparison_all.csv')
print('\n완료!')
