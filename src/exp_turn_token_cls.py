"""EXP-D: Token Classification (BIO 태깅) — 위협 데이터로 턴 경계 분류기 학습"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
import csv, re, random, warnings
warnings.filterwarnings('ignore')
from exp_turn_common import *

print('='*60)
print('EXP-D: Token Classification (턴 경계 분류기)')
print('='*60)

SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
MODEL_NAME = 'klue/roberta-base'
MAX_LEN = 256; BATCH_SIZE = 32; EPOCHS = 5; LR = 3e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── 학습 데이터 준비: 원본 위협 데이터 ──
with open('data/train.csv', 'r', encoding='utf-8-sig') as f:
    train_rows = list(csv.DictReader(f))

def prepare_sample(conversation):
    """대화 → (flat_text, token_labels)"""
    turns = [t.strip() for t in conversation.split('\n') if t.strip()]
    if len(turns) < 2:
        return None

    # flat text (구두점 제거)
    clean_turns = [strip_punct(t) for t in turns]
    flat = ' '.join(clean_turns)

    # 토큰화
    enc = tokenizer(flat, max_length=MAX_LEN, truncation=True, return_offsets_mapping=True)
    tokens = enc['input_ids']
    offsets = enc['offset_mapping']

    # 각 턴의 시작 문자 위치
    turn_char_starts = set()
    char_pos = 0
    for i, ct in enumerate(clean_turns):
        if i > 0:
            turn_char_starts.add(char_pos)
        char_pos += len(ct) + 1  # +1 for space

    # 토큰 라벨: 턴 시작 토큰 = 1, 나머지 = 0
    labels = []
    for start, end in offsets:
        if start == 0 and end == 0:  # special token
            labels.append(-100)
        elif start in turn_char_starts or (start > 0 and start - 1 in turn_char_starts):
            labels.append(1)
        else:
            labels.append(0)

    return {
        'input_ids': tokens,
        'attention_mask': enc['attention_mask'],
        'labels': labels
    }

# 데이터 준비
print('학습 데이터 준비 중...')
all_samples = []
for r in train_rows:
    s = prepare_sample(r['conversation'])
    if s:
        all_samples.append(s)

random.shuffle(all_samples)
split = int(len(all_samples) * 0.8)
train_samples = all_samples[:split]
val_samples = all_samples[split:]
print(f'  학습: {len(train_samples)}건, 검증: {len(val_samples)}건')

class TurnDataset(Dataset):
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        # 패딩
        pad_len = self.max_len - len(s['input_ids'])
        input_ids = s['input_ids'] + [1] * pad_len  # pad token
        attention_mask = s['attention_mask'] + [0] * pad_len
        labels = s['labels'] + [-100] * pad_len
        return {
            'input_ids': torch.tensor(input_ids[:self.max_len]),
            'attention_mask': torch.tensor(attention_mask[:self.max_len]),
            'labels': torch.tensor(labels[:self.max_len])
        }

train_loader = DataLoader(TurnDataset(train_samples, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TurnDataset(val_samples, MAX_LEN), batch_size=BATCH_SIZE)

# ── 모델 ──
class TurnBoundaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(hidden)

device = torch.device('cpu')
model = TurnBoundaryClassifier(MODEL_NAME).to(device)

# 동결 (빠른 학습)
for param in model.backbone.embeddings.parameters():
    param.requires_grad = False
for i in range(9):
    for param in model.backbone.encoder.layer[i].parameters():
        param.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=torch.tensor([1.0, 5.0]))  # 턴 경계에 가중치

# ── 학습 ──
print('학습 시작...')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 검증
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(ids, mask)
            preds = logits.argmax(-1)
            for p, l in zip(preds.view(-1).cpu(), labels.view(-1).cpu()):
                if l.item() != -100:
                    val_preds.append(p.item())
                    val_labels.append(l.item())

    f1 = f1_score(val_labels, val_preds, average='macro')
    print(f'  Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {f1:.4f}')

# ── 테스트 데이터 적용 ──
print('\n테스트 데이터 적용...')
rows = load_data()
all_p, all_r, all_f = [], [], []

model.eval()
for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual_boundaries(row['conversation'])
    words = flat.split()

    enc = tokenizer(flat, return_tensors='pt', max_length=MAX_LEN, truncation=True)
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        preds = logits.argmax(-1)[0].cpu()

    # 토큰 → 단어 매핑
    word_to_token = []
    token_pos = 1
    for w in words:
        word_to_token.append(token_pos)
        w_tokens = tokenizer.encode(w, add_special_tokens=False)
        token_pos += len(w_tokens)
        if token_pos >= preds.size(0) - 1:
            break

    # 턴 시작으로 예측된 토큰 → 단어 인덱스
    predicted_boundaries = []
    for wi, tp in enumerate(word_to_token):
        if tp < preds.size(0) and preds[tp].item() == 1:
            if wi > 0:  # 첫 단어는 제외
                predicted_boundaries.append(wi)

    p, r, f = print_result(row['idx'], row['class'], flat, predicted_boundaries, actual)
    if actual:
        all_p.append(p); all_r.append(r); all_f.append(f)

print(f'\n{"="*60}')
print(f'EXP-D 위협 데이터 평균: P={sum(all_p)/len(all_p):.3f} R={sum(all_r)/len(all_r):.3f} F1={sum(all_f)/len(all_f):.3f}')
print(f'{"="*60}')
