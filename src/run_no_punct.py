"""
구두점 제거 실험: Multi-head + 동결 9층 + 6ep
전처리에서 . ? ! 제거하여 숏컷 해소 검증

실행: caffeinate -s & && python -u src/run_no_punct.py > outputs/no_punct_log.txt 2>&1 &
"""
import pandas as pd
import numpy as np
import re
import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
os.environ["PYTORCH_MPS_DISABLE"] = "1"
device = torch.device('cpu')
print(f'Device: {device}')

# ── 데이터 ──
DATA_DIR = 'data'
train_df = pd.read_csv(f'{DATA_DIR}/train_final.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_final.csv')
test_df_raw = pd.read_csv(f'{DATA_DIR}/test.csv')

label2id = {'갈취 대화':0, '기타 괴롭힘 대화':1, '직장 내 괴롭힘 대화':2, '협박 대화':3, '일반 대화':4}
id2label = {v:k for k,v in label2id.items()}

def preprocess(text):
    if not isinstance(text, str): return ''
    text = text.replace('\n', ' ')
    text = re.sub(r'[.?!]', '', text)  # 구두점 제거
    return re.sub(r'\s+', ' ', text).strip()

train_df['text'] = train_df['conversation'].apply(preprocess)
train_df['label'] = train_df['class'].map(label2id)
val_df['text'] = val_df['conversation'].apply(preprocess)
val_df['label'] = val_df['class'].map(label2id)
test_df_raw['text'] = test_df_raw['conversation'].apply(lambda x: preprocess(str(x).replace('"','')))

train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
test_texts = test_df_raw['text'].tolist()
print(f'Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}')

# 구두점 제거 검증
for name, texts in [('Train 일반', [t for t,l in zip(train_texts, train_labels) if l==4]),
                     ('Train 위협', [t for t,l in zip(train_texts, train_labels) if l!=4]),
                     ('Test', test_texts)]:
    periods = sum(t.count('.') for t in texts) / len(texts)
    print(f'  {name}: 마침표 {periods:.3f}개/건 (0이어야 함)')

# ── 설정 ──
MODEL_NAME = 'klue/roberta-base'
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BACKBONE_LR = 2e-6
HEAD_LR = 2e-5
LS = 0.1
DROPOUT = 0.3
FREEZE_LAYERS = 9

print(f'\n{"="*60}')
print(f'구두점 제거 실험 | Multi-head | Epoch {EPOCHS} | 동결 {FREEZE_LAYERS}층')
print(f'{"="*60}')

# ── Dataset ──
class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# ── 동결 ──
def freeze_lower_layers(backbone, n):
    for param in backbone.embeddings.parameters():
        param.requires_grad = False
    for i in range(n):
        for param in backbone.encoder.layer[i].parameters():
            param.requires_grad = False

# ── 모델 ──
class MultiHeadClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.equality_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))
        self.ending_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))
        self.threat_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))
        self.layer_norm = nn.LayerNorm(h)
        self.final_classifier = nn.Sequential(nn.Linear(h+3,256), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(256,num_classes))

    def mean_pooling(self, hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.layer_norm(self.mean_pooling(out.last_hidden_state, attention_mask))
        eq = self.equality_head(pooled)
        end = self.ending_head(pooled)
        threat = self.threat_head(pooled)
        return self.final_classifier(torch.cat([pooled, eq, end, threat], dim=-1))

# ── 학습/평가 ──
def train_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, preds, trues = 0, [], []
    for batch in loader:
        ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(trues, preds, average='macro')

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(trues, preds, average='macro'), preds, trues

# ── 메인 ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ConversationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ConversationDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = ConversationDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

os.makedirs('outputs', exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
model = MultiHeadClassifier(MODEL_NAME, len(label2id), DROPOUT).to(device)
freeze_lower_layers(model.backbone, FREEZE_LAYERS)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  전체: {total_p:,} | 학습: {train_p:,} ({train_p/total_p*100:.1f}%)')

head_p, backbone_p = [], []
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    (backbone_p if 'backbone' in name else head_p).append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_p, 'lr': BACKBONE_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': head_p, 'lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY},
])
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
criterion = nn.CrossEntropyLoss(label_smoothing=LS)

best_f1, best_epoch = 0, 0
save_path = 'outputs/best_model_no_punct.pt'
total_start = time.time()

for epoch in range(EPOCHS):
    start = time.time()
    t_loss, t_f1 = train_epoch(model, train_loader, criterion, optimizer, scheduler)
    v_loss, v_f1, _, _ = evaluate(model, val_loader, criterion)
    elapsed = time.time() - start
    marker = ''
    if v_f1 > best_f1:
        best_f1, best_epoch = v_f1, epoch + 1
        torch.save(model.state_dict(), save_path)
        marker = ' << BEST'
    print(f'  Epoch {epoch+1} | Train F1: {t_f1:.4f} | Val F1: {v_f1:.4f} | {elapsed:.0f}s{marker}')

total_time = (time.time() - total_start) / 60

# Best 모델 로드 & 평가
model.load_state_dict(torch.load(save_path, weights_only=True))
_, final_f1, final_preds, final_trues = evaluate(model, val_loader, criterion)
class_names = [id2label[i] for i in range(5)]
report = classification_report(final_trues, final_preds, target_names=class_names, digits=4, output_dict=True)
normal_f1 = report['일반 대화']['f1-score']

print(f'\n  >> Best: Epoch {best_epoch} | Val F1: {best_f1:.4f} | 일반대화 F1: {normal_f1:.4f} | {total_time:.1f}분')
print(classification_report(final_trues, final_preds, target_names=class_names, digits=4))

# Test 추론
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        test_preds.extend(logits.argmax(-1).cpu().numpy())

test_classes = [str(id2label[p]) for p in test_preds]
test_normal = sum(1 for c in test_classes if c == '일반 대화')
print(f'  >> Test 일반대화: {test_normal}건 / 100건')
print(f'  >> Test 분포: {pd.Series(test_classes).value_counts().to_dict()}')

# Submission 저장
sub = pd.read_csv(f'{DATA_DIR}/submission.csv')
sub['class'] = test_classes
sub.to_csv('outputs/submission_no_punct_str.csv', index=False)
sub_num = sub.copy()
sub_num['class'] = test_preds
sub_num.to_csv('outputs/submission_no_punct_num.csv', index=False)
print(f'\n  >> Submission 저장 완료')

print(f'\n{"="*60}')
print(f'구두점 제거 실험 완료! | Val F1: {best_f1:.4f} | Test 일반: {test_normal}건 | {total_time:.1f}분')
print(f'{"="*60}')
