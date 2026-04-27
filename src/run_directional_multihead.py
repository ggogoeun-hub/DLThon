"""
방향성 Multi-head: escalation vs resolution 분리
- escalation = relu(후반 - 전반) → 후반이 더 거친 부분
- resolution = relu(전반 - 후반) → 해소된 부분
- ending = 마지막 64토큰
- threat = 전체 mean pool

실행: python -u src/run_directional_multihead.py > outputs/directional_mh_log.txt 2>&1 &
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

DATA_DIR = 'data'
train_df = pd.read_csv(f'{DATA_DIR}/train_final.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_final.csv')
test_df_raw = pd.read_csv(f'{DATA_DIR}/test.csv')

label2id = {'갈취 대화':0, '기타 괴롭힘 대화':1, '직장 내 괴롭힘 대화':2, '협박 대화':3, '일반 대화':4}
id2label = {v:k for k,v in label2id.items()}

def preprocess(text):
    if not isinstance(text, str): return ''
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

train_df['text'] = train_df['conversation'].apply(preprocess)
train_df['label'] = train_df['class'].map(label2id)
val_df['text'] = val_df['conversation'].apply(preprocess)
val_df['label'] = val_df['class'].map(label2id)
test_df_raw['text'] = test_df_raw['conversation'].apply(lambda x: preprocess(str(x).replace('"','')))

train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
test_texts = test_df_raw['text'].tolist()
print(f'Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}')

MODEL_NAME = 'klue/roberta-base'
MAX_LEN = 256; BATCH_SIZE = 64; EPOCHS = 3
WARMUP_RATIO = 0.1; WEIGHT_DECAY = 0.01
BACKBONE_LR = 2e-6; HEAD_LR = 2e-5
LS = 0.1; DROPOUT = 0.3; FREEZE_LAYERS = 9

print(f'\n{"="*60}')
print(f'방향성 Multi-head | escalation/resolution 분리 | {EPOCHS}ep | 동결 {FREEZE_LAYERS}층')
print(f'{"="*60}')

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

def freeze_lower_layers(backbone, n):
    for param in backbone.embeddings.parameters():
        param.requires_grad = False
    for i in range(n):
        for param in backbone.encoder.layer[i].parameters():
            param.requires_grad = False

class DirectionalMultiHeadClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size  # 768

        # escalation: relu(후반 - 전반) → 후반이 더 거친 방향만
        self.escalation_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        # resolution: relu(전반 - 후반) → 해소 방향만
        self.resolution_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        # ending: 마지막 64토큰
        self.ending_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        # threat: 전체 mean pool
        self.threat_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        self.layer_norm = nn.LayerNorm(h)

        # 768 (pooled) + 4 (스칼라) = 772
        self.final_classifier = nn.Sequential(
            nn.Linear(h + 4, 256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes))

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        seq_lens = attention_mask.sum(dim=1)

        # 전체 mean pool
        pooled = self.layer_norm((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))

        esc_list, res_list, end_list = [], [], []
        for i in range(hidden.size(0)):
            L = int(seq_lens[i].item())
            mid = L // 2

            # 전반/후반 풀링
            fm = torch.zeros(hidden.size(1), device=hidden.device); fm[1:mid] = 1.0
            sm = torch.zeros(hidden.size(1), device=hidden.device); sm[mid:L-1] = 1.0
            first = (hidden[i] * fm.unsqueeze(-1)).sum(0) / fm.sum().clamp(min=1e-9)
            second = (hidden[i] * sm.unsqueeze(-1)).sum(0) / sm.sum().clamp(min=1e-9)

            # 방향 분리
            esc_list.append(torch.relu(second - first))   # 에스컬레이션
            res_list.append(torch.relu(first - second))   # 해소

            # ending
            st = max(1, L - 64)
            em = torch.zeros(hidden.size(1), device=hidden.device); em[st:L-1] = 1.0
            end_list.append((hidden[i] * em.unsqueeze(-1)).sum(0) / em.sum().clamp(min=1e-9))

        esc_score = self.escalation_head(torch.stack(esc_list))
        res_score = self.resolution_head(torch.stack(res_list))
        end_score = self.ending_head(torch.stack(end_list))
        threat_score = self.threat_head(pooled)

        combined = torch.cat([pooled, esc_score, res_score, end_score, threat_score], dim=-1)
        return self.final_classifier(combined)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ConversationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ConversationDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = ConversationDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(SEED); np.random.seed(SEED)
model = DirectionalMultiHeadClassifier(MODEL_NAME, len(label2id), DROPOUT).to(device)
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
save_path = 'outputs/best_model_directional_mh.pt'
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

model.load_state_dict(torch.load(save_path, weights_only=True))
_, final_f1, final_preds, final_trues = evaluate(model, val_loader, criterion)
class_names = [id2label[i] for i in range(5)]
report = classification_report(final_trues, final_preds, target_names=class_names, digits=4, output_dict=True)
normal_f1 = report['일반 대화']['f1-score']

print(f'\n  >> Best: Epoch {best_epoch} | Val F1: {best_f1:.4f} | 일반대화 F1: {normal_f1:.4f} | {total_time:.1f}분')
print(classification_report(final_trues, final_preds, target_names=class_names, digits=4))

model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        test_preds.extend(logits.argmax(-1).cpu().numpy())
test_classes = [str(id2label[p]) for p in test_preds]
test_normal = sum(1 for c in test_classes if c == '일반 대화')
print(f'  >> Test 일반대화: {test_normal}건 / 100건')

sub = pd.read_csv(f'{DATA_DIR}/submission.csv')
sub['class'] = test_classes
sub.to_csv('outputs/submission_directional_mh_str.csv', index=False)

print(f'\n{"="*60}')
print(f'방향성 Multi-head 완료! | Val F1: {best_f1:.4f} | Test 일반: {test_normal}건 | {total_time:.1f}분')
print(f'{"="*60}')
