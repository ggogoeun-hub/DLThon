"""
Stage 3 + 추가 HP: 로컬 Mac 실행용
6종 자동 순회: M2-H1, M2-H2, M2-H3, M3-B, M3-C, M3-D

실행: source venv/bin/activate && python src/run_stage3.py
"""
import pandas as pd
import numpy as np
import re
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

# ── 시드 & 디바이스 ──
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.environ["PYTORCH_MPS_DISABLE"] = "1"
device = torch.device('cpu')
print(f'Device: {device}')

# ── 데이터 로드 ──
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

# ── 설정 ──
MODEL_NAME = 'klue/roberta-base'
MAX_LEN = 256
BATCH_SIZE = 64  # 로컬 CPU 속도 최적화 (RAM 16GB 내 안전)
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

BEST_BACKBONE_LR = 2e-6  # M2-C 결과: 차등 10배가 맥락 보존에 최적
BEST_HEAD_LR = 2e-5
BEST_LS = 0.1            # M2-E 결과: 과신 방지로 일반화 향상
BEST_DROPOUT = 0.3
FREEZE_LAYERS = 9  # RoBERTa 하위 9층 동결 (0~8번 layer). 상위 3층만 학습

EXPERIMENTS = [
    ('M2-H1', 'Dropout 0.1',     BEST_BACKBONE_LR, BEST_HEAD_LR, BEST_LS, 0.1,  'mean'),
    ('M2-H2', 'Dropout 0.5',     BEST_BACKBONE_LR, BEST_HEAD_LR, BEST_LS, 0.5,  'mean'),
    ('M2-H3', 'Head LR 3e-5',    BEST_BACKBONE_LR, 3e-5,         BEST_LS, BEST_DROPOUT, 'mean'),
    ('M3-B',  'Ending Pooling',   BEST_BACKBONE_LR, BEST_HEAD_LR, BEST_LS, BEST_DROPOUT, 'ending'),
    ('M3-C',  'Speaker-aware',    BEST_BACKBONE_LR, BEST_HEAD_LR, BEST_LS, BEST_DROPOUT, 'speaker'),
    ('M3-D',  'Multi-head',       BEST_BACKBONE_LR, BEST_HEAD_LR, BEST_LS, BEST_DROPOUT, 'multihead'),
]

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

# ── 하위 N층 동결 함수 ──
def freeze_lower_layers(backbone, num_layers_to_freeze):
    """
    RoBERTa 백본의 하위 N개 encoder layer를 동결.
    - 하위층: 범용 언어 지식 (문법, 형태소) → 미세 조정 불필요
    - 상위층: 과제 특화 맥락 파악 → 학습 필요

    RoBERTa-base는 encoder.layer.0 ~ encoder.layer.11 (총 12층)
    num_layers_to_freeze=6이면 layer.0~5 동결, layer.6~11만 학습
    """
    # 임베딩 동결
    for param in backbone.embeddings.parameters():
        param.requires_grad = False

    # 하위 N개 layer 동결
    for i in range(num_layers_to_freeze):
        for param in backbone.encoder.layer[i].parameters():
            param.requires_grad = False

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f'  백본 동결: 하위 {num_layers_to_freeze}층 + 임베딩')
    print(f'  백본 파라미터: {total:,} (학습: {trainable:,} | 동결: {frozen:,})')


# ── Multi-head 분류기 ──
class MultiHeadClassifier(nn.Module):
    """
    3개 전문 헤드(대등성, 마무리톤, 위협의도) + 최종 결합 분류기
    """
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size

        self.equality_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))
        self.ending_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))
        self.threat_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        self.layer_norm = nn.LayerNorm(h)
        self.final_classifier = nn.Sequential(
            nn.Linear(h + 3, 256), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes))

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.layer_norm(self.mean_pooling(out.last_hidden_state, attention_mask))
        eq = self.equality_head(pooled)
        end = self.ending_head(pooled)
        threat = self.threat_head(pooled)
        combined = torch.cat([pooled, eq, end, threat], dim=-1)
        return self.final_classifier(combined)

# ── 표준 분류기 (Mean/CLS/Ending/Speaker) ──
class ConversationClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3, pooling='mean'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        h = self.backbone.config.hidden_size
        input_dim = h * 2 if pooling == 'speaker' else h
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling == 'mean':
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif self.pooling == 'cls':
            pooled = hidden[:, 0, :]
        elif self.pooling == 'ending':
            seq_lens = attention_mask.sum(dim=1)
            ending_pooled = []
            for i in range(hidden.size(0)):
                end = int(seq_lens[i].item())
                start = max(0, end - 64)
                ending_pooled.append(hidden[i, start:end, :].mean(dim=0))
            pooled = torch.stack(ending_pooled)
        elif self.pooling == 'speaker':
            B, S, H = hidden.shape
            pos = torch.arange(S, device=hidden.device).unsqueeze(0).expand(B, -1)
            odd_m = ((pos % 2 == 0) & (attention_mask == 1)).unsqueeze(-1).float()
            even_m = ((pos % 2 == 1) & (attention_mask == 1)).unsqueeze(-1).float()
            a = (hidden * odd_m).sum(1) / odd_m.sum(1).clamp(min=1e-9)
            b = (hidden * even_m).sum(1) / even_m.sum(1).clamp(min=1e-9)
            pooled = torch.cat([a, b], dim=-1)

        pooled = self.layer_norm(pooled)
        return self.fc2(self.dropout(self.activation(self.fc1(pooled))))

def create_model(model_name, num_classes, dropout_rate, pooling, freeze_layers=0):
    if pooling == 'multihead':
        model = MultiHeadClassifier(model_name, num_classes, dropout_rate)
    else:
        model = ConversationClassifier(model_name, num_classes, dropout_rate, pooling=pooling)
    if freeze_layers > 0:
        freeze_lower_layers(model.backbone, freeze_layers)
    return model

# ── 학습/검증 ──
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

# ── 메인 루프 ──
print(f'\n{"="*60}')
print(f'Stage 3 + 추가 HP | {len(EXPERIMENTS)}종 | CPU')
print(f'{"="*60}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ConversationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ConversationDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = ConversationDataset(test_texts, [0]*len(test_texts), tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_results = []
os.makedirs('outputs', exist_ok=True)

for exp_id, desc, backbone_lr, head_lr, ls, dropout, pooling in EXPERIMENTS:
    print(f'\n{"="*60}')
    print(f'[{exp_id}] {desc}')
    print(f'{"="*60}')

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = create_model(MODEL_NAME, len(label2id), dropout, pooling, freeze_layers=FREEZE_LAYERS).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  전체: {total_p:,} | 학습: {trainable_p:,} ({trainable_p/total_p*100:.1f}%)')

    head_p, backbone_p = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 동결된 파라미터 제외
        (backbone_p if 'backbone' in name else head_p).append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_p, 'lr': backbone_lr, 'weight_decay': WEIGHT_DECAY},
        {'params': head_p, 'lr': head_lr, 'weight_decay': WEIGHT_DECAY},
    ])
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    best_f1, best_epoch = 0, 0
    save_path = f'outputs/best_model_{exp_id}.pt'
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

    all_results.append({
        'exp_id': exp_id, 'desc': desc, 'val_f1': round(best_f1, 4),
        'normal_val_f1': round(normal_f1, 4), 'test_normal': test_normal,
        'best_epoch': best_epoch, 'time_min': round(total_time, 1)
    })

    del model, optimizer, scheduler; gc.collect()

# ── 결과 저장 ──
comp = pd.DataFrame(all_results).sort_values('val_f1', ascending=False)
comp.to_csv('outputs/stage3_results.csv', index=False)
print(f'\n{"="*60}')
print('전체 완료! 결과:')
print('=' * 60)
print(comp.to_string(index=False))
