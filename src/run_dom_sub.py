"""
지배/복종 공존 피처 실험: 구두점 유지 + Multi-head + 지배/복종 스코어
- 기존 M3-D 구조에 지배/복종 어휘 기반 스칼라 피처 추가
- 숏컷을 "이기는" 보조 신호 제공

실행: python -u src/run_dom_sub.py > outputs/dom_sub_log.txt 2>&1 &
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

# ── 지배/복종/대등 어휘 ──
DOMINANT = ['죽여', '죽일', '죽인다', '죽여버', '내놔', '시키는', '닥쳐', '패버', '때려',
            '찌르', '뒤져', '맞을래', '유포', '합의금', '죽을래', '부숴', '깽판', '짤리고']
SUBMISSIVE = ['죄송', '잘못', '살려', '제발', '용서', '그만', '안할게', '하지마',
              '도와주', '못하겠', '알겠습니다', '잘못했']
MUTUAL = ['그래', '알겠어', '좋아', '가자', '먹자', '하자', '맞아', '그치',
          '고마워', '오케이', '미안해', '인정', '대박', '화이팅']

def compute_features(text):
    """텍스트에서 지배/복종/대등 피처 5개 추출"""
    dom_count = sum(1 for w in DOMINANT if w in text)
    sub_count = sum(1 for w in SUBMISSIVE if w in text)
    mut_count = sum(1 for w in MUTUAL if w in text)
    coexist = 1.0 if dom_count > 0 and sub_count > 0 else 0.0
    asymmetry = abs(dom_count - sub_count) / max(dom_count + sub_count, 1)
    return [dom_count, sub_count, mut_count, coexist, asymmetry]

# ── 데이터 ──
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

# 피처 계산
train_df['features'] = train_df['text'].apply(compute_features)
val_df['features'] = val_df['text'].apply(compute_features)
test_df_raw['features'] = test_df_raw['text'].apply(compute_features)

train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
train_features = train_df['features'].tolist()
val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
val_features = val_df['features'].tolist()
test_texts = test_df_raw['text'].tolist()
test_features = test_df_raw['features'].tolist()
print(f'Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}')

# 피처 분포 확인
for cls in label2id:
    feats = [f for f, l in zip(train_features, train_labels) if l == label2id[cls]]
    avg_coexist = sum(f[3] for f in feats) / len(feats)
    avg_mutual = sum(f[2] for f in feats) / len(feats)
    print(f'  {cls}: 공존율={avg_coexist:.3f}, 대등어휘={avg_mutual:.1f}개')

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
N_FEATURES = 5  # dom, sub, mut, coexist, asymmetry

print(f'\n{"="*60}')
print(f'지배/복종 피처 실험 | Multi-head + {N_FEATURES}피처 | Epoch {EPOCHS} | 동결 {FREEZE_LAYERS}층')
print(f'{"="*60}')

# ── Dataset ──
class ConversationDataset(Dataset):
    def __init__(self, texts, labels, features, tokenizer, max_len):
        self.texts, self.labels, self.features = texts, labels, features
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'features': torch.tensor(self.features[idx], dtype=torch.float),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# ── 동결 ──
def freeze_lower_layers(backbone, n):
    for param in backbone.embeddings.parameters():
        param.requires_grad = False
    for i in range(n):
        for param in backbone.encoder.layer[i].parameters():
            param.requires_grad = False

# ── 모델 ──
class DomSubClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3, n_features=5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size  # 768

        # 기존 Multi-head 보조 헤드
        self.equality_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))
        self.ending_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))
        self.threat_head = nn.Sequential(nn.Linear(h,128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128,1))

        self.layer_norm = nn.LayerNorm(h)

        # 768 (pooled) + 3 (보조헤드) + 5 (지배/복종 피처) = 776
        self.final_classifier = nn.Sequential(
            nn.Linear(h + 3 + n_features, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def mean_pooling(self, hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, features):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.layer_norm(self.mean_pooling(out.last_hidden_state, attention_mask))
        eq = self.equality_head(pooled)
        end = self.ending_head(pooled)
        threat = self.threat_head(pooled)

        # pooled(768) + 보조(3) + 지배/복종 피처(5) = 776
        combined = torch.cat([pooled, eq, end, threat, features], dim=-1)
        return self.final_classifier(combined)

# ── 학습/평가 ──
def train_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, preds, trues = 0, [], []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask, features)
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
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            logits = model(ids, mask, features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(trues, preds, average='macro'), preds, trues

# ── 메인 ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ConversationDataset(train_texts, train_labels, train_features, tokenizer, MAX_LEN)
val_dataset = ConversationDataset(val_texts, val_labels, val_features, tokenizer, MAX_LEN)
test_dataset = ConversationDataset(test_texts, [0]*len(test_texts), test_features, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

os.makedirs('outputs', exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
model = DomSubClassifier(MODEL_NAME, len(label2id), DROPOUT, N_FEATURES).to(device)
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
save_path = 'outputs/best_model_dom_sub.pt'
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
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['features'].to(device))
        test_preds.extend(logits.argmax(-1).cpu().numpy())

test_classes = [str(id2label[p]) for p in test_preds]
test_normal = sum(1 for c in test_classes if c == '일반 대화')
print(f'  >> Test 일반대화: {test_normal}건 / 100건')
print(f'  >> Test 분포: {pd.Series(test_classes).value_counts().to_dict()}')

# Submission 저장
sub = pd.read_csv(f'{DATA_DIR}/submission.csv')
sub['class'] = test_classes
sub.to_csv('outputs/submission_dom_sub_str.csv', index=False)
sub_num = sub.copy()
sub_num['class'] = test_preds
sub_num.to_csv('outputs/submission_dom_sub_num.csv', index=False)
print(f'\n  >> Submission 저장 완료')

print(f'\n{"="*60}')
print(f'지배/복종 피처 실험 완료! | Val F1: {best_f1:.4f} | Test 일반: {test_normal}건 | {total_time:.1f}분')
print(f'{"="*60}')
