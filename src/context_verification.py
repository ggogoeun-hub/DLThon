"""
맥락 파악 검증: 비속어/위협 키워드 포함 대화에서 모델이 키워드가 아닌 맥락으로 판단하는지 확인

실행: source venv/bin/activate && python src/context_verification.py
"""
import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report

# ── 설정 ──
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.environ["PYTORCH_MPS_DISABLE"] = "1"
device = torch.device('cpu')

# ── 데이터 ──
DATA_DIR = 'data'
val_df = pd.read_csv(f'{DATA_DIR}/val_final.csv')

label2id = {'갈취 대화':0, '기타 괴롭힘 대화':1, '직장 내 괴롭힘 대화':2, '협박 대화':3, '일반 대화':4}
id2label = {v:k for k,v in label2id.items()}

def preprocess(text):
    if not isinstance(text, str): return ''
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

val_df['text'] = val_df['conversation'].apply(preprocess)
val_df['label'] = val_df['class'].map(label2id)

# ── 검증 키워드 그룹 ──
KEYWORD_GROUPS = {
    '위협 관련 ("죽", "살려", "칼")': ['죽', '살려', '칼'],
    '금전 관련 ("돈", "만원", "빌려")': ['돈', '만원', '빌려'],
    '비속어 ("새끼", "씨발", "미친")': ['새끼', '씨발', '시발', '미친'],
    '직장 관련 ("부장", "과장", "회사")': ['부장', '과장', '회사'],
    '관용 표현 ("죽을래", "죽겠다", "때릴")': ['죽을래', '죽겠다', '때릴', '때려'],
}

print('=' * 60)
print('맥락 파악 검증: 키워드별 분류 정확도 분석')
print('=' * 60)

# ── Val 데이터에서 키워드별 분포 먼저 확인 ──
print('\n[키워드별 Val 데이터 분포]')
for group_name, keywords in KEYWORD_GROUPS.items():
    mask = val_df['conversation'].apply(lambda x: any(kw in str(x) for kw in keywords))
    subset = val_df[mask]
    if len(subset) == 0:
        print(f'\n{group_name}: 0건')
        continue
    print(f'\n{group_name}: {len(subset)}건')
    class_dist = subset['class'].value_counts()
    for cls, cnt in class_dist.items():
        print(f'  {cls}: {cnt}건')

# ── 모델 정의 (Stage 1/2와 동일) ──
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

class ConversationClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.layer_norm = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.layer_norm(pooled)
        return self.fc2(self.dropout(self.activation(self.fc1(pooled))))

# ── 모델 로드 & 예측 ──
# Stage 1에서 저장된 best model 경로가 없으므로 새로 추론
MODELS_TO_TEST = [
    ('RoBERTa', 'klue/roberta-base'),
    ('KcELECTRA', 'beomi/KcELECTRA-base'),
]

MAX_LEN = 256
BATCH_SIZE = 32
BACKBONE_LR = 2e-6
HEAD_LR = 2e-5
EPOCHS = 3

tokenizer_cache = {}
results_by_model = {}

for model_label, model_name in MODELS_TO_TEST:
    print(f'\n{"="*60}')
    print(f'[{model_label}] {model_name} 학습 + 검증')
    print(f'{"="*60}')

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Train 데이터도 필요
    train_df = pd.read_csv(f'{DATA_DIR}/train_final.csv')
    train_df['text'] = train_df['conversation'].apply(preprocess)
    train_df['label'] = train_df['class'].map(label2id)

    train_dataset = ConversationDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LEN)
    val_dataset = ConversationDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = ConversationClassifier(model_name, len(label2id), 0.3).to(device)

    # 옵티마이저 (차등 10배)
    head_p, backbone_p = [], []
    for name, param in model.named_parameters():
        (backbone_p if 'backbone' in name else head_p).append(param)
    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW([
        {'params': backbone_p, 'lr': BACKBONE_LR, 'weight_decay': 0.01},
        {'params': head_p, 'lr': HEAD_LR, 'weight_decay': 0.01},
    ])
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 학습
    import time
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        for batch in train_loader:
            ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()

        # Val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                logits = model(ids, mask)
                preds.extend(logits.argmax(-1).cpu().numpy())
                trues.extend(batch['label'].numpy())
        val_f1 = f1_score(trues, preds, average='macro')
        elapsed = time.time() - start
        marker = ''
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_preds = preds.copy()
            marker = ' << BEST'
        print(f'  Epoch {epoch+1} | Val F1: {val_f1:.4f} | {elapsed:.0f}s{marker}')

    # 결과 저장
    results_by_model[model_label] = best_preds
    val_df[f'pred_{model_label}'] = best_preds
    val_df[f'pred_class_{model_label}'] = [id2label[p] for p in best_preds]

    del model, optimizer, scheduler, train_dataset, train_loader
    import gc; gc.collect()

# ── 키워드별 맥락 파악 정확도 비교 ──
print(f'\n{"="*60}')
print('키워드별 맥락 파악 정확도 비교')
print('=' * 60)

summary_rows = []

for group_name, keywords in KEYWORD_GROUPS.items():
    mask = val_df['conversation'].apply(lambda x: any(kw in str(x) for kw in keywords))
    subset = val_df[mask]
    if len(subset) == 0:
        continue

    print(f'\n[{group_name}] ({len(subset)}건)')

    for model_label, _ in MODELS_TO_TEST:
        pred_col = f'pred_{model_label}'
        correct = (subset['label'] == subset[pred_col]).sum()
        accuracy = correct / len(subset) * 100

        # 일반 대화 서브셋의 정확도
        normal_subset = subset[subset['class'] == '일반 대화']
        if len(normal_subset) > 0:
            normal_correct = (normal_subset['label'] == normal_subset[pred_col]).sum()
            normal_acc = normal_correct / len(normal_subset) * 100
            normal_info = f'일반대화 {normal_correct}/{len(normal_subset)} ({normal_acc:.1f}%)'
        else:
            normal_acc = None
            normal_info = '일반대화 0건'

        print(f'  {model_label:>12}: 전체 {correct}/{len(subset)} ({accuracy:.1f}%) | {normal_info}')

        summary_rows.append({
            'keyword_group': group_name,
            'model': model_label,
            'total_samples': len(subset),
            'accuracy': round(accuracy, 1),
            'normal_samples': len(normal_subset) if len(normal_subset) > 0 else 0,
            'normal_accuracy': round(normal_acc, 1) if normal_acc is not None else None,
        })

# ── 오분류 상세 분석 ──
print(f'\n{"="*60}')
print('오분류 상세: 일반 대화인데 위협으로 잘못 분류된 사례')
print('=' * 60)

for model_label, _ in MODELS_TO_TEST:
    pred_col = f'pred_class_{model_label}'
    misclassified = val_df[(val_df['class'] == '일반 대화') & (val_df[pred_col] != '일반 대화')]

    print(f'\n[{model_label}] 일반→위협 오분류: {len(misclassified)}건')
    if len(misclassified) > 0:
        print(f'  오분류 대상:')
        print(f'  {misclassified[pred_col].value_counts().to_dict()}')
        for i, (_, row) in enumerate(misclassified.head(3).iterrows()):
            print(f'  [{i+1}] 예측: {row[pred_col]}')
            print(f'      내용: {row["conversation"][:100]}...')

# ── 결과 저장 ──
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/context_verification.csv', index=False)
print(f'\n결과 저장: outputs/context_verification.csv')

print(f'\n{"="*60}')
print('검증 완료!')
print('=' * 60)
