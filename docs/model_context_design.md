# 맥락 인식 모델 설계안

> 현재 문제: 모델이 단어/구두점 패턴으로 분류 (숏컷), 맥락(누가→누구에게, 의도, 결말)을 파악 못함
> 목표: strategy_v2의 "쌍방 대등 + 마무리 평화 + 위협 의도 없음" 판별 기준을 모델 구조에 직접 반영

---

## 현재 모델 (B01~B03)

```
텍스트 전체 → KcELECTRA → [CLS] → Linear → 5클래스
```

문제: [CLS] 하나에 대화 전체를 압축 → 단어 빈도에 의존, 구조적 맥락 소실

---

## 제안: Context-Aware Multi-Head Model

### 아이디어

대화를 "덩어리"가 아니라 **구조적으로** 읽는다:
1. **화자 A와 B를 분리**해서 권력 비대칭을 감지
2. **대화 마무리(후반부)를 별도로** 읽어서 결말의 톤을 판단
3. **보조 태스크**로 "대등성", "마무리 평화", "위협 의도"를 명시적으로 학습

### 아키텍처

```
입력 텍스트
  │
  ▼
KcELECTRA (공유 백본)
  │
  ├─── [CLS] ──────────────────── Main Head → 5클래스 (최종 분류)
  │
  ├─── 홀수 턴 tokens (화자A) ─┐
  │                             ├─ 차이 벡터 ─── Symmetry Head → 대등/비대칭 (보조)
  ├─── 짝수 턴 tokens (화자B) ─┘
  │
  └─── 마지막 2~3턴 tokens ──── Ending Head → 평화/위협 마무리 (보조)
```

### 핵심 컴포넌트

#### 1. Speaker-Aware Pooling (화자 분리)

대화를 턴 단위로 나눈 뒤, 홀수 턴(화자A)과 짝수 턴(화자B)의 hidden state를 각각 풀링:

```python
class SpeakerAwarePooling(nn.Module):
    """화자별 representation 추출 + 대칭성 벡터 생성"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)  # attention weight

    def forward(self, hidden_states, speaker_mask_a, speaker_mask_b):
        # 화자 A 토큰만 추출 → attention pooling
        repr_a = self._masked_attn_pool(hidden_states, speaker_mask_a)
        # 화자 B 토큰만 추출 → attention pooling
        repr_b = self._masked_attn_pool(hidden_states, speaker_mask_b)

        # 대칭성 벡터: 두 화자 표현의 차이 + 상호작용
        symmetry = torch.cat([
            repr_a - repr_b,      # 차이
            repr_a * repr_b,      # 상호작용
        ], dim=-1)

        return repr_a, repr_b, symmetry

    def _masked_attn_pool(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)
```

**이게 왜 중요한가:**
- 위협 대화: 화자A(가해자)와 B(피해자)의 representation이 매우 다름 (비대칭)
- 일반 대화: 화자A와 B의 representation이 유사 (대등)
- 모델이 이 차이를 명시적으로 학습

#### 2. Ending-Aware Pooling (마무리 분석)

대화 후반 20% 토큰에 attention을 집중:

```python
class EndingPooling(nn.Module):
    """대화 마지막 부분에 집중하는 풀링"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        seq_len = attention_mask.sum(dim=1, keepdim=True)
        # 마지막 20% 위치만 마스킹
        positions = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        ending_mask = positions >= (seq_len * 0.8)
        ending_mask = ending_mask & attention_mask.bool()

        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~ending_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)
```

**이게 왜 중요한가:**
- 위협 대화: 후반부에 위협 밀도 급증 (협박 0.26→0.64)
- 일반 대화: 후반부도 평화로움
- 마무리 톤을 별도로 읽으면 "격앙 → 굴복" vs "장난 → 화해" 구분 가능

#### 3. Multi-Head Loss (보조 태스크)

```python
class ContextAwareClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, hidden_size=768):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)

        # Main head: 5클래스 분류
        self.main_head = nn.Linear(hidden_size, num_classes)

        # Aux head 1: 대등성 (symmetric=0, asymmetric=1)
        self.speaker_pool = SpeakerAwarePooling(hidden_size)
        self.symmetry_head = nn.Linear(hidden_size * 2, 2)

        # Aux head 2: 마무리 (peaceful=0, threatening=1)
        self.ending_pool = EndingPooling(hidden_size)
        self.ending_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, speaker_mask_a=None, speaker_mask_b=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Main: [CLS]
        cls = self.dropout(hidden[:, 0, :])
        main_logits = self.main_head(cls)

        # Aux 1: 대등성
        sym_logits = None
        if speaker_mask_a is not None:
            _, _, symmetry = self.speaker_pool(hidden, speaker_mask_a, speaker_mask_b)
            sym_logits = self.symmetry_head(self.dropout(symmetry))

        # Aux 2: 마무리
        ending_repr = self.ending_pool(hidden, attention_mask)
        end_logits = self.ending_head(self.dropout(ending_repr))

        return main_logits, sym_logits, end_logits
```

### 보조 라벨 생성 (자동)

보조 태스크 라벨을 별도 어노테이션 없이 **메인 라벨에서 자동 유도**:

```python
def derive_aux_labels(main_label):
    """
    메인 라벨 → 보조 라벨 자동 생성
    0=갈취, 1=기타괴롭힘, 2=일반, 3=직장괴롭힘, 4=협박
    """
    # 대등성: 일반=대등(0), 나머지=비대칭(1)
    symmetry_label = 0 if main_label == 2 else 1

    # 마무리: 일반=평화(0), 나머지=위협(1)
    ending_label = 0 if main_label == 2 else 1

    return symmetry_label, ending_label
```

### 학습: Multi-Task Loss

```python
def compute_loss(main_logits, sym_logits, end_logits,
                 main_labels, sym_labels, end_labels,
                 alpha=0.3, beta=0.2):
    """
    메인 + 보조 태스크 결합 손실
    alpha: 대등성 가중치
    beta: 마무리 가중치
    """
    main_loss = F.cross_entropy(main_logits, main_labels, label_smoothing=0.1)

    sym_loss = F.cross_entropy(sym_logits, sym_labels) if sym_logits is not None else 0
    end_loss = F.cross_entropy(end_logits, end_labels)

    total = main_loss + alpha * sym_loss + beta * end_loss
    return total
```

---

## Speaker Mask 생성 방법

대화 텍스트에서 화자를 구분하는 방법:

### 방법 A: 문장 부호 기반 (실용적)

```python
def create_speaker_masks(text, tokenizer, max_length):
    """
    마침표/물음표/느낌표로 발화를 분리 → 홀수=A, 짝수=B
    """
    # 발화 분리 (., ?, ! 기준)
    utterances = re.split(r'(?<=[.?!])\s+', text)

    speaker_a_spans = []  # 홀수 발화 (0, 2, 4, ...)
    speaker_b_spans = []  # 짝수 발화 (1, 3, 5, ...)

    current_pos = 0
    for i, utt in enumerate(utterances):
        start = text.find(utt, current_pos)
        end = start + len(utt)
        if i % 2 == 0:
            speaker_a_spans.append((start, end))
        else:
            speaker_b_spans.append((start, end))
        current_pos = end

    # 토큰 레벨로 변환
    encoding = tokenizer(text, max_length=max_length, truncation=True, return_offsets_mapping=True)
    offsets = encoding['offset_mapping']

    mask_a = torch.zeros(max_length, dtype=torch.bool)
    mask_b = torch.zeros(max_length, dtype=torch.bool)

    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for s, e in speaker_a_spans:
            if tok_start >= s and tok_end <= e:
                mask_a[tok_idx] = True
        for s, e in speaker_b_spans:
            if tok_start >= s and tok_end <= e:
                mask_b[tok_idx] = True

    return mask_a, mask_b
```

### 방법 B: 학습 가능한 턴 임베딩 (고급)

```python
# 토큰마다 "몇 번째 턴인지" 임베딩 추가
# KcELECTRA의 token_type_ids를 턴 번호로 활용
# (구현 복잡도 높음 — 방법 A로 시작 후 필요시 전환)
```

---

## 구현 난이도 & 실행 계획

| 구성 요소 | 난이도 | 우선순위 | 비고 |
|-----------|--------|---------|------|
| Ending Pooling | 🟢 쉬움 | 1순위 | attention_mask만으로 구현 가능 |
| Multi-head Loss | 🟢 쉬움 | 1순위 | 보조 라벨 자동 유도 |
| Speaker Mask 생성 | 🟡 중간 | 2순위 | 문장부호 기반 분리 |
| Speaker-Aware Pooling | 🟡 중간 | 2순위 | mask 생성이 핵심 |
| 학습 가능 턴 임베딩 | 🔴 어려움 | 3순위 | 토크나이저 수정 필요 |

### 실행 순서 제안

```
Phase 1: Ending + Multi-head만 먼저 (1~2시간)
  → [CLS] + Ending Pooling + aux loss
  → Speaker 분리 없이도 "마무리 평화" 학습 가능

Phase 2: Speaker-Aware 추가 (2~3시간)
  → 문장부호 기반 화자 분리
  → 대등성 보조 태스크 추가

Phase 3: 결과 비교
  → B04 (데이터 개선만) vs B04+ (데이터 + 모델 구조)
```

---

## 기대 효과

| 현재 실패 패턴 | 이 모델이 해결하는 방법 |
|---------------|----------------------|
| "죽겠다"(비유) → 협박 | Ending Head가 마무리 톤 확인 → 평화로우면 일반 |
| "15만원 분실" → 갈취 | Symmetry Head가 대등 관계 확인 → 대등이면 일반 |
| "대표님 인사" → 괴롭힘 | Symmetry Head가 양쪽 화법 비교 → 자연스러운 비대칭이면 일반 |
| 119 신고 → 협박 | Ending Head가 구조 요청/안내 종료 확인 → 평화 마무리면 일반 |

---

## vs 현재 모델 비교

| | 현재 (B03) | 제안 (Context-Aware) |
|---|---|---|
| 구조 | [CLS] → Linear | [CLS] + Speaker + Ending → Multi-head |
| 학습 신호 | 5클래스 라벨만 | 5클래스 + 대등성 + 마무리 |
| 맥락 이해 | 단어/구두점 패턴 | 화자 관계 + 대화 흐름 |
| 파라미터 증가 | — | ~2% (풀링 레이어 + 보조 헤드) |
| 학습 시간 증가 | — | ~10~20% (보조 태스크 연산) |
