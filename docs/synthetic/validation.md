# 합성 데이터 품질 검증 기준

> 생성된 합성 일반 대화가 4축 가이드라인을 만족하는지 자동 검증

---

## 검증 항목 체크리스트

### Pass/Fail 기준

| 검증 항목 | Pass 기준 | Fail 시 조치 |
|---|---|---|
| **글자 수** | 150~400자 | 재생성 |
| **발화 턴 수** | 8~12턴 | 재생성 |
| **금지 단어** | BANNED_WORDS 미포함 | 즉시 폐기 |
| **이모티콘** | ㅋ/ㅎ/ㅠ/ㅜ 2연속 없음 | 정규식 제거 후 재검증 |
| **말줄임** | ... 없음 | 제거 후 재검증 |
| **발화 대칭비** | < 1.5 | 재생성 |
| **존댓말 비대칭** | < 0.3 | 재생성 |
| **굴복 표현** | < 1회/대화 | 재생성 |
| **비속어** | 0회 | 즉시 폐기 |

### 통계 검증 (배치 단위)

생성된 전체 합성 데이터셋에 대해:

| 지표 | 목표 범위 | 비교 대상 |
|---|---|---|
| 발화 대칭비 mean | 1.0~1.3 | 위협 평균 1.84~2.00 |
| 비대칭(>1.5) 비율 | < 20% | 위협 53~60% |
| 존댓말 비대칭 mean | < 0.15 | 위협 0.24~0.38 |
| 강한 비대칭(>0.3) 비율 | < 10% | 위협 42~55% |
| 에스컬레이션(>0) 비율 | < 25% | 위협 33~59% |
| 굴복/대등 비율 | < 0.1 | 위협 0.41~1.02 |
| 느낌표 평균 | 0.5~2.0 | 위협 0.70~1.85 |
| 물음표 평균 | 3.0~5.5 | 위협 4.08~4.87 |
| 마침표 평균 | 3.0~8.0 | 위협 ~6.18 |

---

## 자동 검증 코드

```python
import re
import numpy as np

BANNED_WORDS = ['죽여', '죽어', '맞을래', '시키는 대로', '내놔', '닥쳐', '패버릴']
PROFANITY = ['씨발', '병신', '개새끼', '미친놈', '꺼져']
EMO_RE = re.compile(r'[ㅋㅎㅠㅜ]{2,}')
ELLIPSIS_RE = re.compile(r'\.{2,}')

SUBMISSIVE = ['죄송합니다', '잘못했습니다', '살려주세요', '제발', '용서', '안됩니다', '못합니다']
POLITE = ['습니다', '세요', '에요', '해요', '하세요', '입니다', '합니다', '까요', '나요', '죠']
CASUAL = ['해', '야', '어', '지', '냐', '라', '거든', '잖아', '는데']


def validate_single(text):
    """단일 대화 검증. (pass, issues) 반환"""
    issues = []
    
    # 1. 길이
    if len(text) < 150 or len(text) > 400:
        issues.append(f'길이: {len(text)}자')
    
    # 2. 턴 수
    turns = [t.strip() for t in text.split('\n') if t.strip()]
    if len(turns) < 6 or len(turns) > 14:
        issues.append(f'턴수: {len(turns)}')
    
    # 3. 금지 단어
    for word in BANNED_WORDS + PROFANITY:
        if word in text:
            issues.append(f'금지어: {word}')
    
    # 4. 이모티콘
    if EMO_RE.search(text):
        issues.append('이모티콘 포함')
    
    # 5. 말줄임
    if ELLIPSIS_RE.search(text):
        issues.append('말줄임 포함')
    
    # 6. 발화 대칭비
    if len(turns) >= 2:
        odd = [len(turns[i]) for i in range(0, len(turns), 2)]
        even = [len(turns[i]) for i in range(1, len(turns), 2)]
        if even:
            a, b = np.mean(odd), np.mean(even)
            if min(a, b) > 0:
                ratio = max(a, b) / min(a, b)
                if ratio > 1.5:
                    issues.append(f'발화 비대칭: {ratio:.2f}')
    
    # 7. 존댓말 비대칭
    if len(turns) >= 4:
        def polite_ratio(tlist):
            p = sum(1 for t in tlist for e in POLITE if t.endswith(e))
            c = sum(1 for t in tlist for e in CASUAL if t.endswith(e))
            return p / (p + c) if (p + c) > 0 else 0.5
        odd_t = [turns[i] for i in range(0, len(turns), 2)]
        even_t = [turns[i] for i in range(1, len(turns), 2)]
        asym = abs(polite_ratio(odd_t) - polite_ratio(even_t))
        if asym > 0.3:
            issues.append(f'존댓말 비대칭: {asym:.2f}')
    
    # 8. 굴복 표현
    sub_count = sum(1 for s in SUBMISSIVE if s in text)
    if sub_count >= 1:
        issues.append(f'굴복 표현 {sub_count}회')
    
    return len(issues) == 0, issues


def validate_batch(df, text_col='conversation'):
    """배치 검증. 통과율 및 이슈 통계 출력"""
    results = df[text_col].apply(validate_single)
    passed = results.apply(lambda x: x[0]).sum()
    total = len(df)
    
    print(f'통과: {passed}/{total} ({passed/total*100:.1f}%)')
    
    # 이슈 빈도
    all_issues = []
    for _, issues in results:
        all_issues.extend(issues)
    
    if all_issues:
        from collections import Counter
        print('\n이슈 빈도:')
        for issue, count in Counter(all_issues).most_common(10):
            print(f'  {issue}: {count}건')
    
    return results
```

---

## 사용법

```python
import pandas as pd

synthetic = pd.read_csv('synthetic_normal_combined.csv')
results = validate_batch(synthetic, text_col='conversation')
```
