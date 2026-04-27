"""턴 탐지 실험 공통 유틸"""
import csv
import re

DATA_PATH = 'data/data-exp/trun-split-exp/turn_split_test_25.csv'

def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def strip_punct(text):
    return re.sub(r'\s+', ' ', re.sub(r'[.?!]', '', text)).strip()

def flatten(conversation):
    """원본 대화 → 구두점 제거 + \n 제거 flat text"""
    return strip_punct(conversation.replace('\n', ' '))

def get_actual_boundaries(conversation):
    """원본 \n 대화에서 턴 경계의 단어 인덱스 리스트 반환"""
    turns = [t.strip() for t in conversation.split('\n') if t.strip()]
    if len(turns) <= 1:
        return []
    boundaries = []
    word_idx = 0
    for turn in turns[:-1]:
        clean = strip_punct(turn)
        word_idx += len(clean.split())
        boundaries.append(word_idx)
    return boundaries

def evaluate(predicted, actual, tolerance=2):
    if not predicted or not actual:
        return 0, 0, 0
    correct = 0
    matched_actual = set()
    for pred in predicted:
        for i, act in enumerate(actual):
            if abs(pred - act) <= tolerance and i not in matched_actual:
                correct += 1
                matched_actual.add(i)
                break
    precision = correct / len(predicted) if predicted else 0
    recall = correct / len(actual) if actual else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def print_result(idx, cls, flat_text, predicted, actual):
    words = flat_text.split()
    print(f'\n[{idx}] {cls}')
    print(f'  실제 경계: {actual}')
    print(f'  예측 경계: {predicted}')
    p, r, f = evaluate(predicted, actual)
    print(f'  P={p:.2f} R={r:.2f} F1={f:.2f}')
    # 시각화
    line = ''
    for i, w in enumerate(words):
        if i in set(predicted):
            line += ' [P] '
        if actual and i in set(actual):
            line += ' [A] '
        line += w + ' '
    print(f'  {line[:200]}...' if len(line) > 200 else f'  {line}')
    return p, r, f
