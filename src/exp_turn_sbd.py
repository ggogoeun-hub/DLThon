"""EXP-A: 다국어 문장 경계 탐지 모델 (SBD)"""
from punctuators.models.punc_cap_seg_model import PunctCapSegModelONNX as PunctuationModel
from exp_turn_common import *

print('='*60)
print('EXP-A: 다국어 SBD 모델 (1-800-BAD-CODE)')
print('='*60)

model = PunctuationModel.from_pretrained("1-800-BAD-CODE/sentence_boundary_detection_multilang")
rows = load_data()

all_p, all_r, all_f = [], [], []

for row in rows:
    flat = flatten(row['conversation'])
    actual = get_actual_boundaries(row['conversation'])

    # SBD 모델로 문장 경계 예측
    result = model.infer([flat])
    predicted_text = result[0] if result else flat

    # 예측된 문장 경계를 단어 인덱스로 변환
    # 모델 출력에서 구두점이 삽입된 위치를 찾음
    flat_words = flat.split()
    predicted_boundaries = []

    # 모델은 구두점을 삽입하므로, 삽입된 위치를 추적
    pred_words = predicted_text.split()
    flat_idx = 0
    for pw in pred_words:
        clean_pw = strip_punct(pw)
        if not clean_pw:
            continue
        if flat_idx < len(flat_words):
            flat_idx += 1
        # 구두점이 붙어있으면 문장 경계
        if pw != clean_pw and flat_idx > 0:
            predicted_boundaries.append(flat_idx)

    p, r, f = print_result(row['idx'], row['class'], flat, predicted_boundaries, actual)
    if actual:  # 정답 있는 것만 집계
        all_p.append(p); all_r.append(r); all_f.append(f)

print(f'\n{"="*60}')
print(f'EXP-A 위협 데이터 평균: P={sum(all_p)/len(all_p):.3f} R={sum(all_r)/len(all_r):.3f} F1={sum(all_f)/len(all_f):.3f}')
print(f'{"="*60}')
