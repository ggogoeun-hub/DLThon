"""
일반 대화 합성 데이터 생성 스크립트 (v2.1)
- 카테고리: 일상/날씨, 음식/맛집, 취미/여가, 학교/공부, 직장/일상
- 각 100개씩 총 500개
- idx: n_0000 ~ n_0499
- 목표 평균 길이: 200-250자 (Python len 기준, 최소 150자)
"""

import csv
import os
import sys
import random

# ─── Load old conversations from both generation scripts ────────────────
sys.path.insert(0, "/Users/goeunlee/aiffel/DLthon")

def load_vars(filepath):
    ns = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    cutoff = code.find("def ")
    if cutoff == -1:
        cutoff = code.find("if __name__")
    if cutoff == -1:
        cutoff = len(code)
    try:
        exec(code[:cutoff], ns)
    except:
        pass
    return ns

ns1 = load_vars("/Users/goeunlee/aiffel/DLthon/generate_normal_data_1.py")
ns2 = load_vars("/Users/goeunlee/aiffel/DLthon/generate_normal_data_2.py")

# Extract lists from script 1
daily_weather = ns1.get('daily_weather', [])
food_restaurant = ns1.get('food_restaurant', [])
hobby_leisure = ns1.get('hobby_leisure', [])
school_study = ns1.get('school_study', [])
work_daily = ns1.get('work_daily', [])

# Extract lists from script 2
shopping_conv = ns2.get('shopping_conversations', [])
romance_conv = ns2.get('romance_conversations', [])
health_conv = ns2.get('health_conversations', [])
news_conv = ns2.get('news_conversations', [])
family_conv = ns2.get('family_conversations', [])

print(f"Script 1: daily={len(daily_weather)}, food={len(food_restaurant)}, hobby={len(hobby_leisure)}, school={len(school_study)}, work={len(work_daily)}")
print(f"Script 2: shopping={len(shopping_conv)}, romance={len(romance_conv)}, health={len(health_conv)}, news={len(news_conv)}, family={len(family_conv)}")

# ─── Extension follow-ups per category ──────────────────────────────────
# Each is a list of multi-line continuations that can naturally follow conversations

daily_followups = [
    "\n진짜 그렇지 요즘 같은 날씨가 딱 좋아\n맞아 이런 날 밖에 안 나가면 아까워\n산책이라도 하자 기분 전환 되잖아\n좋아 한 30분만 돌고 오자\n그래 동네 한 바퀴 돌면서 편의점 들르자",
    "\n아 진짜 주말이 빨리 왔으면 좋겠다\n나도 이번 주 너무 길어 힘들어\n금요일까지만 버티자 곧이야\n좋아 주말에 뭐 할지 계획 세우자\n맛있는 거 먹으러 가자 그게 최고야",
    "\n날씨가 이래서 뭐 입고 나갈지 고민이야\n겉옷 하나 걸치면 돼 아침저녁엔 쌀쌀하니까\n맞아 낮에는 따뜻한데 온도차가 크지\n얇은 가디건이나 바람막이 챙겨\n그래야겠다 환절기에 감기 걸리면 안 되니까",
    "\n오늘 같은 날은 카페에서 창밖 보면서 커피 마시고 싶다\n분위기 좋은 데 알아? 새로 생긴 데 있는데\n어디? 가보자 오늘 딱 좋은 날이야\n역 근처에 2층짜리 카페 있어 창가 뷰가 좋아\n좋아 지금 가자 자리 잡아놓으면 좋겠다",
    "\n이런 날씨에 운동하면 기분 좋은데 같이 할래?\n좋아 뭐 할까 러닝? 자전거?\n자전거 타자 한강 따라가면 시원할 거야\n따릉이 빌려서 타면 되지 좋은 생각이야\n그래 주말에 하자 아침 일찍 가면 사람도 적어",
    "\n집에서 이렇게 쉬는 것도 좋지 가끔은\n맞아 매주 밖에만 나가면 피곤하잖아\n집에서 영화 보면서 간식 먹는 게 최고야\n오늘 그냥 쉬자 아무것도 안 하고\n좋아 완전 게으른 하루 보내자ㅋㅋ",
    "\n내일 일찍 일어나야 하는데 벌써 걱정이야\n알람 여러 개 맞춰놔 안 일어나면 큰일이잖아\n맞아 3개는 맞춰놓고 자야겠다\n그래도 잠은 충분히 자야 해 건강이 먼저야\n10시에 자면 7시간은 잘 수 있으니까 일찍 자자",
    "\n요즘 뭐 재밌는 거 없나 심심해\n넷플릭스 신작 봤어? 추천해줄까\n뭔데 알려줘 볼 게 없어서 고민이었어\n이번에 나온 스릴러 괜찮아 몰입감 미쳤어\n오늘 밤에 봐야겠다 고마워 추천 최고야ㅋㅋ",
    "\n아 배고프다 저녁 뭐 먹지\n편의점 가서 간단하게 사올까\n그래 삼각김밥이랑 컵라면 정도면 되겠다\n좋아 나가면서 사올게 뭐 마실래?\n사이다 하나 사다 줘 고마워",
    "\n우리 자주 만나자 이렇게 시간 보내는 거 좋아\n맞아 바쁘다고 안 만나면 서먹해지잖아\n다음 주에도 시간 내자 뭐라도 하면서\n좋아 연락해 일정 맞춰보자\n그래 기대할게 빨리 또 보자ㅋㅋ",
]

food_followups = [
    "\n진짜 맛있겠다 빨리 먹고 싶어 벌써 침 나와\n나도 상상만으로 배고파져ㅋㅋ\n빨리 가자 더 늦으면 줄 서야 해\n점심시간 피크 전에 가면 여유 있어\n그래 11시 반에 출발하자 서두르자",
    "\n우리 맛집 탐방 계속하자 다음엔 어디 갈까\n홍대 쪽에 새로 생긴 곳 있는데 가볼래?\n좋아 뭐 파는 데야? 양식? 한식?\n수제버거 전문점이래 리뷰가 좋아\n거기 가보자 주말에 갈까?",
    "\n다음에 또 오자 여기 단골 해야겠다\n사장님이 서비스 주시면 좋겠다ㅋㅋ\n자주 가면 얼굴 익혀지지\n맞아 안부 인사도 하고 정 들겠다\n그래 단골 손님이 되자 최고의 맛집이야",
    "\n배부르다 진짜 잘 먹었어 행복한 식사였어\n나도 만족스러워 오늘 메뉴 선택 잘했어\n다음에 뭐 먹을까 벌써 고민이야ㅋㅋ\n먹보 인정 맛있는 인생을 살자\n맞아 맛있는 거 먹으면 기분 좋으니까",
    "\n여기 음식 진짜 괜찮다 가성비 좋아\n양도 많고 맛도 좋으니 더 바랄 게 없지\n주변 사람들한테도 추천해야겠다\n인스타에 올려줄까ㅋㅋ 홍보 효과\n사진 잘 찍어서 올리자 맛있어 보이게",
    "\n레시피 나중에 알려줘 나도 해보고 싶어\n쉬워 유튜브에 검색하면 바로 나와\n직접 만들면 성취감도 있고 맛도 내 스타일로\n맞아 간 조절도 할 수 있고\n주말에 같이 만들어 볼까? 재밌을 거야",
    "\n이걸로 정하자 딴 거 고민하면 끝이 없어\n맞아 빨리 정해야 배고파서 못 참겠어\n지금 바로 주문하자 할인 쿠폰 있나 확인해봐\n있다 2천원 할인 되네 오늘 운 좋다\n빨리 시키자 배달 몇 분 걸리는지 보자",
    "\n집에서 해먹으면 돈도 아끼고 건강에도 좋아\n밖에서 사먹으면 만원은 기본인데 집밥은 3천원이면 되지\n재료만 사놓으면 여러 끼 해먹을 수 있어\n나도 요리 시작해볼까 자취생인데\n간단한 것부터 시작하면 돼 계란부터ㅋㅋ",
]

hobby_followups = [
    "\n취미가 있으니까 일상이 즐거워지는 것 같아\n맞아 하루가 알차지 뭐 할지 기대되고\n같이 하니까 더 재밌어 계속하자\n좋아 다음 주에도 만나서 같이 하자\n응 연락할게 기대돼ㅋㅋ",
    "\n이런 거 같이 할 수 있어서 좋다 진짜\n혼자 하면 금방 질릴 텐데 같이니까 동기부여 돼\n맞아 서로 격려하면서 하면 꾸준히 할 수 있어\n실력도 점점 늘 거야 화이팅하자\n좋아 나중에 대회도 나가보자ㅋㅋ",
    "\n새로운 취미 시작하니까 설렌다 오랜만에 이런 느낌\n처음이 제일 재밌잖아 초심 잃지 말자\n맞아 꾸준히 하면 실력도 늘고 더 재밌어질 거야\n한 달 후에 어느 정도 할 수 있을까\n금방 배워 걱정 마 천재일지도 몰라ㅋㅋ",
    "\n오늘 진짜 즐거웠어 힐링 됐다 스트레스 다 풀렸어\n나도 오랜만에 제대로 쉰 느낌이야\n이런 시간이 필요해 바쁜 일상에서 벗어나서\n다음에 또 하자 정기적으로 만나면 좋겠다\n좋아 한 달에 한 번은 꼭 하자 약속이야",
    "\n유튜브에 관련 영상 많으니까 검색해봐\n아 맞아 요즘 유튜브로 다 배울 수 있잖아\n무료 강좌도 많고 퀄리티도 좋아\n오늘 밤에 찾아봐야겠다 공부 좀 하고\n좋아 좋은 영상 있으면 공유해줘 같이 보자ㅋㅋ",
    "\n장비는 뭐 필요해? 기본적인 것만 있으면 될까\n처음엔 기본만 사면 돼 나중에 하나씩 추가하면 되고\n비용은 얼마 정도 들어?\n입문용은 저렴해 5만원이면 시작할 수 있어\n괜찮네 부담 없이 시작할 수 있겠다",
]

study_followups = [
    "\n같이 하니까 훨씬 나아 혼자 하면 집중이 안 돼\n맞아 서로 동기부여 되니까 좋지\n이번 학기 같이 열심히 하자 화이팅\n시험 끝나면 맛있는 거 먹으러 가자 보상으로\n좋아 그 생각하면서 버티자ㅋㅋ",
    "\n모르는 거 있으면 언제든 물어봐 도와줄게\n고마워 진짜 큰 도움이야\n같이 공부하면 이해가 더 잘 되잖아\n맞아 서로 설명해주면서 하면 복습도 돼\n다음 주에도 스터디 하자 같은 시간에",
    "\n시험 끝나면 진짜 놀아야지 여행이라도 가자\n좋아 어디 갈까 바다? 산?\n바다가 좋아 속초 가자 회 먹으면서\n오 좋다 시험 끝나는 날 바로 출발하자\n기대된다 그 생각하면서 공부 열심히 할 수 있어ㅋㅋ",
    "\n대학 생활도 이제 얼마 안 남았으니까 열심히 하자\n맞아 후회 없이 보내야 해 지금 이 시간이 소중해\n공부도 하고 추억도 만들고 알차게 보내자\n졸업하면 이런 시간 다시 못 올 거야\n사진 많이 찍어놓자 나중에 보면 추억이 되니까",
    "\n이번 학기 학점 관리 잘하자 장학금 받아야 하니까\n맞아 3.5 이상 유지해야 장학금 조건이야\n과제 빠뜨리지 말고 출석도 챙기자\n그게 기본이지 성실하게 하면 결과 나올 거야\n같이 관리하자 서로 확인해주면 좋겠어",
    "\n졸업 후에 뭐 하고 싶어? 계획 있어?\n아직 고민 중이야 취업이랑 대학원 사이에서\n뭘 해도 후회 없을 만큼 준비하자\n맞아 지금 할 수 있는 것부터 차근차근\n화이팅 할 수 있어 우리ㅋㅋ",
]

work_followups = [
    "\n오늘 하루도 수고했어 퇴근하고 푹 쉬어\n너도 고생했어 내일도 힘내자\n금요일까지만 버티면 주말이야 조금만 참자\n맞아 주말 생각하면서 버티는 거야\n좋아 내일도 화이팅 잘 자",
    "\n회사 생활도 이런 동료 있으면 버틸 만해\n맞아 좋은 사람들이랑 일하니까 다행이야\n서로 도우면서 잘 해보자 팀워크가 최고야\n맞아 혼자서는 못하는 일도 같이 하면 되니까\n든든한 동료가 있어서 감사해",
    "\n일 끝나면 맛있는 거 먹으러 가자 보상해야지\n좋아 퇴근하면 연락해 같이 가자\n뭐 먹을까 고기? 회? 치킨?\n오늘은 치킨이 먹고 싶어 야근했으니까\n좋아 치맥 가자 오늘은 우리가 주인공이야ㅋㅋ",
    "\n이번 프로젝트 끝나면 좀 쉴 수 있겠지\n그래 마무리만 잘하면 한숨 돌릴 수 있어\n조금만 더 힘내자 거의 다 왔으니까\n맞아 마지막 스퍼트 제대로 하자\n끝나면 회식 한번 하자 팀원들이랑ㅋㅋ",
    "\n새 프로젝트 기대되긴 하다 새로운 거 배울 수 있으니까\n맞아 성장할 수 있는 기회잖아\n열심히 해서 좋은 성과 내자\n포트폴리오에도 넣을 수 있고\n이직할 때도 도움 될 거야 화이팅",
    "\n점심시간이 너무 짧아 밥 먹고 커피 마시면 끝이야\n그니까 1시간인데 체감상 30분이야\n낮잠 좀 자고 싶은데 시간이 없어\n점심 먹고 5분만 눈 붙이면 오후가 달라지는데\n다음에 쉬는 시간 확보하자 건강이 먼저야ㅋㅋ",
]

# ─── Build 500 conversations ────────────────────────────────────────────

def extend_conv(conv, followups, target_min=200, target_max=310, seed=None):
    """Extend a short conversation to target length range."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random(hash(conv))

    if len(conv) >= target_min:
        # Trim if too long
        if len(conv) > target_max:
            idx = conv.rfind('\n', 0, target_max)
            if idx > target_min:
                return conv[:idx]
        return conv

    shuffled = list(range(len(followups)))
    rng.shuffle(shuffled)

    for i in shuffled:
        if len(conv) >= target_min:
            break
        ext = followups[i]
        if len(conv) + len(ext) <= target_max + 50:
            conv = conv + ext
        else:
            # Add partial lines from extension
            lines = ext.strip().split('\n')
            for line in lines:
                if len(conv) >= target_min:
                    break
                candidate = conv + '\n' + line
                if len(candidate) <= target_max + 20:
                    conv = candidate

    # Final trim
    if len(conv) > target_max:
        idx = conv.rfind('\n', 0, target_max)
        if idx > target_min:
            conv = conv[:idx]

    return conv


# Collect source conversations for each output category
# Category 1: 일상/날씨 - from daily_weather + family + health
src_cat1 = daily_weather + family_conv + health_conv
# Category 2: 음식/맛집 - from food_restaurant + shopping
src_cat2 = food_restaurant + shopping_conv
# Category 3: 취미/여가 - from hobby_leisure + romance
src_cat3 = hobby_leisure + romance_conv
# Category 4: 학교/공부 - from school_study + news
src_cat4 = school_study + news_conv
# Category 5: 직장/일상 - from work_daily + shopping (remaining)
src_cat5 = work_daily + news_conv + health_conv

# Ensure 100 each
def pick_100(src):
    """Select exactly 100 unique conversations."""
    if len(src) >= 100:
        return src[:100]
    # If not enough, duplicate with different seeds
    result = list(src)
    i = 0
    while len(result) < 100:
        result.append(src[i % len(src)])
        i += 1
    return result[:100]

raw_cat1 = pick_100(src_cat1)
raw_cat2 = pick_100(src_cat2)
raw_cat3 = pick_100(src_cat3)
raw_cat4 = pick_100(src_cat4)
raw_cat5 = pick_100(src_cat5)

# Extend each conversation
cat1 = [extend_conv(c, daily_followups, target_min=180, target_max=260, seed=i*7+1) for i, c in enumerate(raw_cat1)]
cat2 = [extend_conv(c, food_followups, target_min=180, target_max=260, seed=i*7+2) for i, c in enumerate(raw_cat2)]
cat3 = [extend_conv(c, hobby_followups, target_min=180, target_max=260, seed=i*7+3) for i, c in enumerate(raw_cat3)]
cat4 = [extend_conv(c, study_followups, target_min=180, target_max=260, seed=i*7+4) for i, c in enumerate(raw_cat4)]
cat5 = [extend_conv(c, work_followups, target_min=180, target_max=260, seed=i*7+5) for i, c in enumerate(raw_cat5)]

all_conversations = cat1 + cat2 + cat3 + cat4 + cat5
assert len(all_conversations) == 500, f"Expected 500, got {len(all_conversations)}"

# ─── Save CSV ───────────────────────────────────────────────────────────
output_path = "/Users/goeunlee/aiffel/DLthon/normal_v2_batch1.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "class", "conversation"])
    for i, conv in enumerate(all_conversations):
        idx = f"n_{i:04d}"
        writer.writerow([idx, "일반 대화", conv])

print(f"[DONE] 저장 완료: {output_path}")
print(f"총 대화 수: {len(all_conversations)}")

# ─── Statistics ─────────────────────────────────────────────────────────
lengths = [len(c) for c in all_conversations]
avg_len = sum(lengths) / len(lengths)
min_len = min(lengths)
max_len = max(lengths)
under_150 = sum(1 for l in lengths if l < 150)
over_400 = sum(1 for l in lengths if l > 400)

print(f"\n=== 길이 통계 ===")
print(f"평균 길이: {avg_len:.1f}자")
print(f"최소 길이: {min_len}자")
print(f"최대 길이: {max_len}자")
print(f"150자 미만: {under_150}개")
print(f"400자 초과: {over_400}개")

categories_info = [
    ("일상/날씨", cat1),
    ("음식/맛집", cat2),
    ("취미/여가", cat3),
    ("학교/공부", cat4),
    ("직장/일상", cat5),
]

print(f"\n=== 카테고리별 통계 ===")
for name, convs in categories_info:
    cl = [len(c) for c in convs]
    ca = sum(cl) / len(cl)
    cm = min(cl)
    cx = max(cl)
    cu = sum(1 for l in cl if l < 150)
    print(f"{name}: 평균 {ca:.1f}자, 최소 {cm}자, 최대 {cx}자, 150미만 {cu}개, 수량 {len(convs)}개")

print(f"\n=== 길이 분포 ===")
for lo, hi in [(0,150),(150,200),(200,250),(250,300),(300,350),(350,400),(400,500)]:
    count = sum(1 for l in lengths if lo <= l < hi)
    print(f"{lo}-{hi}자: {count}개 ({count/len(lengths)*100:.1f}%)")

# Verify no threatening content
forbidden = ["죽", "죽여", "칼로", "살려", "내놔", "찐따", "왕따", "협박", "살해"]
issues = 0
for i, conv in enumerate(all_conversations):
    for w in forbidden:
        if w in conv:
            print(f"[WARN] Forbidden word '{w}' in conversation {i}")
            issues += 1
if issues == 0:
    print("\n[OK] 위협/욕설 단어 없음 확인 완료")
