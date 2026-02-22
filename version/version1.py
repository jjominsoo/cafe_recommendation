import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [DATA LOADING] ---
@st.cache_data
def load_data():
    # 파일 읽기
    df = pd.read_csv('cafe_pca.csv') 
    
    # 기존에 similarity 칼럼이 있다면 0으로 초기화 (계산 전 깨끗하게 비움)
    if 'similarity' in df.columns:
        df['similarity'] = 0.0
    return df

df_reduced = load_data()

# --- [PC MAP] ---
pc_map = {
    1: {'pos': ['fun_various', 'air_condition', 'swimming_pool', 'clean2'], 'neg': ['deafening']},
    2: {'pos': ['clean', 'friendly', 'toilet', 'talk', 'seat'], 'neg': []},
    3: {'pos': ['detail_explain', 'class_time', 'one_day_class', 'private', 'recommend'], 'neg': []},
    4: {'pos': ['package2', 'sensory', 'flower', 'read_book', 'fresh2'], 'neg': []},
    5: {'pos': ['food', 'plenty', 'fresh', 'group', 'big'], 'neg': []},
    6: {'pos': ['parent', 'theme', 'experience', 'play_various', 'space'], 'neg': []},
    7: {'pos': ['clean_facility2', 'personal_space', 'rest_facility', 'quiet', 'atmosphere'], 'neg': []},
    8: {'pos': ['side_dish', 'live', 'long', 'alcohol_alone', 'alcohol'], 'neg': []},
    9: {'pos': ['promotion_product_various', 'product_various', 'discount', 'trendy_product'], 'neg': ['easy']},
    10: {'pos': ['easy', 'group2', 'custom_class', 'design', 'order_made'], 'neg': []},
    11: {'pos': ['game_various', 'seat_space', 'plenty_food', 'clean_facility', 'group'], 'neg': []},
    12: {'pos': ['outside', 'menu', 'various', 'concept', 'fast'], 'neg': []},
    13: {'pos': ['worth_cost', 'kid'], 'neg': ['various', 'fast', 'concept']},
    14: {'pos': ['special_product', 'comfort', 'various'], 'neg': ['book', 'visual']},
    15: {'pos': ['visual', 'set_composition', 'special_product', 'comfort', 'resonable_price'], 'neg': []},
    16: {'pos': ['book', 'outside', 'view'], 'neg': ['worth_cost', 'kid']},
    17: {'pos': ['fare', 'atmosphere', 'room'], 'neg': ['fast', 'various']},
    18: {'pos': ['fare', 'various', 'fast'], 'neg': ['healthy_taste', 'menu']},
    19: {'pos': ['bread', 'special'], 'neg': ['concentrate', 'fast', 'various']},
    20: {'pos': ['party', 'music', 'pet'], 'neg': ['healthy_taste', 'menu']},
    21: {'pos': ['safe', 'theme'], 'neg': ['play_various', 'space', 'resonable_price']},
    22: {'pos': ['side_dish2', 'calm', 'bread', 'parking'], 'neg': ['alone']},
    23: {'pos': ['order_made', 'design', 'present'], 'neg': ['cost', 'bread']},
    24: {'pos': ['order_made', 'cost', 'bread'], 'neg': ['photo', 'dessert']},
    25: {'pos': ['tea', 'calm', 'alone', 'package'], 'neg': ['concept']},
    26: {'pos': ['book_various', 'room_space', 'play_various'], 'neg': ['pet_environment', 'tea']},
    27: {'pos': ['tea', 'concept', 'room'], 'neg': ['package', 'present']},
    28: {'pos': ['room_space', 'pet_environment', 'plenty_food'], 'neg': ['book_various', 'seat_space']},
    29: {'pos': ['pet', 'cost', 'room', 'dessert'], 'neg': ['bread']},
    30: {'pos': ['pet', 'bread'], 'neg': ['cost', 'music', 'dessert']},
    31: {'pos': ['pet_environment', 'book_various', 'safe'], 'neg': ['experience', 'space']},
    32: {'pos': ['room', 'package'], 'neg': ['pet', 'side_dish2', 'fare']},
    33: {'pos': ['tea'], 'neg': ['room', 'cozy', 'calm', 'alone']},
    34: {'pos': ['side_dish2', 'alone', 'present'], 'neg': ['cozy', 'special_day']},
    35: {'pos': ['party'], 'neg': ['concept', 'present', 'pet', 'alcohol_alone']},
    36: {'pos': ['parking', 'dessert', 'present', 'cozy'], 'neg': ['package']}
}

# --- [UI: TITLE] ---
st.title("☕ AI 취향 저격 카페 추천")
st.markdown("당신의 성향을 분석하여 최적의 카페를 찾아드립니다.")

# --- [STEP 1: SURVEY] ---
with st.form("survey_form"):
    st.subheader("1. 어떤 시간을 보내고 싶나요?")
    persona_choice = st.radio(
        "가장 끌리는 목적을 골라주세요",
        ["몰입과 영감 (조용, 사색)", "장인의 맛 (빵, 시그니처)", "체험과 배움", "비주얼/SNS", "음악과 사교"]
    )
    
    st.subheader("2. 추가 고려 사항 (중복 가능)")
    filter_choices = st.multiselect(
        "해당하는 것을 선택하세요",
        ["반려동물 동반", "아이와 함께", "단체 모임", "가성비 중요"]
    )
    
    st.subheader("3. 필수 편의 시설")
    conveni_choices = st.multiselect(
        "포기할 수 없는 시설은?",
        ["주차장 필수", "깨끗한 화장실/서비스", "야외 테라스/개방감"]
    )
    
    submitted = st.form_submit_button("나만의 카페 찾기")

# --- [RECOMMENDATION FUNCTION] ---
def recommend_cafes(user_answers, df_reduced, pc_map, top_n=5):
    # 36차원 유저 벡터 생성
    user_vector = np.zeros(36)
    for pc_num, columns in pc_map.items():
        idx = pc_num - 1
        for word in user_answers:
            if word in columns['pos']:
                user_vector[idx] += 1.0
            elif word in columns['neg']:
                user_vector[idx] -= 1.0

    # 카페 데이터(PC1~PC36)만 추출
    # 열 이름이 'PC1', 'PC2'... 와 같이 시작한다고 가정합니다.
    pc_cols = [f'PC{i}' for i in range(1, 37)]
    cafe_features = df_reduced[pc_cols].values

    # 코사인 유사도 계산
    similarities = cosine_similarity(user_vector.reshape(1, -1), cafe_features).flatten()

    # 결과 정렬 및 반환
    df_result = df_reduced.copy()
    df_result['similarity'] = similarities
    return df_result.sort_values(by='similarity', ascending=False).head(top_n)

# --- [STEP 2: RECOMMENDATION LOGIC] ---
if submitted:
    # 1. 유저 답변을 데이터 키워드로 매핑 (이 부분이 추가되어야 계산이 됩니다)
    user_keywords = []
    
    # Q1 매핑
    persona_map = {
        "몰입과 영감 (조용, 사색)": ['quiet', 'alone', 'calm', 'view', 'tea'],
        "장인의 맛 (빵, 시그니처)": ['bread', 'special_product', 'dessert', 'fresh'],
        "체험과 배움": ['one_day_class', 'experience', 'custom_class', 'design'],
        "비주얼/SNS": ['visual', 'photo', 'concept', 'flower', 'package'],
        "음악과 사교": ['music', 'party', 'live', 'alcohol', 'talk']
    }
    user_keywords.extend(persona_map.get(persona_choice, []))
    
    # Q2 매핑
    filter_map = {
        "반려동물 동반": ['pet', 'pet_environment'],
        "아이와 함께": ['kid', 'parent', 'safe'],
        "단체 모임": ['group', 'group2', 'seat_space'],
        "가성비 중요": ['worth_cost', 'resonable_price', 'discount']
    }
    for choice in filter_choices:
        user_keywords.extend(filter_map.get(choice, []))
        
    # Q3 매핑
    conveni_map = {
        "주차장 필수": ['parking'],
        "깨끗한 화장실/서비스": ['clean', 'friendly', 'toilet'],
        "야외 테라스/개방감": ['outside', 'fresh2']
    }
    for choice in conveni_choices:
        user_keywords.extend(conveni_map.get(choice, []))

    # 2. 추천 함수 호출
    # 결과로 데이터프레임이 반환됩니다.
    result_df = recommend_cafes(user_keywords, df_reduced, pc_map, top_n=5)
    
    # 3. 결과 리포트 출력
    st.balloons()
    st.header("🎯 당신을 위한 분석 결과")
    
    for i, (idx, row) in enumerate(result_df.iterrows()):
        # row['열1'] 부분은 실제 카페 이름 컬럼명으로 수정하세요 (예: row['store_name'])
        with st.expander(f"{i+1}위: {row['열1']} (매치율 {row['similarity']*100:.1f}%)"):
            st.write(f"**유사도 점수:** {row['similarity']:.4f}")
            st.write("당신의 취향 벡터와 가장 유사한 데이터 패턴을 가진 카페입니다.")