import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [페이지 설정 및 스타일] ---
st.set_page_config(page_title="AI Cafe Finder", layout="centered")

# CSS: 선택된 버튼과 일반 버튼을 시각적으로 구분
st.markdown("""
    <style>
    /* 기본 버튼 스타일 */
    div.stButton > button {
        width: 100%;
        height: 120px;
        font-size: 22px !important;
        font-weight: bold;
        border-radius: 20px;
        background-color: #ffffff;
        border: 2px solid #f0f2f6;
        transition: all 0.2s ease;
    }
    /* 마우스를 올렸을 때 */
    div.stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    /* 보기가 선택되었을 때 강조하기 위한 보조 스타일 (st.info 활용) */
    .selection-tag {
        color: #ff4b4b;
        font-weight: bold;
        background: #fff0f0;
        padding: 5px 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [DATA & PC MAP] ---
@st.cache_data
def load_data():
    df = pd.read_csv('cafe_pca.csv')
    if 'similarity' in df.columns:
        df['similarity'] = 0.0
    return df

df_reduced = load_data()

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

# --- [SESSION STATE] ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'selections' not in st.session_state:
    st.session_state.selections = {"persona": None, "filters": [], "conveni": []}

# --- [RECOMMENDATION FUNCTION] ---
def recommend_cafes(user_answers, df_reduced, pc_map, top_n=5):
    user_vector = np.zeros(36)
    for pc_num, columns in pc_map.items():
        idx = pc_num - 1
        for word in user_answers:
            if word in columns['pos']:
                user_vector[idx] += 1.0
            elif word in columns['neg']:
                user_vector[idx] -= 1.0
    pc_cols = [f'PC{i}' for i in range(1, 37)]
    cafe_features = df_reduced[pc_cols].values
    similarities = cosine_similarity(user_vector.reshape(1, -1), cafe_features).flatten()
    df_result = df_reduced.copy()
    df_result['similarity'] = similarities
    return df_result.sort_values(by='similarity', ascending=False).head(top_n)

# --- [UI 메인 타이틀] ---
st.title("☕ 나만의 취향 카페 찾기")
st.progress(min(st.session_state.step / 3, 1.0))

# --- [STEP 1: PERSONA (단일 선택 + Toggle)] ---
if st.session_state.step == 1:
    st.subheader("Q1. 오늘 어떤 시간을 보내고 싶나요?")
    
    options = ["몰입과 영감 (조용, 사색)", "장인의 맛 (빵, 시그니처)", "체험과 배움", "비주얼/SNS"]
    icons = ["🧘", "🍞", "🎨", "📸"]
    
    c1, c2 = st.columns(2)
    for i, opt in enumerate(options):
        # 이미 선택된 상태라면 이모지 변경 및 표시
        is_selected = st.session_state.selections['persona'] == opt
        label = f"{icons[i]} (선택됨)\n{opt}" if is_selected else f"{icons[i]}\n{opt}"
        
        target_col = c1 if i % 2 == 0 else c2
        if target_col.button(label, key=f"p_{i}"):
            # 토글 로직: 이미 선택된 걸 누르면 해제, 아니면 선택
            st.session_state.selections['persona'] = None if is_selected else opt
            st.rerun()

    if st.session_state.selections['persona']:
        col_space, col_next = st.columns([4, 1])
        if col_next.button("다음 ➔"):
            st.session_state.step = 2
            st.rerun()

# --- [STEP 2: FILTER (다중 선택 + Toggle)] ---
elif st.session_state.step == 2:
    st.subheader("Q2. 추가로 고려해야 할 상황이 있나요?")
    
    options = {"반려동물 동반": "🐶", "아이와 함께": "👶", "단체 모임": "👥", "가성비 중요": "💰"}
    c1, c2 = st.columns(2)
    
    for i, (opt, icon) in enumerate(options.items()):
        is_selected = opt in st.session_state.selections['filters']
        label = f"✅ {icon}\n{opt}" if is_selected else f"{icon}\n{opt}"
        
        target_col = c1 if i % 2 == 0 else c2
        if target_col.button(label, key=f"f_{i}"):
            if is_selected:
                st.session_state.selections['filters'].remove(opt)
            else:
                st.session_state.selections['filters'].append(opt)
            st.rerun()

    col_back, col_space, col_next = st.columns([1, 3, 1])
    if col_back.button("⬅️ 이전"):
        st.session_state.step = 1
        st.rerun()
    if col_next.button("다음 ➔"):
        st.session_state.step = 3
        st.rerun()

# --- [STEP 3: CONVENIENCE (다중 선택 + Toggle)] ---
elif st.session_state.step == 3:
    st.subheader("Q3. 포기할 수 없는 '편의 시설'은?")
    
    options = {"주차장 필수": "🚗", "깨끗한 화장실/서비스": "🚻", "야외 테라스/개방감": "🌿"}
    c1, c2 = st.columns(2)
    
    for i, (opt, icon) in enumerate(options.items()):
        is_selected = opt in st.session_state.selections['conveni']
        label = f"✅ {icon}\n{opt}" if is_selected else f"{icon}\n{opt}"
        
        target_col = c1 if i % 2 == 0 else c2
        if target_col.button(label, key=f"c_{i}"):
            if is_selected:
                st.session_state.selections['conveni'].remove(opt)
            else:
                st.session_state.selections['conveni'].append(opt)
            st.rerun()

    col_back, col_space, col_done = st.columns([1, 2, 2])
    if col_back.button("⬅️ 이전"):
        st.session_state.step = 2
        st.rerun()
    if col_done.button("✅ 결과 분석하기"):
        st.session_state.step = 4
        st.rerun()

# --- [STEP 4: RESULT PAGE] ---
elif st.session_state.step == 4:
    st.balloons()
    
    # 키워드 매핑
    user_keywords = []
    
    persona_map = {
        "몰입과 영감 (조용, 사색)": ['quiet', 'alone', 'calm', 'view', 'tea'],
        "장인의 맛 (빵, 시그니처)": ['bread', 'special_product', 'dessert', 'fresh'],
        "체험과 배움": ['one_day_class', 'experience', 'custom_class', 'design'],
        "비주얼/SNS": ['visual', 'photo', 'concept', 'flower', 'package']
    }
    user_keywords.extend(persona_map.get(st.session_state.selections['persona'], []))
    
    filter_map = {
        "반려동물 동반": ['pet', 'pet_environment'],
        "아이와 함께": ['kid', 'parent', 'safe'],
        "단체 모임": ['group', 'group2', 'seat_space'],
        "가성비 중요": ['worth_cost', 'resonable_price', 'discount']
    }
    for f in st.session_state.selections['filters']:
        user_keywords.extend(filter_map.get(f, []))
        
    conveni_map = {
        "주차장 필수": ['parking'],
        "깨끗한 화장실/서비스": ['clean', 'friendly', 'toilet'],
        "야외 테라스/개방감": ['outside', 'fresh2']
    }
    for c in st.session_state.selections['conveni']:
        user_keywords.extend(conveni_map.get(c, []))

    result_df = recommend_cafes(user_keywords, df_reduced, pc_map, top_n=5)
    
    st.header("🎯 당신을 위한 분석 결과")
    for i, (idx, row) in enumerate(result_df.iterrows()):
        # row['열1']을 실제 카페명 컬럼으로 변경하세요
        with st.expander(f"{i+1}위: {row['열1']} (매치율 {row['similarity']*100:.1f}%)"):
            st.write(f"**유사도:** {row['similarity']:.4f}")
            st.write("분석된 유저님의 취향에 가장 부합하는 공간입니다.")

    if st.button("🔄 다시 테스트하기"):
        st.session_state.step = 1
        st.session_state.selections = {"persona": None, "filters": [], "conveni": []}
        st.rerun()