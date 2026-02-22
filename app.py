import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- [페이지 설정 및 스타일] ---
st.set_page_config(page_title="Cafe Finder Pro", layout="wide") # 넓은 화면 사용


# NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"] # 발급받은 ID 입력
# NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"] # 발급받은 Secret 입력
NAVER_CLIENT_ID = "h5Boba0NG1huDKOpvL6O" # 발급받은 ID 입력
NAVER_CLIENT_SECRET = "4vslkqoNEF" # 발급받은 Secret 입력
# CSS: 선택된 버튼과 일반 버튼을 시각적으로 구분
st.markdown("""     
    <style>
    /* 전체 여백 줄이기 */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }
    /* 기본 버튼 스타일 */
    div.stButton > button {
        width: 100% !important;
        height: 80px !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        white-space: pre-wrap !important;
        line-height: 1.2 !important;
        border-radius: 15px !important;
        font-size: 16px !important;
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
    /* 선택된 버튼 (Primary 타입) 스타일 */
    div.stButton > button[kind="primary"] {
        border: 3px solid #ff4b4b !important;
        color: #ff4b4b !important;
        background-color: #fff5f5 !important;
    }
    /* 선택 안 된 버튼 (Secondary 타입) 스타일 */
    div.stButton > button[kind="secondary"] {
        border: 2px solid #f0f2f6 !important;
        color: #31333F !important;
        background-color: white !important;
    }
    /* 이미지 카드 스타일 */
    .cafe-img {
        border-radius: 10px;
        width: 50%;
        height: 200px;
        object-fit: cover;
        margin-bottom: 15px;
    }
    /* 결과 페이지 상세 정보 카드 */
    .detail-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #ff4b4b;
        font-size: 0.9rem;
    }
            
    /* 팝업 내부 이미지 스타일 */
    .popup-img {
        border-radius: 15px;
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
    }

    /* 상세 설명 텍스트 박스 */
    .popup-desc {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
        height: 100%;
    }
            
    /* 헤더 여백 조절 */
    h1 { font-size: 2rem !important; padding-bottom: 0.5rem; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.2rem !important; }
                 
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

def get_naver_info(query):
    """네이버 지역 검색 API를 통해 카페 정보를 가져옵니다."""
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    params = {"query": query, "display": 1}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        items = res.json().get('items')
        return items[0] if items else None
    return None

def get_naver_image(query):
    """이미지 검색 API: 카페 외관/내부 사진"""
    url = "https://openapi.naver.com/v1/search/image"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    # 더 정확한 사진을 위해 '카페' 키워드 추가
    params = {"query": query + " 카페", "display": 1, "sort": "sim"}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        items = res.json().get('items')
        return items[0]['link'] if items else None
    return None

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

# --- [상세 정보 팝업 함수] ---
@st.dialog("카페 상세 정보", width="large")
def show_cafe_detail(cafe_name):
    info = get_naver_info(cafe_name)
    img_url = get_naver_image(cafe_name)
    
    if info:
        title = info['title'].replace('<b>', '').replace('</b>', '')
        col_img, col_txt = st.columns([1, 1]) # 5:5 분할
        
        with col_img:
            if img_url:
                st.image(img_url, use_container_width=True)
            else:
                st.info("이미지를 불러올 수 없습니다.")
        
        with col_txt:
            st.markdown(f"### {title}")
            st.markdown(f"**📍 주소**\n{info['address']}")
            st.markdown(f"**🏢 분류**\n{info['category']}")
            st.write("---")
            st.link_button("🗺️ 네이버 지도에서 보기", f"https://map.naver.com/v5/search/{cafe_name}")
    else:
        st.error("정보를 찾을 수 없습니다.")

# --- [SESSION STATE] ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'selections' not in st.session_state:
    st.session_state.selections = {"persona": None, "filters": [], "conveni": []}
if 'detail_cafe' not in st.session_state: st.session_state.detail_cafe = None

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
if st.session_state.step <= 3:
    if st.session_state.step == 1:
        st.subheader("Q1. 오늘 어떤 시간을 보내고 싶나요?")
        
        options = ["몰입과 영감 (조용, 사색)", "장인의 맛 (빵, 시그니처)", "체험과 배움", "비주얼/SNS"]
        icons = ["🧘", "🍞", "🎨", "📸"]
        for i, opt in enumerate(options):
            is_sel = st.session_state.selections['persona'] == opt
            # 선택 여부에 따라 primary/secondary 타입 변경 (테두리 색 결정)
            if st.button(f"{icons[i]} {opt}", key=f"p_{i}", type="primary" if is_sel else "secondary"):
                st.session_state.selections['persona'] = None if is_sel else opt
                st.rerun()
            
    # --- [STEP 2: FILTER (다중 선택 + Toggle)] ---
    elif st.session_state.step == 2:
        st.subheader("Q2. 추가로 고려해야 할 상황이 있나요?(중복선택가능)")
        options = {"반려동물 동반": "🐶", "아이와 함께": "👶", "단체 모임": "👥", "가성비 중요": "💰"}
        for i, (opt, icon) in enumerate(options.items()):
            is_sel = opt in st.session_state.selections['filters']
            if st.button(f"{icon} {opt}", key=f"f_{i}", type="primary" if is_sel else "secondary"):
                if is_sel: st.session_state.selections['filters'].remove(opt)
                else: st.session_state.selections['filters'].append(opt)
                st.rerun()
            
    # --- [STEP 3: CONVENIENCE (다중 선택 + Toggle)] ---
    elif st.session_state.step == 3:
        st.subheader("Q3. 포기할 수 없는 '편의 시설'은?")
        options = {"주차장 필수": "🚗", "깨끗한 화장실/서비스": "🚻", "야외 테라스/개방감": "🌿"}
        for i, (opt, icon) in enumerate(options.items()):
            is_sel = opt in st.session_state.selections['conveni']
            if st.button(f"{icon} {opt}", key=f"c_{i}", type="primary" if is_sel else "secondary"):
                if is_sel: st.session_state.selections['conveni'].remove(opt)
                else: st.session_state.selections['conveni'].append(opt)
                st.rerun()
            
    st.write("---")
    nav_cols = st.columns([1, 1, 1, 1, 1])
    with nav_cols[0]: # 이전 버튼 (왼쪽)
        if st.session_state.step > 1:
            if st.button("⬅️ 이전"):
                st.session_state.step -= 1
                st.rerun()
    with nav_cols[4]: # 다음/결과 버튼 (오른쪽 끝)
        if st.session_state.step < 3:
            if st.button("다음 ➔"):
                if st.session_state.selections['persona'] or st.session_state.step > 1:
                    st.session_state.step += 1
                    st.rerun()
        else:
            if st.button("✅ 결과 분석"):
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
    
    # 화면 분할
    col_list, col_detail = st.columns([1, 1.2])

    with col_list:
        st.header("🎯 맞춤 카페 추천")
        for i, (idx, row) in enumerate(result_df.iterrows()):
            # row['열1']을 실제 카페명 컬럼명(예: 'cafe_name')으로 바꿔주세요.
            cafe_name = row['열1'] 
            if st.button(f"{i+1}위: {cafe_name}", key=f"res_{i}"):
                st.session_state.detail_cafe = cafe_name

    with col_detail:
        st.header("🔍 상세 정보")
        if st.session_state.detail_cafe:
            info = get_naver_info(st.session_state.detail_cafe)
            img_url = get_naver_image(st.session_state.detail_cafe)
            if info:
                clean_title = info['title'].replace('<b>', '').replace('</b>', '')
                with st.container():
                    # 이미지 표시
                    if img_url:
                        st.image(img_url, use_container_width=True, caption=f"{clean_title} 현장 이미지")
                    
                    st.markdown(f"""
                    <div class="detail-card">
                        <h2>{clean_title}</h2>
                        <hr>
                        <p><b>🏢 분류:</b> {info['category']}</p>
                        <p><b>📍 위치:</b> {info['address']}</p>
                        <p><b>🛣️ 도로명:</b> {info['roadAddress']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    st.link_button("🗺️ 네이버 지도에서 길찾기", f"https://map.naver.com/v5/search/{st.session_state.detail_cafe}")
            else:
                st.warning("상세 정보를 찾을 수 없습니다.")
        else:
            st.info("왼쪽 리스트에서 카페를 클릭하면 네이버 정보를 보여드립니다.")

    if st.button("🔄 다시 테스트하기"):
        st.session_state.step = 1
        st.session_state.selections = {"persona": None, "filters": [], "conveni": []}
        st.session_state.detail_cafe = None
        st.rerun()