import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- [페이지 설정] ---
st.set_page_config(page_title="Cafe Finder Pro", layout="wide")

# API KEY
NAVER_CLIENT_ID = "h5Boba0NG1huDKOpvL6O"
NAVER_CLIENT_SECRET = "4vslkqoNEF"

# --- [CSS: 한 화면 고정 및 좌측 정렬 레이아웃] ---
st.markdown("""
    <style>
    /* 1. 전체 화면 스크롤 방지 및 여백 제거 */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden;
        max-height: 100vh;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        height: 100vh;
    }

    /* 2. 타이틀 좌측 정렬 및 간격 조절 */
    h1, h2, h3, [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        margin-bottom: 0.5rem !important;
    }

    /* 3. 버튼 스타일 (가로로 길고 세로는 적당하게) */
    div.stButton > button {
        width: 600px !important;
        height: 110px !important; /* 기존의 적당한 세로 높이 */
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        margin-bottom: 5px;
        transition: all 0.3s;
    }

    /* 4. 내비게이션 및 결과 버튼 (컴팩트하게) */
    [data-testid="column"] div.stButton > button {
        height: 55px !important;
        font-size: 18px !important;
    }

    /* 5. 팝업(Dialog) 애니메이션 및 노이즈 제거 */
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    div[data-testid="stDialog"] > div {
        animation: slideInRight 0.4s ease-out !important;
        border: none !important; /* 테두리 선 제거 */
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
        max-width: 900px !important;
    }
    
    /* 6. 상세페이지 이미지 규격화 */
    .popup-img {
        width: 100%;
        aspect-ratio: 16 / 9; /* 이미지 비율 고정 */
        object-fit: cover;
        border-radius: 12px;
    }

    /* 불필요한 간격 제거 */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [API 함수] ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('cafe_pca.csv')
    except:
        return pd.DataFrame({'열1': [f'카페_{i}' for i in range(1, 10)], **{f'PC{j}': np.random.rand(9) for j in range(1, 37)}})

def get_naver_info(query):
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": 1}
    res = requests.get(url, headers=headers, params=params)
    return res.json().get('items')[0] if res.status_code == 200 and res.json().get('items') else None

@st.cache_data(ttl=3600)
def get_naver_image(query):
    url = "https://openapi.naver.com/v1/search/image"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query + " 카페 내부", "display": 1}
    res = requests.get(url, headers=headers, params=params)
    return res.json().get('items')[0]['link'] if res.status_code == 200 and res.json().get('items') else None

# --- [상세 정보 다이얼로그] ---
@st.dialog("카페 상세 정보", width="large")
def show_cafe_detail(cafe_name):
    info = get_naver_info(cafe_name)
    img_url = get_naver_image(cafe_name)
    
    if info:
        title = info['title'].replace('<b>', '').replace('</b>', '')
        col_img, col_txt = st.columns([1.2, 1])
        with col_img:
            if img_url:
                # 스타일 적용을 위해 HTML로 이미지 출력
                st.markdown(f'<img src="{img_url}" class="popup-img">', unsafe_allow_html=True)
            else:
                st.info("📷 이미지가 없습니다.")
        with col_txt:
            st.markdown(f"## {title}")
            st.markdown(f"**🏷️ 카테고리**: {info['category']}")
            st.markdown(f"**📍 주소**: {info['address']}")
            st.write("---")
            st.link_button("🗺️ 네이버 지도 바로가기", f"https://map.naver.com/v5/search/{cafe_name}")
    else:
        st.error("데이터를 불러오지 못했습니다.")

# --- [메인 로직] ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'selections' not in st.session_state: st.session_state.selections = {"persona": None, "filters": [], "conveni": []}

# 상단 타이틀 부 (컴팩트하게)
st.title("☕ Cafe Finder")
st.progress(min(st.session_state.step / 3, 1.0))

# --- [설문 단계] ---
if st.session_state.step <= 3:
    if st.session_state.step == 1:
        st.subheader("Q1. 오늘 어떤 시간을 보내고 싶나요?")
        options = [("🧘 몰입과 영감", "몰입과 영감 (조용, 사색)"), ("🍞 장인의 맛", "장인의 맛 (빵, 시그니처)"), 
                   ("🎨 체험과 배움", "체험과 배움"), ("📸 비주얼/SNS", "비주얼/SNS")]
    elif st.session_state.step == 2:
        st.subheader("Q2. 누구와 함께 가시나요?")
        options = [("🐶 반려동물", "반려동물"), ("👶 아이와 함께", "아이와 함께"), 
                   ("👥 단체 모임", "단체 모임"), ("💰 가성비", "가성비")]
    elif st.session_state.step == 3:
        st.subheader("Q3. 꼭 필요한 편의 시설은?")
        options = [("🚗 넓은 주차장", "주차장"), ("🚻 깨끗한 화장실", "화장실/서비스"), ("🌿 야외 테라스", "테라스/개방감"), ("🔌 콘센트", "콘센트")]

    # 2x2 격자 버튼
    for i in range(0, len(options), 2):
        c1, c2 = st.columns(2)
        with c1:
            l, v = options[i]
            is_sel = (st.session_state.selections['persona'] == v) if st.session_state.step == 1 else (v in st.session_state.selections['filters'] or v in st.session_state.selections['conveni'])
            if st.button(l, key=f"btn_{i}", type="primary" if is_sel else "secondary"):
                if st.session_state.step == 1: st.session_state.selections['persona'] = v
                elif st.session_state.step == 2: st.session_state.selections['filters'].append(v) if v not in st.session_state.selections['filters'] else st.session_state.selections['filters'].remove(v)
                else: st.session_state.selections['conveni'].append(v) if v not in st.session_state.selections['conveni'] else st.session_state.selections['conveni'].remove(v)
                st.rerun()
        with c2:
            if i+1 < len(options):
                l, v = options[i+1]
                is_sel = (st.session_state.selections['persona'] == v) if st.session_state.step == 1 else (v in st.session_state.selections['filters'] or v in st.session_state.selections['conveni'])
                if st.button(l, key=f"btn_{i+1}", type="primary" if is_sel else "secondary"):
                    if st.session_state.step == 1: st.session_state.selections['persona'] = v
                    elif st.session_state.step == 2: st.session_state.selections['filters'].append(v) if v not in st.session_state.selections['filters'] else st.session_state.selections['filters'].remove(v)
                    else: st.session_state.selections['conveni'].append(v) if v not in st.session_state.selections['conveni'] else st.session_state.selections['conveni'].remove(v)
                    st.rerun()

    # 하단 내비게이션 (한 화면 구성을 위해 딱 붙임)
    st.write("---")
    nav_c = st.columns([1, 1])
    with nav_c[0]:
        if st.session_state.step > 1:
            if st.button("⬅️ 이전 단계"): st.session_state.step -= 1; st.rerun()
    with nav_c[1]:
        label = "✅ 결과 보기" if st.session_state.step == 3 else "다음 단계 ➔"
        if st.button(label): st.session_state.step += 1; st.rerun()

# --- [결과 단계] ---
elif st.session_state.step == 4:
    st.subheader("🎯 당신을 위한 카페 추천")
    test_cafes = ["어니언 안국", "블루보틀 삼청", "테라로사 포스코센터점", "앤트러사이트 한남", "프릳츠 도화점"]
    
    for i, name in enumerate(test_cafes):
        if st.button(f"🏆 {i+1}위 | {name}", key=f"res_{i}"):
            show_cafe_detail(name)
    
    if st.button("🔄 다시 하기"):
        st.session_state.step = 1; st.rerun()