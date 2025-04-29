# 코드 최상단에 추가
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import random
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime  # datetime import 추가

# 페이지 설정
st.set_page_config(page_title="광진구 착한가게 소개 챗봇", page_icon="🏪")

# 페이지 제목
st.title("🏪 광진구 착한가게 소개 챗봇")
st.write("광진구의 다양한 착한가게에 대한 정보를 물어보세요.")

# 임베딩 모델 설정 (세션 상태에 저장하여 재로딩 방지)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 지원 모델 사용

embedding_model = load_embedding_model()

# Chroma DB 클라이언트 설정
@st.cache_resource
def get_chroma_client():
    # 메모리에 저장하는 클라이언트 생성
    client = chromadb.Client()
    
    # 사용자 정의 임베딩 함수 설정
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 컬렉션 생성 또는 가져오기
    try:
        collection = client.get_collection(name="gwangjin_shops", embedding_function=embedding_function)
    except:
        collection = client.create_collection(name="gwangjin_shops", embedding_function=embedding_function)
        
        # 광진구 착한가게 데이터 (실제로는 공공데이터 등으로 수집)
        shops_data = [
            "착한식당 '맛있는 한끼'는 중곡동에 위치한 한식당으로, 지역 농산물을 활용한 건강한 식단을 합리적인 가격에 제공합니다. 특히 어르신들에게 10% 할인 혜택을 제공합니다.",
            "광진구 구의동의 '친환경 마트'는 지역 생산 제품과 친환경 제품을 판매하며, 매달 수익의 5%를 지역 취약계층에 기부하고 있습니다.",
            "화양동 '따뜻한 빵집'은 매일 신선한 빵을 구워 판매하며, 폐업시간에 남은 빵을 지역 아동센터에 기부하는 활동을 하고 있습니다.",
            "건대입구역 근처의 '착한 문구점'은 학생들에게 10% 할인을 제공하며, 학기 초에는 저소득층 학생들에게 무료로 학용품을 지원합니다.",
            "자양동의 '마을 세탁소'는 독거노인과 장애인 가정의 세탁물을 무료로 수거하여 세탁 서비스를 제공하는 착한가게입니다."
        ]
        
        # 데이터 추가
        for i, text in enumerate(shops_data):
            collection.add(
                documents=[text],
                metadatas=[{"source": f"shop_{i}"}],
                ids=[f"doc_{i}"]
            )
    
    return collection

# 벡터 데이터베이스 컬렉션 가져오기
collection = get_chroma_client()

# 벡터 유사도 검색 함수
def search_shops_data(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results['documents'] and results['documents'][0]:
        return results['documents'][0]
    else:
        return ["관련 데이터를 찾을 수 없습니다."]

# 응답 생성 함수
def generate_response(query, context):
    if not context or context[0] == "관련 데이터를 찾을 수 없습니다.":
        return "죄송합니다, 질문에 관련된 정보를 찾을 수 없습니다."
    
    # 임베딩 모델을 사용한 간단한 응답 생성
    response = f"광진구 착한가게 정보: {' '.join(context)}"
    
    # 질문 키워드에 따라 응답 맞춤화
    if "할인" in query.lower() or "혜택" in query.lower():
        response += "\n\n광진구 착한가게들은 다양한 할인 혜택을 제공하고 있으며, 특히 노인과 학생들을 위한 할인 정책이 많습니다."
    elif "위치" in query.lower() or "어디" in query.lower():
        response += "\n\n광진구 착한가게는 중곡동, 구의동, 화양동, 건대입구, 자양동 등 광진구 전역에 분포되어 있습니다."
    elif "기부" in query.lower() or "봉사" in query.lower():
        response += "\n\n많은 착한가게들이 수익의 일부를 기부하거나 지역 사회를 위한 봉사 활동에 참여하고 있습니다."
    
    return response

# 챗봇 응답 생성 함수
def chat_response(question):
    # 관련 데이터 검색
    relevant_data = search_shops_data(question)
    
    # 응답 생성
    return generate_response(question, relevant_data)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요 (예: 광진구에 어떤 착한가게가 있나요?)"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 사용자 메시지 저장
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 응답 생성
    with st.spinner("답변 생성 중..."):
        response = chat_response(prompt)

    # 응답 메시지 표시
    with st.chat_message("assistant"):
        st.markdown(response)

    # 응답 메시지 저장
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# 예시 질문
st.sidebar.header("예시 질문")
example_questions = [
    "광진구에 어떤 착한가게들이 있나요?",
    "착한가게들이 제공하는 할인 혜택은 무엇인가요?",
    "착한가게들은 어떤 사회공헌 활동을 하고 있나요?",
    "중곡동 근처에 있는 착한가게를 알려주세요."
]

for question in example_questions:
    if st.sidebar.button(question):
        # 사용자 메시지 표시 및 저장
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 응답 생성
        with st.spinner("답변 생성 중..."):
            response = chat_response(question)

        # 응답 메시지 표시 및 저장
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # 페이지 새로고침
        st.rerun()

# 사이드바 설정
with st.sidebar:
    st.header("데이터 관리")
    
    # 데이터 추가 섹션
    with st.expander("새 착한가게 데이터 추가"):
        new_data = st.text_area("새로운 착한가게 정보를 입력하세요")
        if st.button("데이터 추가"):
            if new_data:
                # 새 ID 생성
                new_id = f"doc_{int(random.random() * 10000)}"
                
                # 데이터 추가
                collection.add(
                    documents=[new_data],
                    metadatas=[{"source": "user_input", "date_added": str(datetime.now())}],
                    ids=[new_id]
                )
                st.success("착한가게 정보가 추가되었습니다!")
            else:
                st.error("정보를 입력해주세요.")
    
    # 대화 기록 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.chat_history = []
        st.rerun()

    # 데이터 확인 섹션
    with st.expander("착한가게 데이터 확인"):
        # 모든 데이터 가져오기
        all_data = collection.get()
        if all_data and 'documents' in all_data:
            st.write("현재 분석에 사용 중인 착한가게 정보:")
            for idx, data in enumerate(all_data['documents']):
                st.write(f"{idx+1}. {data}")
        else:
            st.write("데이터가 없습니다.")
