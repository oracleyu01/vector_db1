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


# 페이지 설정
st.set_page_config(page_title="부동산 데이터 분석 챗봇", page_icon="🏠")

# 페이지 제목
st.title("🏠 부동산 데이터 분석 챗봇")
st.write("네이버 블로그에서 수집한 부동산 데이터에 대해 질문해보세요.")

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
        collection = client.get_collection(name="property_data", embedding_function=embedding_function)
    except:
        collection = client.create_collection(name="property_data", embedding_function=embedding_function)
        
        # 가상의 부동산 데이터 (실제로는 크롤링 등으로 수집)
        property_data = [
            "강남 아파트 가격이 3개월 연속 하락세를 보이고 있습니다. 많은 블로거들이 금리 인상의 영향이라고 분석하고 있습니다.",
            "경기도 지역 아파트는 전월 대비 2.5% 하락했으며, 매수자들의 관망세가 계속되고 있습니다.",
            "30대 블로거들은 대출 규제와 금리 인상으로 내집 마련이 더 어려워졌다고 호소하고 있습니다.",
            "부동산 전문가들은 현재 시장 상황을 '조정기'로 보고 있으며, 1-2년간 조정이 계속될 것으로 전망합니다.",
            "40-50대 블로거들은 투자용 부동산의 가치 하락과 임대 수익률 감소에 대한 우려를 표하고 있습니다."
        ]
        
        # 데이터 추가
        for i, text in enumerate(property_data):
            collection.add(
                documents=[text],
                metadatas=[{"source": f"blog_{i}"}],
                ids=[f"doc_{i}"]
            )
    
    return collection

# 벡터 데이터베이스 컬렉션 가져오기
collection = get_chroma_client()

# 벡터 유사도 검색 함수
def search_property_data(query, n_results=3):
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
        return "죄송합니다, 질문에 관련된 데이터를 찾을 수 없습니다."
    
    # 임베딩 모델을 사용한 간단한 응답 생성 (실제로는 더 정교한 방법 사용 가능)
    # 여기서는 가장 유사한 문서를 기반으로 응답 생성
    response = f"블로그 데이터 분석 결과: {' '.join(context)}"
    
    # 질문 키워드에 따라 응답 맞춤화
    if "가격" in query.lower():
        response += "\n\n가격 동향을 살펴보면, 전반적으로 하락세를 보이고 있습니다."
    elif "전망" in query.lower() or "앞으로" in query.lower():
        response += "\n\n향후 시장 전망은 1-2년간의 조정기가 예상됩니다."
    elif "30대" in query.lower() or "젊은" in query.lower():
        response += "\n\n특히 30대는 대출 규제와 금리 인상에 민감하게 반응하고 있습니다."
    
    return response

# 챗봇 응답 생성 함수
def chat_response(question):
    # 관련 데이터 검색
    relevant_data = search_property_data(question)
    
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
if prompt := st.chat_input("질문을 입력하세요 (예: 최근 아파트 가격 변화 추세는 어떤가요?)"):
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
    "최근 아파트 가격 변화에 대한 사람들의 생각이 어떤가요?",
    "30대들은 부동산 시장에 대해 어떻게 생각하나요?",
    "부동산 시장 앞으로 어떻게 될까요?",
    "경기도 지역 아파트 가격은 어떻게 변했나요?"
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
    with st.expander("새 데이터 추가"):
        new_data = st.text_area("새로운 부동산 데이터를 입력하세요")
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
                st.success("데이터가 추가되었습니다!")
            else:
                st.error("데이터를 입력해주세요.")
    
    # 대화 기록 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.chat_history = []
        st.rerun()

    # 데이터 확인 섹션
    with st.expander("부동산 데이터 확인"):
        # 모든 데이터 가져오기
        all_data = collection.get()
        if all_data and 'documents' in all_data:
            st.write("현재 분석에 사용 중인 데이터:")
            for idx, data in enumerate(all_data['documents']):
                st.write(f"{idx+1}. {data}")
        else:
            st.write("데이터가 없습니다.")
