import streamlit as st
from data_loader import load_documents, chunk_documents
from embeddings import create_embeddings, load_model
from chatbot import chatbot_rag
from config import logger

def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
    
    # Thêm CSS tùy chỉnh
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stExpander {
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Chatbot")
    st.write("Hỏi bất kỳ câu hỏi nào!")

    # Khởi tạo dữ liệu và mô hình nếu chưa có
    if 'initialized' not in st.session_state:
        with st.spinner("Đang tải dữ liệu và mô hình..."):
            try:
                contexts, questions, answers = load_documents()
                st.session_state.chunks, st.session_state.metadata = chunk_documents(contexts, questions, answers)
                st.session_state.embeddings = create_embeddings(st.session_state.chunks)
                st.session_state.model = load_model()
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Khởi tạo thất bại: {e}")
                logger.error(f"Initialization error: {e}")
                return

    # Khởi tạo lịch sử trò chuyện
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Nút xóa lịch sử trò chuyện
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.chat_history = []
        st.success("Đã xóa lịch sử trò chuyện.")

    # Form nhập câu hỏi
    with st.form(key='question_form'):
        query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?")
        submit_button = st.form_submit_button(label="Gửi câu hỏi")

    if submit_button and query:
        with st.spinner("Đang xử lý câu hỏi..."):
            answer, source = chatbot_rag(
                query,
                st.session_state.chunks,
                st.session_state.embeddings,
                st.session_state.metadata,
                st.session_state.model
            )
            st.session_state.chat_history.append({"question": query, "answer": answer, "source": source})
            st.subheader("Câu trả lời:")
            st.markdown(f"**{answer}**")

    # Hiển thị lịch sử trò chuyện
    if st.session_state.chat_history:
        st.subheader("Lịch sử trò chuyện")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Câu hỏi {len(st.session_state.chat_history) - i}: {chat['question']}"):
                st.markdown(f"**Trả lời**: {chat['answer']}")
                # st.write(f"**Nguồn**: {chat['source']}")

if __name__ == "__main__":
    main()