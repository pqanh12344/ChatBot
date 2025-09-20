import streamlit as st
from data_loader import load_documents, chunk_documents
from embeddings import create_embeddings, load_model
from chatbot_qdrant import chatbot_rag, connect_qdrant, create_collection_if_not_exists, upload_documents_to_qdrant
from config import logger, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_SIZE

client = connect_qdrant(QDRANT_URL, QDRANT_API_KEY)
create_collection_if_not_exists(client, COLLECTION_NAME, VECTOR_SIZE)

def main():
    st.set_page_config(
        page_title="ChatBot", 
        page_icon="💬", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS để tạo giao diện giống ChatBot với nền trắng
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Main App Styling */
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #1f1f1f;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f7f7f8;
        padding: 1rem;
        border-right: 1px solid #e5e5e5;
    }
    
    .sidebar-content {
        background-color: #f7f7f8;
        color: #1f1f1f;
        padding: 0;
    }
    
    .sidebar-title {
        color: #666666;
        font-size: 14px;
        font-weight: 600;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        border-bottom: 1px solid #e5e5e5;
        background-color: #ffffff;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .model-selector {
        background-color: #ffffff;
        color: #1f1f1f;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .plus-badge {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Welcome Title */
    .welcome-title {
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin: 3rem 0;
    }
    
    /* Chat Messages */
    .chat-message {
        max-width: 768px;
        margin: 0 auto 2rem auto;
        padding: 1.5rem;
    }
    
    .user-message {
        background-color: #f8f9fa;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        margin-left: 20%;
        padding: 1rem 1.5rem;
    }
    
    .bot-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Input Container */
    .input-section {
        max-width: 768px;
        margin: 2rem auto;
        padding: 0 2rem;
        position: relative;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 24px;
        color: #1f1f1f;
        padding: 16px 24px;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16,163,127,0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999999;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #10a37f;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(16,163,127,0.2);
    }
    
    .stButton > button:hover {
        background-color: #0d8f6f;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16,163,127,0.3);
    }
    
    /* Form Submit Button */
    .stFormSubmitButton > button {
        background-color: #10a37f;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stFormSubmitButton > button:hover {
        background-color: #0d8f6f;
    }
    
    /* Expander Styling */
    .stExpander {
        background-color: #f8f9fa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .stExpander > div > div > div > div {
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    </style>
    """, unsafe_allow_html=True)

    # Khởi tạo dữ liệu và mô hình nếu chưa có
    if 'initialized' not in st.session_state:
        with st.spinner("Đang tải dữ liệu và mô hình..."):
            try:
                contexts, questions, answers = load_documents()
                st.session_state.chunks, st.session_state.metadata = chunk_documents(contexts, questions, answers)
                st.session_state.embeddings = create_embeddings(st.session_state.chunks)
                upload_documents_to_qdrant(client, COLLECTION_NAME, st.session_state.chunks, st.session_state.embeddings)
                st.session_state.model = load_model()
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Khởi tạo thất bại: {e}")
                logger.error(f"Initialization error: {e}")
                return

    # Khởi tạo lịch sử trò chuyện
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:        
        # New chat button
        if st.button("📝 Đoạn chat mới", key="new_chat", help="Bắt đầu cuộc trò chuyện mới", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Đã tạo cuộc trò chuyện mới!")
        
        st.markdown('<div class="sidebar-title">🔍 Tìm kiếm đoạn chat</div>', unsafe_allow_html=True)
        search_query = st.text_input("Search", placeholder="Tìm kiếm...", key="search_input")
        
        # Chat history in sidebar
        if st.session_state.chat_history:
            st.markdown('<div class="sidebar-title">💬 Đoạn chat</div>', unsafe_allow_html=True)
            
            # Hiển thị lịch sử chat ngắn gọn trong sidebar
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                truncated_question = chat['question'][:40] + "..." if len(chat['question']) > 40 else chat['question']
                if st.button(f"💬 {truncated_question}", key=f"sidebar_chat_{i}", help=chat['question']):
                    st.info(f"Đã chọn: {chat['question']}")

    # Main content area
    # Header
    # Welcome title hoặc chat history
    if not st.session_state.chat_history:
        st.markdown('<h1 class="welcome-title">What do you want to do today?</h1>', unsafe_allow_html=True)
    else:
        # Hiển thị lịch sử chat
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f'''
            <div class="chat-message">
                <div class="user-message">
                    <strong>🙋‍♂️ Bạn:</strong><br>
                    {chat["question"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f'''
            <div class="chat-message">
                <div class="bot-message">
                    <strong>🤖 ChatBot:</strong><br>
                    {chat["answer"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Nút xóa lịch sử trò chuyện
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("🗑️ Xóa lịch sử trò chuyện", key="clear_history"):
                st.session_state.chat_history = []
                st.success("Đã xóa lịch sử trò chuyện.")
                st.rerun()
    
    # Input area ở giữa màn hình
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Form nhập câu hỏi
    with st.form(key='question_form', clear_on_submit=True):
        query = st.text_input(
            "Chat Input", 
            placeholder="Hỏi bất kỳ điều gì...", 
            key="user_question",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            submit_button = st.form_submit_button("📤 Gửi câu hỏi", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Xử lý gửi câu hỏi
    if submit_button and query.strip():
        with st.spinner("🤔 Đang xử lý câu hỏi..."):
            try:
                answer, source = chatbot_rag(
                    client,
                    query,
                    st.session_state.model
                )
                st.session_state.chat_history.append({
                    "question": query, 
                    "answer": answer, 
                    "source": source
                })
                
                # Hiển thị câu trả lời mới nhất
                st.subheader("💡 Câu trả lời:")
                st.markdown(f"**{answer}**")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra: {e}")
                logger.error(f"Chat error: {e}")
    
    # Hiển thị lịch sử trò chuyện chi tiết ở cuối
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📋 Lịch sử trò chuyện chi tiết")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"❓ Câu hỏi {len(st.session_state.chat_history) - i}: {chat['question']}", expanded=False):
                st.markdown(f"**💬 Trả lời**: {chat['answer']}")
                if chat.get('source'):
                    st.caption(f"📄 Nguồn: {chat['source']}")

if __name__ == "__main__":
    main()