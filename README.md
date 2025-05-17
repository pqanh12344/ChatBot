# Chatbot RAG: A Retrieval-Augmented Generation Chatbot

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using Python, Streamlit, and state-of-the-art NLP models. The chatbot leverages a combination of document retrieval and text generation to provide accurate and contextually relevant answers to user queries, specifically tailored for Vietnamese language datasets like ViQuAD. It features a user-friendly web interface, efficient document processing, and robust error handling.

## Features
- **Document Retrieval**: Uses FAISS and Sentence Transformers for efficient similarity-based retrieval of relevant documents.
- **Text Generation**: Employs a lightweight `distilgpt2` model for generating concise and accurate answers.
- **Text Preprocessing**: Cleans and standardizes input data to ensure consistency.
- **Chunking**: Splits large documents into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
- **Streamlit Interface**: Provides an intuitive UI with chat history, source display, and custom styling.
- **Error Handling**: Includes comprehensive logging and exception management for reliability.
- **Caching**: Utilizes Streamlit's caching to optimize performance by storing precomputed embeddings and processed data.

## Tech Stack
- **Programming Language**: Python 3.8+
- **Web Framework**: Streamlit
- **NLP Libraries**:
  - Sentence Transformers (`halong_embedding`) for embeddings
  - Transformers (`distilgpt2`) for text generation
  - LangChain for text splitting
- **Vector Search**: FAISS for efficient similarity search
- **Other Libraries**: NumPy, Pandas, Torch, Re, Logging

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place your dataset files (`viquad.contexts`, `viquad.questions`, `viquad.answers`) in the `./data/` directory.
   - Ensure the files are in UTF-8 encoded text format with one entry per line.

5. **Download Models**:
   - The `halong_embedding` and `distilgpt2` models will be automatically downloaded from Hugging Face when the script runs for the first time.

### Requirements
Create a `requirements.txt` file with the following:
```text
streamlit
sentence-transformers
langchain
faiss-cpu
transformers
torch
numpy
pandas
tqdm
```

## Usage
1. **Run the Application**:
   ```bash
   streamlit run ui.py
   ```
   This will launch the Streamlit web interface in your default browser.

2. **Interact with the Chatbot**:
   - Enter a question in the text input field (e.g., "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?").
   - Click **Gửi câu hỏi** to get a response.
   - View the answer, source document, and chat history in the interface.

3. **Clear Chat History**:
   - Click the **Xóa lịch sử trò chuyện** button to reset the conversation.

## Project Structure
```plaintext
├── ui.py                  # Main application script
├── data/                   # Directory for dataset files
│   ├── viquad.contexts
│   ├── viquad.questions
│   ├── viquad.answers
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## How It Works
1. **Data Loading**: Reads context, question, and answer files from the `./data/` directory.
2. **Text Preprocessing**: Cleans text by removing extra whitespace and standardizing format.
3. **Document Chunking**: Splits contexts into smaller chunks for efficient processing.
4. **Embedding Creation**: Converts text chunks into vector embeddings using `halong_embedding`.
5. **Retrieval**: Uses FAISS to find the most relevant document chunks for a given query.
6. **Answer Generation**: Combines retrieved documents and example prompts to generate answers with `distilgpt2`.
7. **UI Rendering**: Displays questions, answers, sources, and chat history via Streamlit.

## Performance Optimizations
- **Caching**: Leverages Streamlit's `@st.cache_data` to avoid redundant computation.
- **Batch Processing**: Processes embeddings in smaller batches to reduce memory usage.
- **Lightweight Model**: Uses `distilgpt2` for faster inference on CPU.
- **Memory Management**: Includes garbage collection (`gc.collect()`) to free up memory.

## Limitations
- **Hardware**: Optimized for CPU, which may be slower than GPU-based setups.
- **Dataset Size**: Currently limited to processing the first 100 lines of input files for faster initialization.
- **Language**: Primarily designed for Vietnamese (ViQuAD dataset), but can be adapted for other languages.
- **Model Accuracy**: `distilgpt2` is lightweight but may not match the performance of larger models like GPT-3.

## Future Improvements
- Support for larger datasets with pagination or streaming.
- Integration with more powerful language models (e.g., LLaMA, GPT-4).
- Multi-language support by fine-tuning embeddings and generation models.
- GPU acceleration for faster embedding and inference.
- Enhanced UI with real-time typing indicators and richer formatting.

## Contact
For questions or feedback, please open an issue on the repository or contact the maintainers.

---
Built with ❤️ by Phan Quoc Anh
