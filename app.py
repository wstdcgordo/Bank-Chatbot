import os
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the API key
API_KEY_GM = os.getenv("API_KEY_GM")

# Configure Google Gemini API
genai.configure(api_key=API_KEY_GM) # type: ignore

def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text() # type: ignore
    return text

def ask_gemini_question(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash") # type: ignore
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📄 Chat with Your PDF")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("PDF text extracted!")

    # Input question
    question = st.text_input("Ask a question about the PDF")

    if question:
        # Combine question with PDF content
        prompt = f"You are a helpful assistant, and you will specifically cater bank transaction and related inquiries. Here's a document:\n\n{pdf_text}\n\nQuestion: {question}"
        
        with st.spinner("Thinking..."):
            answer = ask_gemini_question(prompt)

        st.markdown("### 🤖 Answer")
        st.write(answer)