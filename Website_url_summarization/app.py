import validators
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="Text Summarizer | Website",
    page_icon="🦜"
)

st.title("🦜 Text Summarizer")
st.caption("Summarize content from Websites videos using Groq + LangChain")

# ---------------- Sidebar ----------------
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

# ---------------- Input ----------------
url = st.text_input(
    "Enter Website or YouTube URL",
    placeholder="https://example.com",
    label_visibility="collapsed"
)

# ---------------- Button ----------------
if st.button("🚀 Summarize"):

    # ---------- Validation ----------
    if not groq_api_key.strip():
        st.error("❌ Please enter your Groq API Key")
        st.stop()

    if not url.strip():
        st.error("❌ Please enter a URL")
        st.stop()

    if not validators.url(url):
        st.error("❌ Please enter a valid URL")
        st.stop()

    try:
        with st.spinner("Fetching content and generating summary..."):

            # ---------- LLM ----------
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key
            )

            prompt = ChatPromptTemplate.from_template(
                """
                Provide a clear, well-structured summary of the following content
                in approximately 300 words:

                {text}
                """
            )

            chain = prompt | llm | StrOutputParser()

            # ---------- LOAD CONTENT ----------
            full_text = ""

            # ---- YOUTUBE ----
            if "youtube.com" in url or "youtu.be" in url:
                try:
                    video_id = url.split("v=")[-1].split("&")[0]

                    transcript = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=["en"]
                    )

                    full_text = " ".join(
                        item["text"] for item in transcript
                    )

                except (TranscriptsDisabled, NoTranscriptFound):
                    st.error("❌ This YouTube video has no captions available")
                    st.stop()

                except Exception:
                    st.error("❌ Failed to fetch YouTube transcript")
                    st.stop()

            # ---- WEBSITE ----
            else:
                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                docs = loader.load()

                if not docs:
                    st.error("❌ No readable content found on the website")
                    st.stop()

                full_text = "\n\n".join(doc.page_content for doc in docs)

            # ---------- SAFETY LIMIT ----------
            full_text = full_text[:6000]

            # ---------- SUMMARIZE ----------
            summary = chain.invoke({"text": full_text})

            st.success(summary)

    except Exception as e:
        st.exception(e)
