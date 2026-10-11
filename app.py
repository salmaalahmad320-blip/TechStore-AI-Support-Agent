import streamlit as st

from src.chatbot import generate_answer
from src.rag import RAGService
from src.rag_errors import ChatbotError, KnowledgeBaseError, VectorStoreError
from src.retriever import select_source_documents


st.set_page_config(
    page_title="TechStore AI Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }
    .hero-eyebrow {
        color: #4b78d1;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero-description {
        color: #738096;
        font-size: 1.05rem;
        max-width: 760px;
        line-height: 1.65;
    }
    .chat-empty-state {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1.5rem 0;
        background: rgba(128, 128, 128, 0.035);
    }
    .chat-empty-state p {
        color: #738096;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


st.markdown('<div class="hero-eyebrow">TechStore · Customer Care</div>', unsafe_allow_html=True)
st.title("AI Support Agent")
st.markdown(
    """
    <p class="hero-description">
    Get clear answers about TechStore products and policies. The assistant uses
    retrieval-augmented generation (RAG) to find relevant passages in your
    knowledge base, search them with FAISS, and draft grounded responses with an LLM.
    </p>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("knowledge_base_status", "Checking knowledge base")
st.session_state.setdefault("knowledge_base_error", "")
st.session_state.setdefault("upload_notice", None)

rag_service = get_rag_service()
try:
    indexed_chunks = rag_service.sync_knowledge_base()
    st.session_state["knowledge_base_status"] = "Knowledge base ready"
    st.session_state["knowledge_base_error"] = ""
    if indexed_chunks:
        st.session_state["upload_notice"] = (
            "success",
            f"Knowledge base updated with {indexed_chunks} new chunk(s).",
        )
except (KnowledgeBaseError, VectorStoreError) as error:
    st.session_state["knowledge_base_status"] = "Knowledge base needs attention"
    st.session_state["knowledge_base_error"] = str(error)

with st.sidebar:
    st.markdown("### Knowledge base")
    st.caption("Add company documents so answers can use your latest information.")

    with st.form("document_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="You can select multiple files. They will be saved and indexed.",
        )
        if uploaded_files:
            st.caption(
                f"{len(uploaded_files)} file(s) selected · "
                f"{sum(file.size for file in uploaded_files) / (1024 * 1024):.1f} MB"
            )
        upload_submitted = st.form_submit_button(
            "Add documents",
            type="primary",
            width="stretch",
        )

    if upload_submitted:
        if not uploaded_files:
            st.session_state["upload_notice"] = (
                "info",
                "Choose at least one PDF or TXT file to upload.",
            )
        else:
            try:
                saved_paths, indexed_chunks = rag_service.add_uploaded_files(
                    uploaded_files
                )
                st.session_state["knowledge_base_status"] = "Knowledge base ready"
                st.session_state["knowledge_base_error"] = ""
                if indexed_chunks:
                    st.session_state["upload_notice"] = (
                        "success",
                        f"Added {len(saved_paths)} file(s) and indexed "
                        f"{indexed_chunks} chunk(s).",
                    )
                else:
                    st.session_state["upload_notice"] = (
                        "success",
                        "Selected file(s) are saved and already indexed.",
                    )
            except (KnowledgeBaseError, VectorStoreError) as error:
                st.session_state["knowledge_base_status"] = (
                    "Knowledge base needs attention"
                )
                st.session_state["knowledge_base_error"] = str(error)
                st.session_state["upload_notice"] = ("error", str(error))

    notice = st.session_state.get("upload_notice")
    if notice:
        level, message = notice
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)
        st.session_state["upload_notice"] = None

    st.divider()
    st.markdown("### System status")
    if st.session_state["knowledge_base_error"]:
        st.error(st.session_state["knowledge_base_status"])
        st.caption(st.session_state["knowledge_base_error"])
    else:
        st.success(st.session_state["knowledge_base_status"])

    st.divider()
    st.markdown("### Technology")
    st.caption("Streamlit · LangChain · OpenAI · FAISS · RAG")

    st.divider()
    st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        width="stretch",
        on_click=lambda: st.session_state.update({"messages": []}),
    )

st.markdown("### Conversation")
if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="chat-empty-state">
            <strong>How can we help?</strong>
            <p>Ask about products, shipping, returns, payments, or warranty.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Answers are grounded in indexed documents and include sources when available.")

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=":material/person:" if message["role"] == "user" else ":material/smart_toy:",
    ):
        st.markdown(message["content"])
        citations = message.get("sources", [])
        if citations:
            st.caption("Sources · " + " · ".join(citations))
        excerpts = message.get("excerpts", [])
        if excerpts:
            with st.expander("View retrieved source excerpts"):
                for citation, excerpt in excerpts:
                    st.markdown(f"**{citation}**")
                    st.write(excerpt)

question = st.chat_input(
    "Ask a question about TechStore…",
    max_chars=2000,
)
if question:
    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    try:
        documents = rag_service.retrieve(question)
        if documents:
            answer = generate_answer(
                documents,
                question,
                chat_history=st.session_state["messages"][:-1],
            )
        else:
            answer = (
                "I couldn't find information about that in the knowledge base. "
                "Please contact customer support if you'd like us to confirm."
            )

        cited_documents = select_source_documents(documents)
        citations = [citation for citation, _ in cited_documents]
        excerpts = [
            (citation, document.page_content)
            for citation, document in cited_documents
        ]
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": citations,
                "excerpts": excerpts,
            }
        )
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            st.markdown(answer)
            if citations:
                st.caption("Sources · " + " · ".join(citations))
            if excerpts:
                with st.expander("View retrieved source excerpts"):
                    for citation, excerpt in excerpts:
                        st.markdown(f"**{citation}**")
                        st.write(excerpt)
    except (ChatbotError, KnowledgeBaseError, VectorStoreError, ValueError) as error:
        st.session_state["messages"].pop()
        st.error(f"Couldn't complete that request. {error}")
