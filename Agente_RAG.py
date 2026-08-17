import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

st.set_page_config(page_title="Assistente de IA Corporativo", page_icon="🤖", layout="centered")

@st.cache_resource
def carregar_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

reranker_model = carregar_reranker()

def carregar_e_preparar_chunks(caminho_pdf: str) -> list[Document]:
    """Carrega o PDF, limpa o texto e divide em chunks com metadados enriquecidos."""
    loader = PyPDFLoader(caminho_pdf)
    paginas = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(paginas)
    
    for i, chunk in enumerate(chunks):
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata.update({
            "chunk_id": f"chunk_{i+1}",
            "categoria": "Financeiro / Meios de Pagamento",
            "fonte": os.path.basename(chunk.metadata.get("source", "FAQ.pdf")),
            "pagina_origem": chunk.metadata.get("page", 0) + 1,
            "total_caracteres": len(chunk.page_content)
        })
        
    return chunks

@st.cache_resource
def obter_ou_criar_vectorstore(caminho_pdf: str, pasta_db: str = "./chroma_db") -> Chroma:
    """Gera embeddings e persiste os vetores no ChromaDB se não existirem."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=api_key
    )
    
    if os.path.exists(pasta_db) and os.listdir(pasta_db):
        return Chroma(persist_directory=pasta_db, embedding_function=embeddings)
        
    chunks = carregar_e_preparar_chunks(caminho_pdf)
    return Chroma.from_documents(chunks, embeddings, persist_directory=pasta_db)

def buscar_e_reranquear(
    vectorstore: Chroma, 
    query: str, 
    top_k: int = 8, 
    top_n: int = 3, 
    filtro: dict = None
) -> list[Document]:
    """Busca vetorial com filtro de metadados seguida de reranqueamento Cross-Encoder."""
    candidatos = vectorstore.similarity_search(query, k=top_k, filter=filtro)
    if not candidatos:
        return []

    scores = reranker_model.predict([[query, doc.page_content] for doc in candidatos])
    ranqueados = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in ranqueados[:top_n]]

def montar_contexto_rag(documentos: list[Document]) -> str:
    """Formata os trechos com os metadados de origem para citação clara."""
    return "\n".join(
        f"--- TRECHO {i} [Fonte: {d.metadata.get('fonte')} | Pág: {d.metadata.get('pagina_origem')} | ID: {d.metadata.get('chunk_id')}] ---\n{d.page_content}\n"
        for i, d in enumerate(documentos, 1)
    )

def gerar_resposta_rag(query: str, contexto: str, documentos: list[Document]) -> str:
    """Prompt anti-alucinação, fallback dinâmico e resposta com fontes."""
    if not contexto or not documentos:
        return (
            "Não encontrei essa informação nos documentos disponíveis da nossa base de conhecimento.\n\n"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.0
    )

    prompt = f"""Você é um assistente virtual corporativo especialista nos documentos da empresa.

Sua tarefa é responder à pergunta do colaborador usando EXCLUSIVAMENTE as informações fornecidas no CONTEXTO abaixo.

REGRAS OBRIGATÓRIAS:
1. Responda apenas com fatos extraídos diretamente do contexto. Não invente ou presuma informações externas.
2. Se o contexto não contiver a resposta exata, diga expressamente: "Não encontrei essa informação nos documentos disponíveis."
3. Sempre cite a fonte (nome do arquivo e número da página) ao final ou ao longo da sua explicação.

CONTEXTO RECUPERADO:
{contexto}

PERGUNTA DO COLABORADOR:
{query}

RESPOSTA FORMATADA:"""

    resposta = llm.invoke(prompt)

    # Resposta completa
    conteudo = resposta.content

    # Se o conteúdo for (dict/objeto)
    if isinstance(conteudo, list):
        textos = []
        for bloco in conteudo:
            if isinstance(bloco, dict) and "text" in bloco:
                textos.append(bloco["text"])
            elif isinstance(bloco, str):
                textos.append(bloco)
            elif hasattr(bloco, "text"):
                textos.append(bloco.text)
        return "\n".join(textos)

    return str(conteudo)


# INTERFACE WEB DO STREAMLIT

st.title("🤖 Assistente Virtual Corporativo")
st.caption("Agente de IA baseado na documentação oficial - Respostas com fontes rastreáveis")

pdf_path = "./documentos/FAQ - Métodos de Pagamento.pdf"
vectorstore = obter_ou_criar_vectorstore(pdf_path)
filtro_padrao = {"categoria": "Financeiro / Meios de Pagamento"}

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "fontes" in msg and msg["fontes"]:
            with st.expander("Documentos Consultados 📚 "):
                for fonte in msg["fontes"]:
                    st.write(f"- {fonte}")

if prompt_usuario := st.chat_input("Digite sua pergunta sobre procedimentos ou métodos de pagamento..."):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de dados..."):
            resultados = buscar_e_reranquear(vectorstore, prompt_usuario, top_k=8, top_n=3, filtro=filtro_padrao)
            contexto = montar_contexto_rag(resultados)
            resposta = gerar_resposta_rag(prompt_usuario, contexto, resultados)

            fontes_unicas = list({
                f"📄 {d.metadata.get('fonte')} (Página {d.metadata.get('pagina_origem')})"
                for d in resultados
            })

            st.write(resposta)
            if fontes_unicas:
                with st.expander("📚 Documentos Consultados"):
                    for f in fontes_unicas:
                        st.write(f"- {f}")

            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                st.button("👍", key=f"like_{len(st.session_state.messages)}")
            with col2:
                st.button("👎", key=f"dislike_{len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "assistant", "content": resposta, "fontes": fontes_unicas})
