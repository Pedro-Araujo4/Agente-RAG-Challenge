import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


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

def criar_vectorstore(chunks: list[Document], pasta_db: str = "./chroma_db") -> Chroma:
    """Gera embeddings e persiste os vetores no ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=api_key
    )
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

    # Avalia os pares (query, conteúdo) e ordena pelo score
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
    """
    Prompt anti-alucinação, fallback dinâmico e resposta com fontes.
    """
    # Fallback imediato se a busca não tiver retornado nenhum documento confiável
    if not contexto or not documentos:
        return (
            "Não encontrei essa informação nos documentos disponíveis da nossa base de conhecimento.\n\n"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.0  # máxima fidelidade ao contexto, o que evita alucinações
    )

    # Responder apenas com o contexto
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

    return resposta.content

if __name__ == "__main__":
    pdf_path = "./documentos/FAQ - Métodos de Pagamento.pdf"
    
    # Coleta/Processamento do conteúdo e indexação vetorial
    chunks = carregar_e_preparar_chunks(pdf_path)
    vectorstore = criar_vectorstore(chunks)
    
    # Exemplo de Pergunta
    pergunta = "O que você pode me falar sobre o pagameplo por Pix?"
    filtro = {"categoria": "Financeiro / Meios de Pagamento"}
    
    # Retrieval + Reranking + Montagem de Contexto + Geração de Resposta (Pipeline RAG)
    resultados = buscar_e_reranquear(vectorstore, pergunta, top_k=8, top_n=3, filtro=filtro)
    contexto = montar_contexto_rag(resultados)
    resposta_final = gerar_resposta_rag(pergunta, contexto, resultados)
    
    print("\n================ RESPOSTA FINAL DO AGENTE ================\n")
    print(resposta_final)
