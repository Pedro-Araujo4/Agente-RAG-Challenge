import os
from dotenv import load_dotenv
import pypdf
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

client = genai.Client(api_key=api_key)

def carregar_e_limpar_pdf(caminho_pdf: str) -> list[Document]:
    loader = PyPDFLoader(caminho_pdf)
    documentos_paginas = loader.load()
    
    for doc in documentos_paginas:
        doc.page_content = doc.page_content.strip()
        
    return documentos_paginas

def chunking_e_metadados(documentos_paginas: list[Document]) -> list[Document]:
    """
    Executa exclusivamente o Chunking e a Atribuição de Metadados.
    """

    # CHUNKING 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,        
        chunk_overlap=100,      # Sobreposição para preservar o contexto nas bordas
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tenta quebrar por parágrafos primeiro
    )
    
    # Divide a lista de Documents das páginas em uma lista de Documents menores
    chunks_iniciais = text_splitter.split_documents(documentos_paginas)
    
    # ATRIBUIÇÃO DE METADADOS
    
    chunks_finais = []
    
    for i, chunk in enumerate(chunks_iniciais):
        # Captura os metadados já extraídos pelo PyPDFLoader
        fonte_original = os.path.basename(chunk.metadata.get("source", "FAQ.pdf"))
        pagina_num = chunk.metadata.get("page", 0) + 1  # Converte de base 0 para base 1
        
        # Cria/atualiza o dicionário de metadados
        metadados_enriquecidos = {
            "chunk_id": f"chunk_{i+1}",
            "categoria": "Financeiro / Meios de Pagamento",
            "fonte": fonte_original,
            "pagina_origem": pagina_num,
            "total_caracteres": len(chunk.page_content)
        }
        
        # Aplica os novos metadados mantendo a estrutura do LangChain
        chunk.metadata.update(metadados_enriquecidos)
        chunks_finais.append(chunk)
        
    return chunks_finais

def indexar_chunks_no_vectorstore(chunks: list[Document], pasta_db: str = "./chroma_db") -> Chroma:
    """
    Recebe os chunks com metadados, envia para o modelo de embedding e
    armazena o vetor, o índice HNSW e os metadados no ChromaDB.
    """
    # Configura o modelo oficial de Embeddings da Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )
    
    print(f"--> Gerando embeddings e indexando {len(chunks)} chunks no ChromaDB...")
    
    # Transforma o texto em vetores, constrói o índice HNSW e grava em disco
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=pasta_db
    )
    
    print(" Indexação concluída e salva no diretório local!")
    return vectorstore

if __name__ == "__main__":
    pdf_path = "./documentos/FAQ - Métodos de Pagamento.pdf"
    
    docs_paginas = carregar_e_limpar_pdf(pdf_path)
    
    # Executa Chunking e a Atribuição dos Metadados
    chunks_prontos = chunking_e_metadados(docs_paginas)
    # Indexação Vetorial (Envio ao embedding + Gravando no VectorDB)
    vectorstore = indexar_chunks_no_vectorstore(chunks_prontos)
