# 🤖 Assistente Virtual Corporativo (RAG + Gemini)

Assistente virtual baseado em Inteligência Artificial para consultas à base de conhecimento corporativa. O sistema utiliza a técnica de **Retrieval-Augmented Generation (RAG)** combinada com reranqueamento por **Cross-Encoder** para entregar respostas precisas, fundamentadas e com citação direta de fontes (páginas e documentos originários).

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Framework Web:** Streamlit
* **Orquestração de RAG:** LangChain (`langchain-core`, `langchain-community`)
* **Modelo de Linguagem (LLM):** Google Gemini (`gemini-3.1-flash-lite`) via `langchain-google-genai`
* **Embeddings:** `gemini-embedding-001`
* **Banco Vetorial:** ChromaDB (Persistência Local)
* **Re-ranking:** `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`)
* **Processamento de Documentos:** `PyPDFLoader` & `RecursiveCharacterTextSplitter`

---

## 📂 Estrutura do Projeto

```text
.
├── documentos/
│   └── FAQ - Métodos de Pagamento.pdf   # Base de conhecimento em PDF
├── chroma_db/                            # Diretório local do banco vetorial (gerado automaticamente)
├── .env                                  # Variáveis de ambiente (Chave da API)
├── Agente_RAG.py                                # Código principal da aplicação
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Documentação do projeto

## ⚙️ Pré-requisitos

* **Python 3.10** ou superior instalado.
* Uma chave de API do **Google Gemini** (`GEMINI_API_KEY`).

---
```

## 🚀 Como Executar Localmente

### 1. Clonar o Repositório
```bash
git clone [https://github.com/Pedro-Araujo4/Agente-RAG-Challenge.git](https://github.com/Pedro-Araujo4/Agente-RAG-Challenge.git)
cd Agente-RAG-Challenge
```
### 2. Criar e Ativar o Ambiente Virtual (`venv`)

- **Linux / macOS / WSL:**
```bash
  python3 -m venv .venv
  source .venv/bin/activate
```

### PowerShell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Bash 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente
#### Crie um arquivo chamado .env na raiz do projeto com o seguinte conteúdo:
```bash
GEMINI_API_KEY=sua_chave_api_aqui
```
### 5. Adicionar os Documentos
#### Certifique-se de posicionar o arquivo PDF no caminho configurado no script:
```bash
./documentos/FAQ - Métodos de Pagamento.pdf
```

### 6. Iniciar a Aplicação
#### Execute o comando abaixo no terminal:
```bash
python -m streamlit run app.py
```
## 🌐 Aplicação em Produção (OCI)

### Acesse a aplicação rodando ao vivo na Oracle Cloud Infrastructure:
### 👉 **[http://163.176.222.199:8501/](http://163.176.222.199:8501/)**
