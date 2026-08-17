import os
from dotenv import load_dotenv
import pypdf
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

client = genai.Client()

def extrair_texto_pdf(caminho_pdf: str) -> str:
    """Extrai todo o texto do arquivo PDF."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_pdf}")
    
    reader = pypdf.PdfReader(caminho_pdf)
    texto = ""
    for i, pagina in enumerate(reader.pages):
        conteudo = pagina.extract_text()
        if conteudo:
            conteudo_limpo = conteudo.strip()
            if conteudo_limpo:
                texto += f"\n--- Página {i+1} ---\n{conteudo}\n"
    return texto


if __name__ == "__main__":
    pdf_path = "./documentos/FAQ - Métodos de Pagamento.pdf"  
    
    texto_documento = extrair_texto_pdf(pdf_path)
    print("Conteúdo do PDF extraído.")