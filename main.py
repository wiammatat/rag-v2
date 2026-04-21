import os
import fitz
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importation de tes fonctions depuis logic.py
from logic import run_all_chunking, generate_vector_viz

# --- CONFIGURATION ---
NEO4J_URI = "neo4j+s://239167f9.databases.neo4j.io"
NEO4J_USER = "239167f9"
NEO4J_PWD = "r_NcAvrsZXWMWsl4_YFDcDedCZoRcc9n5aplw4iERHI"
CHEMIN_PDF = r"C:\Users\matat\Desktop\wiwi 222\Netoyage.pdf"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        app.state.driver.verify_connectivity()
        print("✅ Connecté à Neo4j Aura")
    except Exception as e:
        print(f"❌ Erreur Neo4j : {e}")
        app.state.driver = None
    yield
    if hasattr(app.state, 'driver') and app.state.driver:
        app.state.driver.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def extraire_texte_pdf(chemin):
    if not os.path.exists(chemin): return "Erreur : Fichier PDF introuvable."
    try:
        with fitz.open(chemin) as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e: return f"Erreur : {e}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, q: str = None):
    texte_complet = extraire_texte_pdf(CHEMIN_PDF)
    df_chunks = run_all_chunking(texte_complet)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_pour_viz = splitter.split_text(texte_complet[:8000]) 
    
    pca_img, top_docs, summary, search_piliers = generate_vector_viz(chunks_pour_viz, query=q)
    
    piliers_statiques = [
        {"methode": "Top-k Semantic", "score": "0.92", "calcul": "Cosine Similarity", "etat": "Prêt"},
        {"methode": "Keyword Search", "score": "0.45", "calcul": "BM25 Algorithm", "etat": "Prêt"},
        {"methode": "Hybrid Search", "score": "0.88", "calcul": "RRF Ranking", "etat": "Prêt"}
    ]

    # CORRECTION : On passe 'request' en premier argument ET dans le dictionnaire
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "metrics": df_chunks.to_dict(orient="records"),
            "search_piliers": search_piliers if q else piliers_statiques,
            "pca_image": pca_img,
            "relevant_docs": top_docs,
            "summary": summary,
            "query": q,
            "filename": "Netoyage.pdf"
        }
    )

@app.get("/graph", response_class=HTMLResponse)
async def graph_rag(request: Request):
    nodes, links = [], []
    if hasattr(request.app.state, 'driver') and request.app.state.driver:
        with request.app.state.driver.session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n.name as s, m.name as t, type(r) as rel LIMIT 50")
            for record in result:
                nodes.append({"id": record["s"], "label": record["s"]})
                nodes.append({"id": record["t"], "label": record["t"]})
                links.append({"source": record["s"], "target": record["t"], "label": record["rel"]})
    
    unique_nodes = {n["id"]: n for n in nodes}.values()
    
    # CORRECTION : Même chose ici pour la route graph
    return templates.TemplateResponse(
        request=request,
        name="graph.html",
        context={
            "nodes": list(unique_nodes), 
            "links": links, 
            "active_page": "graph"
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)