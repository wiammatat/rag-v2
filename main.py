import os
import fitz
import uvicorn
import community as community_louvain  # pip install python-louvain
import networkx as nx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import RecursiveCharacterTextSplitter

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
        print(" Connecté à Neo4j Aura")
    except Exception as e:
        print(f" Erreur Neo4j : {e}")
        app.state.driver = None
    yield
    if hasattr(app.state, 'driver') and app.state.driver:
        app.state.driver.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static"): 
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def extraire_texte_pdf(chemin):
    if not os.path.exists(chemin): 
        return "Erreur : Fichier PDF introuvable."
    try:
        with fitz.open(chemin) as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e: 
        return f"Erreur : {e}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, q: str = None):
    texte_complet = extraire_texte_pdf(CHEMIN_PDF)
    df_chunks = run_all_chunking(texte_complet)
    
    # Utilisation d'un splitter pour la visualisation
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_pour_viz = splitter.split_text(texte_complet[:8000]) 
    
    # Appel de la logique vectorielle
    pca_img, top_docs, summary, search_piliers = generate_vector_viz(chunks_pour_viz, query=q)
    
    # Définition des piliers par défaut
    piliers_statiques = [
        {"methode": "Top-k Semantic", "score": "0.92", "calcul": "Cosine Similarity", "etat": "Optimisé"},
        {"methode": "Lexical Search", "score": "0.45", "calcul": "BM25 Algorithm", "etat": "Standard"},
        {"methode": "Hybrid Retrieval", "score": "0.88", "calcul": "RRF Ranking", "etat": "Stable"},
        {"methode": "Contextual Retrieval", "score": "0.75", "calcul": "LongContext", "etat": "Actif"},
        {"methode": "Max Marginal Rel.", "score": "0.82", "calcul": "MMR Diversity", "etat": "Diversifié"}
    ]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "search_piliers": search_piliers if q else piliers_statiques,
            "metrics": df_chunks.to_dict(orient="records"),
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
    metrics = {}
    
    if hasattr(request.app.state, 'driver') and request.app.state.driver:
        try:
            with request.app.state.driver.session() as session:
                # 1. Récupération des données depuis Neo4j
                result = session.run("MATCH (n)-[r]->(m) RETURN n.name as s, m.name as t, type(r) as rel, r.type as label_rel")
                
                # Construire un graphe NetworkX pour les calculs de metrics
                G_nx = nx.Graph()
                
                records = list(result)
                for record in records:
                    source = record["s"]
                    target = record["t"]
                    rel_label = record["label_rel"] or record["rel"] or "REL"
                    
                    G_nx.add_edge(source, target)
                    links.append({
                        "source": source, 
                        "target": target, 
                        "label": rel_label
                    })

                # 2. Calculs des algorithmes de graphe
                if len(G_nx.nodes) > 0:
                    # Algorithme de Louvain (Communautés)
                    partition = community_louvain.best_partition(G_nx)
                    modularity = community_louvain.modularity(partition, G_nx)
                    
                    # Métriques de Centralité
                    centrality = nx.degree_centrality(G_nx)
                    density = nx.density(G_nx)
                    
                    # 3. Préparer les nœuds avec leurs clusters et scores
                    for node_name in G_nx.nodes:
                        nodes.append({
                            "id": node_name, 
                            "name": node_name,
                            "cluster": partition[node_name],
                            "centrality": round(centrality[node_name], 2)
                        })
                    
                    metrics = {
                        "modularity": round(modularity, 3),
                        "density": round(density, 3),
                        "nb_communities": len(set(partition.values())),
                        "nb_nodes": len(G_nx.nodes),
                        "nb_edges": len(G_nx.edges)
                    }
        except Exception as e:
            print(f"❌ Erreur Neo4j Graph : {e}")

    return templates.TemplateResponse(
        request=request,
        name="graph.html",
        context={
            "nodes": nodes,
            "links": links,
            "metrics": metrics,
            "active_page": "graph"
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)