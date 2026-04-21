import sys
import subprocess
import os
import re

try:
    import fitz
except ImportError:
    print("Installation de PyMuPDF...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

def generer_pdf_nettoye(nom_entree="10-.pdf", nom_sortie="Netoyage.pdf"):
    if not os.path.exists(nom_entree):
        print(f"Erreur : Le fichier {nom_entree} est introuvable.")
        return

    doc = fitz.open(nom_entree)
    nouveau_doc = fitz.open()
    
    nb_pages_total = len(doc)
    print(f"Nettoyage sélectif et suppression des blancs dans {nom_entree}...")

    phase = "TITRE_ET_RESUME" 
    mots_stop = ["sommaire", "table des matières", "liste des figures", "liste des tableaux", "abstract"]
    mot_relance = ["abréviation", "acronyme", "liste des abréviations"]

    for page_num in range(nb_pages_total):
        page = doc.load_page(page_num)
        texte_page = page.get_text().lower()

        # --- GESTION DES PHASES ---
        if phase == "TITRE_ET_RESUME" and any(m in texte_page for m in mots_stop):
            if "résumé" not in texte_page:
                phase = "SKIP"
                print(f"Phase suppression activée à la page {page_num + 1}")

        if phase == "SKIP" and any(m in texte_page for m in mot_relance):
            phase = "KEEP_ALL"
            print(f"Phase extraction réactivée à la page {page_num + 1}")

        if phase == "SKIP":
            continue

        # --- FILTRAGE DES BLOCS ET PAGES BLANCHES ---
        blocs = page.get_text("blocks")
        blocs_utiles = []
        
        for b in blocs:
            texte_bloc = b[4].strip()
            # Nettoyage des lignes vides internes au bloc
            texte_bloc = "\n".join([line.strip() for line in texte_bloc.splitlines() if line.strip()])
            
            # Filtre : on ignore si vide, si auteur, ou si numéro de page
            if not texte_bloc or "Mathias Dezetter" in texte_bloc or texte_bloc.isdigit():
                continue
            
            # Filtre spécifique à la phase de titre
            if phase == "TITRE_ET_RESUME" and any(m in texte_bloc.lower() for m in mots_stop):
                continue
                
            blocs_utiles.append(texte_bloc)

        # Si après filtrage la page est vide, on ne la crée pas (Supprime les pages blanches)
        if not blocs_utiles:
            continue

        # --- CRÉATION DE LA PAGE ---
        nouvelle_page = nouveau_doc.new_page(width=page.rect.width, height=page.rect.height)
        y_offset = 50 
        
        for texte in blocs_utiles:
            try:
                # On insère le texte condensé sans lignes vides
                nouvelle_page.insert_text((50, y_offset), texte, fontsize=10)
                lignes = texte.count('\n') + 1
                y_offset += lignes * 12 + 6  # Espacement réduit entre blocs
            except Exception:
                pass
            
            if y_offset > page.rect.height - 50:
                break

    nouveau_doc.save(nom_sortie)
    nouveau_doc.close()
    doc.close()
    
    print(f"Nettoyage terminé. Pages blanches et lignes vides supprimées. Fichier : {nom_sortie}")

if __name__ == "__main__":
    generer_pdf_nettoye()