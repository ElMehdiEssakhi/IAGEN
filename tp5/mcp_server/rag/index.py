import hashlib
import sys
from pathlib import Path

# Permet de lancer le fichier directement (python mcp_server/rag/index.py ...)
# comme un module (python -m mcp_server.rag.index ...) depuis tp5.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
import openai

from mcp_server.rag.extraction import extraire_pdf
from mcp_server.rag.vision import decrire_image

DATA = Path(__file__).resolve().parent.parent / "data"
MANUELS = DATA / "manuels"
TEXTES = DATA / "extraction/textes"
FIGURES = DATA / "extraction/figures"
CHROMA_DIR = DATA / "chroma"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("manuels", metadata={"hnsw:space": "cosine"})


def pdfs() -> list[tuple[Path, str]]:
    """[(chemin, doc_id)] pour chaque PDF du dossier manuels."""
    return [(p, hashlib.sha256(p.read_bytes()).hexdigest()[:12])
            for p in sorted(MANUELS.glob("*.pdf"))]


def decouper(texte: str, taille: int = 800, chevauchement: int = 100) -> list[str]:
    texte = " ".join(texte.split())
    return [texte[i:i + taille] for i in range(0, len(texte), taille - chevauchement)]


# ---------- Etape 1 : extraire (local, aucun appel API) ----------

def extraire() -> None:
    for pdf, doc_id in pdfs():
        res = extraire_pdf(pdf, doc_id, TEXTES, FIGURES)
        print(f"{pdf.name} ({doc_id}) : {res.nombre_pages} pages, {res.nombre_images} images")
        for a in res.avertissements:
            print("  !", a)


# ---------- Etape 2 : decrire (reprenable, un .txt par image) ----------

def decrire() -> None:
    for png in sorted(FIGURES.glob("*.png")):
        txt = png.with_suffix(".txt")
        if txt.exists():
            continue
        try:
            desc = decrire_image(png)
        except openai.RateLimitError:
            print("Quota atteint : relance 'decrire' plus tard (reprise automatique).")
            return
        if desc and desc.strip():
            txt.write_text(desc, encoding="utf-8")
            print("decrite :", png.name)


# ---------- Etape 3 : vectoriser (embedding dans Chroma, rejouable) ----------

def vectoriser() -> None:
    col = get_collection()
    for pdf, doc_id in pdfs():
        ids, docs, metas = [], [], []

        for txt in sorted(TEXTES.glob(f"{doc_id}-p*.txt")):
            p = int(txt.stem.split("-p")[1])
            for i, morceau in enumerate(decouper(txt.read_text(encoding="utf-8"))):
                ids.append(f"{doc_id}-p{p}-c{i}")
                docs.append(morceau)
                metas.append({"doc_id": doc_id, "document": pdf.name, "page": p, "type": "texte"})

        for txt in sorted(FIGURES.glob(f"{doc_id}-*.txt")):
            p = int(txt.stem.split("-p")[1].split("-")[0])
            png = txt.with_suffix(".png")
            ids.append(txt.stem)
            docs.append(txt.read_text(encoding="utf-8"))
            metas.append({
                "doc_id": doc_id,
                "document": pdf.name,
                "page": p,
                "type": "figure",
                "figure": png.stem.split("-")[-1] if png.exists() else txt.stem,
                "chemin": png.relative_to(DATA).as_posix() if png.exists() else "",
            })

        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"{pdf.name} : {len(ids)} elements vectorises.")


# ---------- Utilitaires ----------

def lister() -> None:
    res: dict[str, str] = {}
    for m in get_collection().get(include=["metadatas"])["metadatas"]:
        res[m["doc_id"]] = m["document"]
    print(res)


def supprimer(doc_id: str) -> None:
    get_collection().delete(where={"doc_id": doc_id})
    for dossier in (TEXTES, FIGURES):
        for f in dossier.glob(f"{doc_id}-*"):
            f.unlink()


COMMANDES = {
    "extraire": extraire,
    "decrire": decrire,
    "vectoriser": vectoriser,
    "lister": lister,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tout"
    if cmd == "tout":
        extraire(); decrire(); vectoriser()
    elif cmd == "supprimer":
        supprimer(sys.argv[2])
    elif cmd in COMMANDES:
        COMMANDES[cmd]()
    else:
        print("Commandes : extraire | decrire | vectoriser | tout | lister | supprimer <doc_id>")