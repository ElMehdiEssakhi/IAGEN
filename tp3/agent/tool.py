from langchain_core.tools import tool
import config
from retrieval import recuperer_sources

@tool
def retriever_tool(query: str, image_path: str | None = None) -> str:
    """Search the knowledge base (PDF text passages AND images/figures/diagrams).
    - query: a short, focused search query (no greetings or filler).
    - image_path: optional path to an image, if the user gave one and wants
      similar images or related passages.
    Returns text passages and image references with their PDF and page."""
    sources = []
    if image_path:
        _, imgs = recuperer_sources(query or "image", k=config.K_IMAGES, type_resultat="image", image_requete=image_path)
        _, txts = recuperer_sources(query or "image", k=config.K, type_resultat="texte", image_requete=image_path)
    else:
        _, txts = recuperer_sources(query, k=config.K, type_resultat="texte")
        _, imgs = recuperer_sources(query, k=config.K_IMAGES, type_resultat="image")
    sources = txts + imgs

    lines = []
    for s in sources:
        header = f"[{s['type'].upper()}] {s['pdf']} p.{s['page']} (sim={s['similarite']:.2f})"
        if s["type"] == "image":
            lines.append(f"{header}\nImage file: {s['source']}\nCaption/content: {s['contenu']}")
        else:
            lines.append(f"{header}\n{s['contenu']}")
    return "\n\n---\n\n".join(lines)