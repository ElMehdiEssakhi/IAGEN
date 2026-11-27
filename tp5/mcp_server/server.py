import sys
from pathlib import Path

# Permet de lancer le fichier directement (python mcp_server/server.py)
# comme un module (python -m mcp_server.server) depuis tp5.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from mcp_server.rag.agent_rag import repondre

mcp = FastMCP("manuels", host="127.0.0.1", port=8000)


@mcp.tool()
def rag_multimodal(question: str, image: str | None = None) -> dict:
    """Recherche dans les manuels (passages et figures) et retourne une
    reponse sourcee. `image` est une reference (ex: img_px200_01)."""
    return repondre(question, image)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # http://127.0.0.1:8000/mcp
