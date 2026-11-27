import base64
import mimetypes
from pathlib import Path

from llm import get_llm

PROMPT = ("Image d'un manuel d'appareil electronique. Decris-la factuellement "
          "en francais : pictogramme (forme, couleur), voyants et leur etat, "
          "texte ou code visible, ce que la figure illustre.")


def decrire_image(chemin: Path) -> str:
    mime = mimetypes.guess_type(chemin.name)[0] or "image/png"
    data = base64.b64encode(chemin.read_bytes()).decode()
    return get_llm(vision=True).invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
    }]).content