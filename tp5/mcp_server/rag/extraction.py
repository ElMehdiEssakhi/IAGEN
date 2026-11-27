import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

MIN_PX = 64      # raw pixel size
MIN_PT = 60      # displayed size on the page (points)
MAX_PAGES = 3    # same image on more pages = logo / button


@dataclass
class ResultatExtraction:
    pdf: str
    doc_id: str
    nombre_pages: int = 0
    nombre_images: int = 0
    pages_sans_texte: list[int] = field(default_factory=list)
    avertissements: list[str] = field(default_factory=list)


def _occurrences(document) -> dict[int, int]:
    """xref -> number of pages where the image appears."""
    occ: dict[int, int] = {}
    for page in document:
        for xref in {im[0] for im in page.get_images(full=True)}:
            occ[xref] = occ.get(xref, 0) + 1
    return occ


def _pixmap_rgb(document, xref) -> pymupdf.Pixmap:
    """Embedded image -> RGB pixmap without alpha (handles JP2, CMYK, gray)."""
    pix = pymupdf.Pixmap(document, xref)
    if pix.colorspace is None or pix.colorspace.n != 3 or pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix


def _extraire_images_page(document, page, numero, doc_id, image_dir, occ, avertissements) -> int:
    enregistrees, vues = 0, set()

    for info in page.get_images(full=True):
        xref, smask = info[0], info[1]
        if xref in vues:
            continue
        vues.add(xref)

        png = image_dir / f"{doc_id}-p{numero}-img{xref}.png"
        if png.exists():
            enregistrees += 1
            continue

        try:
            if occ.get(xref, 0) > MAX_PAGES:
                logger.debug("p%d xref %d : image repetee, ignoree", numero, xref)
                continue
            if min(info[2], info[3]) < MIN_PX:
                continue

            rects = page.get_image_rects(xref)
            zone = (rects[0] & page.rect) if rects else None
            if zone is not None and (zone.is_empty or min(zone.width, zone.height) < MIN_PT):
                continue                                  # button / icon

            if smask and zone is not None:
                # separate mask: rendering the area is more faithful than the raw image
                page.get_pixmap(clip=zone, dpi=150).save(png)
            else:
                _pixmap_rgb(document, xref).save(png)
            enregistrees += 1

        except Exception as erreur:
            png.unlink(missing_ok=True)                   # never keep a half-written file
            avertissements.append(f"Page {numero} : image {xref} - {erreur}")
            logger.warning("p%d xref %d : %s", numero, xref, erreur)

    return enregistrees


def extraire_pdf(pdf_path, doc_id, text_dir: Path, image_dir: Path) -> ResultatExtraction:
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    res = ResultatExtraction(pdf=pdf_path.name, doc_id=doc_id)

    with pymupdf.open(pdf_path) as document:
        res.nombre_pages = len(document)
        occ = _occurrences(document)

        for numero, page in enumerate(document, start=1):
            texte = page.get_text("text", sort=True)
            fichier = text_dir / f"{doc_id}-p{numero}.txt"
            if not fichier.exists():
                fichier.write_text(texte, encoding="utf-8")
            if not texte.strip():
                res.pages_sans_texte.append(numero)

            res.nombre_images += _extraire_images_page(
                document, page, numero, doc_id, image_dir, occ, res.avertissements
            )

    if res.pages_sans_texte:
        res.avertissements.append(f"Pages sans texte (scan ?) : {res.pages_sans_texte}")

    logger.info("%s : %d pages, %d images", pdf_path.name, res.nombre_pages, res.nombre_images)
    return res