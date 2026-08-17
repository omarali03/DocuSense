from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRResult:
    text: str
    confidence: float
    page: int


def _ocr_pil_image(img: object, page: int) -> OCRResult:
    import pytesseract

    data: dict[str, list[object]] = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT
    )
    words: list[str] = []
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word_str = str(word).strip()
        if word_str:
            conf = float(str(data["conf"][i]))
            if conf > 0:
                words.append(word_str)
                confs.append(conf)

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return OCRResult(text=" ".join(words), confidence=avg_conf / 100.0, page=page)


def ocr_pdf_pages(pdf_path: Path) -> list[OCRResult]:
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path))
    return [_ocr_pil_image(img, i + 1) for i, img in enumerate(images)]


def ocr_image_file(image_path: Path) -> OCRResult:
    from PIL import Image

    img = Image.open(image_path)
    return _ocr_pil_image(img, 0)
