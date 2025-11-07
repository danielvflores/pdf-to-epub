#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import fitz
import click
from tqdm import tqdm
from ebooklib import epub
import re
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='🔧 %(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class PDFtoEPUBConverter:
    def __init__(self, input_dir: str = "input", output_dir: str = "output", mode: str = "auto"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"📂 Input: {self.input_dir.absolute()}")
        logger.info(f"📂 Output: {self.output_dir.absolute()}")
        logger.info(f"🎯 Modo: {mode}")

    def detect_document_type(self, pdf_path: Path) -> str:
        if self.mode != "auto":
            return self.mode
        logger.info("🔍 Detectando tipo de documento...")
        doc = fitz.open(pdf_path)
        total_images = total_text_chars = 0
        pages_to_analyze = min(5, len(doc))
        
        for page_num in range(pages_to_analyze):
            page = doc[page_num]
            total_images += len(page.get_images())
            total_text_chars += len(page.get_text().strip())
        
        doc.close()
        avg_images_per_page = total_images / pages_to_analyze
        avg_text_per_page = total_text_chars / pages_to_analyze
        
        if avg_images_per_page > 10 or avg_text_per_page < 500:
            logger.info(f"📚 Detectado como MANGA/CÓMIC (imágenes: {avg_images_per_page:.1f}/página, texto: {avg_text_per_page:.0f} chars/página)")
            return "comic"
        else:
            logger.info(f"📄 Detectado como TEXTO (imágenes: {avg_images_per_page:.1f}/página, texto: {avg_text_per_page:.0f} chars/página)")
            return "text"

    def extract_pages_as_images(self, pdf_path: Path) -> Dict:
        logger.info(f"📖 Extrayendo páginas como imágenes: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        content = {
            'pages': [],
            'metadata': {
                'title': doc.metadata.get('title') or pdf_path.stem,
                'author': doc.metadata.get('author') or 'Autor Desconocido',
                'total_pages': len(doc)
            }
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            content['pages'].append({
                'data': pix.tobytes("png"),
                'filename': f"page_{page_num+1:03d}.png",
                'page_number': page_num + 1
            })
            pix = None
        
        doc.close()
        logger.info(f"✅ Extraídas {len(content['pages'])} páginas como imágenes")
        return content

    def generate_comic_epub(self, content: Dict, output_path: Path) -> bool:
        logger.info(f"📚 Generando EPUB de cómic: {output_path.name}")
        try:
            book = epub.EpubBook()
            metadata = content['metadata']
            book.set_identifier('comic-pdf-to-epub-' + output_path.stem)
            book.set_title(metadata['title'])
            book.set_language('es')
            book.add_author(metadata['author'])
            
            comic_css = '''
                @page { margin: 0; padding: 0; }
                body { margin: 0; padding: 0; text-align: center; background: black; }
                .page-image { width: 100%; height: 100vh; object-fit: contain; display: block; margin: 0 auto; }
                .chapter { page-break-before: always; page-break-after: always; }
            '''
            
            nav_css = epub.EpubItem(uid="nav-css", file_name="style/comic.css", media_type="text/css", content=comic_css)
            book.add_item(nav_css)
            chapters = []
            
            for i, page_data in enumerate(content['pages']):
                img_item = epub.EpubImage()
                img_item.file_name = f"images/{page_data['filename']}"
                img_item.media_type = "image/png"
                img_item.content = page_data['data']
                book.add_item(img_item)
                
                chapter_html = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Página {page_data['page_number']}</title>
    <link rel="stylesheet" type="text/css" href="../style/comic.css"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
</head>
<body>
    <div class="chapter">
        <img src="../images/{page_data['filename']}" alt="Página {page_data['page_number']}" class="page-image"/>
    </div>
</body>
</html>'''
                
                chapter = epub.EpubHtml(title=f'Página {page_data["page_number"]}', file_name=f'chapter_{i+1:03d}.xhtml', lang='es')
                try:
                    chapter.set_content(chapter_html.encode('utf-8'))
                except:
                    chapter.content = chapter_html
                
                chapter.add_item(nav_css)
                book.add_item(chapter)
                chapters.append(chapter)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📄 Procesadas {i + 1}/{len(content['pages'])} páginas...")
            
            book.toc = [(epub.Link(f"chapter_{i+1:03d}.xhtml", f"Página {page['page_number']}", f"chapter_{i+1}"))
                        for i, page in enumerate(content['pages'])]
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.spine = ['nav'] + chapters
            
            logger.info("💾 Escribiendo archivo EPUB...")
            epub.write_epub(str(output_path), book, {})
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ EPUB de cómic generado: {output_path} ({output_path.stat().st_size} bytes)")
                return True
            else:
                logger.error(f"❌ Archivo EPUB no se creó: {output_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error generando EPUB de cómic: {str(e)}")
            return False

    def extract_text_and_images(self, pdf_path: Path) -> Dict:
        logger.info(f"📖 Extrayendo contenido tradicional de: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        content = {
            'text_blocks': [],
            'images': [],
            'metadata': {
                'title': doc.metadata.get('title') or pdf_path.stem,
                'author': doc.metadata.get('author') or 'Autor Desconocido',
                'total_pages': len(doc)
            }
        }
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            page_height = page_rect.height
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    bbox = block.get('bbox', None)
                    y_pos = bbox[1] if bbox and len(bbox) >= 4 else 0
                    block_text = ""
                    font_info = {}
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                block_text += text + " "
                                if not font_info:
                                    font_info = {
                                        'font': span.get('font', ''),
                                        'size': span.get('size', 12),
                                        'flags': span.get('flags', 0)
                                    }
                    block_text = block_text.strip()
                    if block_text:
                        content['text_blocks'].append({
                            'text': block_text,
                            'page': page_num + 1,
                            'font_info': font_info,
                            'bbox': bbox,
                            'y_position': y_pos,
                            'page_height': page_height
                        })

            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - getattr(pix, 'alpha', 0) < 4:
                        img_data = pix.tobytes("png")
                        content['images'].append({
                            'data': img_data,
                            'filename': f"image_p{page_num+1}_{img_index}.png",
                            'page': page_num + 1,
                            'page_height': page_height
                        })
                    pix = None
                except Exception:
                    logger.warning(f"⚠️  Error extrayendo imagen en página {page_num+1}")

        doc.close()

        # Detect and filter headers/footers from text_blocks
        content['text_blocks'] = self._detect_and_filter_headers_footers(content['text_blocks'], content['metadata']['total_pages'])

        logger.info(f"✅ Extraídos {len(content['text_blocks'])} bloques de texto y {len(content['images'])} imágenes")
        return content

    def extract_structured_content(self, pdf_path: Path) -> Dict:
        logger.info(f"📖 Extrayendo contenido estructurado de: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        content = {
            'elements': [],
            'metadata': {
                'title': doc.metadata.get('title') or pdf_path.stem,
                'author': doc.metadata.get('author') or 'Autor Desconocido',
                'total_pages': len(doc)
            }
        }
        
        total_elements = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements = []
            page_rect = page.rect  # dimensiones de la página
            text_dict = page.get_text("dict")
            images = page.get_images()
            
            image_areas = []
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    try:
                        img_rects = page.get_image_rects(xref)
                        img_rect = img_rects[0] if img_rects else fitz.Rect(0, 0, 100, 100)
                    except:
                        img_rect = fitz.Rect(0, page_num * 100, 100, (page_num * 100) + 100)
                    
                    expanded_rect = fitz.Rect(img_rect[0] - 10, img_rect[1] - 10, img_rect[2] + 10, img_rect[3] + 10)
                    image_areas.append({
                        'rect': expanded_rect,
                        'original_rect': img_rect,
                        'xref': xref,
                        'index': img_index
                    })
                except Exception as e:
                    logger.warning(f"⚠️  Error procesando imagen en página {page_num+1}: {str(e)}")
            
            overlapping_areas = {}
            independent_text_blocks = []
            images_with_text = set()
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    bbox = block["bbox"]
                    text_rect = fitz.Rect(bbox)
                    
                    block_text = ""
                    font_info = {}
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                block_text += text + " "
                                if not font_info:
                                    font_info = {
                                        'font': span.get('font', ''),
                                        'size': span.get('size', 12),
                                        'flags': span.get('flags', 0)
                                    }
                    
                    block_text = block_text.strip()
                    
                    if block_text and len(block_text) > 1:
                        overlapping_image = None
                        for img_data in image_areas:
                            if self._rectangles_overlap(text_rect, img_data['rect']):
                                overlapping_image = img_data
                                break
                        
                        is_useful_text = self._is_useful_translation_text(block_text)
                        
                        if overlapping_image and is_useful_text:
                            img_index = overlapping_image['index']
                            if img_index not in overlapping_areas:
                                overlapping_areas[img_index] = {
                                    'text_rects': [text_rect],
                                    'image_area': overlapping_image['original_rect'],
                                    'texts': [block_text],
                                    'image_index': img_index
                                }
                            else:
                                overlapping_areas[img_index]['text_rects'].append(text_rect)
                                overlapping_areas[img_index]['texts'].append(block_text)
                            
                            images_with_text.add(img_index)
                        elif not overlapping_image:
                            independent_text_blocks.append({
                                'type': 'text',
                                'content': block_text,
                                'bbox': bbox,
                                'page': page_num + 1,
                                'font_info': font_info,
                                'y_position': bbox[1],
                                'page_height': page_rect.height
                            })
            
            page_elements.extend(independent_text_blocks)
            
            for img_data in image_areas:
                if img_data['index'] not in images_with_text:
                    try:
                        pix = fitz.Pixmap(doc, img_data['xref'])
                        if pix.n - pix.alpha < 4:
                            img_bytes = pix.tobytes("png")
                            page_elements.append({
                                'type': 'image',
                                'data': img_bytes,
                                'filename': f"img_p{page_num+1:03d}_{img_data['index']:02d}.png",
                                'page': page_num + 1,
                                                'bbox': img_data['original_rect'],
                                                'y_position': img_data['original_rect'][1] if img_data['original_rect'] else float(page_num * 1000 + img_data['index']),
                                                'page_height': page_rect.height
                            })
                        pix = None
                    except Exception as e:
                        logger.warning(f"⚠️  Error extrayendo imagen independiente {img_data['index']} en página {page_num+1}: {str(e)}")
            
            for img_index, overlap_data in overlapping_areas.items():
                try:
                    combined_rect = overlap_data['image_area']
                    for text_rect in overlap_data['text_rects']:
                        combined_rect = self._combine_rects(combined_rect, text_rect)
                    
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, clip=combined_rect)
                    img_data = pix.tobytes("png")
                    
                    page_elements.append({
                        'type': 'image',
                        'data': img_data,
                        'filename': f"combined_p{page_num+1:03d}_img{img_index:02d}.png",
                        'page': page_num + 1,
                        'bbox': combined_rect,
                        'y_position': combined_rect[1],
                        'page_height': page_rect.height,
                        'is_combined': True,
                        'text_count': len(overlap_data['texts'])
                    })
                    pix = None
                except Exception as e:
                    logger.warning(f"⚠️  Error renderizando imagen combinada {img_index} en página {page_num+1}: {str(e)}")
            
            page_elements.sort(key=lambda x: x['y_position'])
            content['elements'].extend(page_elements)
            total_elements += len(page_elements)
        
        doc.close()

        # Primero detectar y filtrar encabezados/pies repetidos dinámicamente
        content['elements'] = self._detect_and_filter_headers_footers(content['elements'], content['metadata']['total_pages'])

        cleaned_elements = self.clean_structured_content(content['elements'], content['metadata']['total_pages'])
        content['elements'] = cleaned_elements
        logger.info(f"✅ Extraídos {len(content['elements'])} elementos estructurados ({total_elements - len(cleaned_elements)} eliminados por limpieza)")
        return content

    def _is_useful_translation_text(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if len(text) < 3:
            return False
        
        noise_patterns = [
            r'^p\s*á\s*g\s*i\s*n\s*a', r'^\d+$', r'^[|]+$', r'^wishesubs$',
            r'^domen$', r'^reversed$', r'^dgf$', r'^\w{1,3}$'
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, text_lower):
                return False
        
        useful_patterns = [
            r'[.!?]', r'\b(que|qué|como|cómo|pero|y|es|está|son|está|dijo|dice)\b',
            r'[,;:]', r'\b(espera|bueno|ah|oh|sí|no|hola|hey)\b'
        ]
        
        for pattern in useful_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return len(text) > 20

    def _rectangles_overlap(self, rect1: fitz.Rect, rect2: fitz.Rect) -> bool:
        return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3])

    def _combine_rects(self, rect1: fitz.Rect, rect2: fitz.Rect) -> fitz.Rect:
        return fitz.Rect(min(rect1[0], rect2[0]), min(rect1[1], rect2[1]), max(rect1[2], rect2[2]), max(rect1[3], rect2[3]))

    def _normalize_text_key(self, text: str) -> str:
        """Normaliza texto para comparación: minúsculas, quitar múltiples espacios y caracteres de control."""
        if not text:
            return ""
        t = re.sub(r"\s+", " ", text.strip().lower())
        return t

    def _detect_and_filter_headers_footers(self, elements: List[Dict], total_pages: int) -> List[Dict]:
        """Detecta textos repetidos en la parte superior/inferior de las páginas (encabezados/pies/páginas)
        y filtra dichos elementos dinámicamente.

        - Busca textos que aparezcan en la zona superior (<12% de la página) o inferior (>88%) en muchas páginas.
        - Si un texto aparece en >= threshold páginas se considera encabezado/pie y se elimina.
        - También detecta numeración de página (solo dígitos o 'N/N') que aparezca repetida y la elimina.
        """
        if not elements:
            return elements

        top_candidates = {}
        bottom_candidates = {}
        pages_seen_by_text_top = {}
        pages_seen_by_text_bottom = {}
        pages_seen_by_reduced_top = {}
        pages_seen_by_reduced_bottom = {}
        reduced_to_textkeys_top = {}
        reduced_to_textkeys_bottom = {}

        # Build candidate lists
        for el in elements:
            if el.get('type') != 'text':
                continue
            text = el.get('content', '').strip()
            if not text:
                continue
            text_key = self._normalize_text_key(text)
            page = el.get('page', 0)
            page_height = el.get('page_height', None)
            y_pos = el.get('y_position', None)
            if page_height and y_pos is not None:
                norm_y = float(y_pos) / float(page_height) if page_height else 0.5
            else:
                # si no tenemos tamaños, usar heurístico por bbox
                bbox = el.get('bbox')
                if bbox and len(bbox) >= 4:
                    # bbox[1] es y0
                    norm_y = float(bbox[1]) / float(bbox[3] - bbox[1] + 1)
                else:
                    norm_y = 0.5

            if norm_y <= 0.12:
                pages_seen_by_text_top.setdefault(text_key, set()).add(page)
                top_candidates.setdefault(text_key, text)
                # reduced form - solo letras, sin números para agrupar "neokhaos 1", "neokhaos 15", etc
                reduced = re.sub(r'[^a-z]+', '', text_key).strip()
                if reduced and len(reduced) >= 3:  # solo si hay al menos 3 letras
                    pages_seen_by_reduced_top.setdefault(reduced, set()).add(page)
                    reduced_to_textkeys_top.setdefault(reduced, set()).add(text_key)
            elif norm_y >= 0.88:
                pages_seen_by_text_bottom.setdefault(text_key, set()).add(page)
                bottom_candidates.setdefault(text_key, text)
                reduced = re.sub(r'[^a-z]+', '', text_key).strip()
                if reduced and len(reduced) >= 3:  # solo si hay al menos 3 letras
                    pages_seen_by_reduced_bottom.setdefault(reduced, set()).add(page)
                    reduced_to_textkeys_bottom.setdefault(reduced, set()).add(text_key)

        threshold = max(2, int(total_pages * 0.4))  # Umbral más bajo: 40% de páginas o mínimo 2

        headers = {k for k, s in pages_seen_by_text_top.items() if len(s) >= threshold}
        footers = {k for k, s in pages_seen_by_text_bottom.items() if len(s) >= threshold}

        # also consider reduced keys (strip digits/punct) to catch footer/header variants with page numbers
        reduced_headers = {r for r, s in pages_seen_by_reduced_top.items() if len(s) >= threshold}
        reduced_footers = {r for r, s in pages_seen_by_reduced_bottom.items() if len(s) >= threshold}

        for r in reduced_headers:
            for tk in reduced_to_textkeys_top.get(r, []):
                headers.add(tk)

        for r in reduced_footers:
            for tk in reduced_to_textkeys_bottom.get(r, []):
                footers.add(tk)

        # Detect numeric page numbers among candidates (also remove)
        def is_page_number_text(s: str) -> bool:
            s_strip = s.strip()
            if re.match(r'^\d+$', s_strip):
                return True
            if re.match(r'^\d+\s*/\s*\d+$', s_strip):
                return True
            # Roman numerals? simple check
            if re.match(r'^[ivxlcdm]+$', s_strip.lower()):
                return True
            return False

        # Add numeric frequent candidates to headers/footers
        for k in list(headers):
            txt = k
            if is_page_number_text(txt):
                headers.add(k)

        for k in list(footers):
            txt = k
            if is_page_number_text(txt):
                footers.add(k)

        # Now filter elements
        cleaned = []
        for el in elements:
            if el.get('type') != 'text':
                cleaned.append(el)
                continue
            text = el.get('content', '').strip()
            text_key = self._normalize_text_key(text)

            if text_key in headers or text_key in footers:
                # skip repeated header/footer
                continue

            # also skip if detected as page number pattern and it appears frequently
            if is_page_number_text(text_key):
                # check how many pages this page number appears in candidate maps
                top_count = len(pages_seen_by_text_top.get(text_key, set()))
                bottom_count = len(pages_seen_by_text_bottom.get(text_key, set()))
                if max(top_count, bottom_count) >= threshold:
                    continue

            cleaned.append(el)

        return cleaned

    def clean_structured_content(self, elements: List[Dict], total_pages: int) -> List[Dict]:
        logger.info("🧹 Limpiando contenido estructurado...")
        text_frequency = {}
        for element in elements:
            if element['type'] == 'text':
                text = element['content'].lower().strip()
                if text:
                    text_frequency[text] = text_frequency.get(text, 0) + 1
        
        threshold = max(3, total_pages // 4)
        frequent_texts = {text for text, count in text_frequency.items() if count >= threshold}
        cleaned_elements = []
        
        for element in elements:
            if element['type'] == 'image':
                cleaned_elements.append(element)
            elif element['type'] == 'text':
                text = element['content'].strip()
                text_lower = text.lower()
                
                if not text or len(text) < 2:
                    continue
                if text_lower in frequent_texts:
                    continue
                if (self._is_page_number(text) or self._is_header_footer(text, total_pages) or 
                    self._is_watermark_or_noise(text)):
                    continue
                
                cleaned_elements.append(element)
        
        logger.info(f"✅ Limpieza completada: {len(cleaned_elements)} elementos conservados")
        return cleaned_elements

    def generate_structured_epub(self, content: Dict, output_path: Path) -> bool:
        logger.info(f"📚 Generando EPUB estructurado: {output_path.name}")
        try:
            book = epub.EpubBook()
            metadata = content['metadata']
            book.set_identifier('structured-pdf-to-epub-' + output_path.stem)
            book.set_title(metadata['title'])
            book.set_language('es')
            book.add_author(metadata['author'])
            
            enhanced_css = '''
                body { font-family: "Times New Roman", serif; line-height: 1.6; margin: 1em; color: #000; background: #fff; }
                p { margin: 0.5em 0; text-align: justify; text-indent: 1em; }
                .large-text { font-size: 1.2em; font-weight: bold; text-align: center; margin: 1em 0; text-indent: 0; }
                .heading { font-size: 1.4em; font-weight: bold; text-align: center; margin: 1.5em 0 1em 0; text-indent: 0; }
                .dialogue { font-style: italic; background-color: #f9f9f9; border-left: 3px solid #ccc; padding: 0.5em; margin: 0.5em 0; text-indent: 0; font-size: 0.95em; }
                .image-container { text-align: center; margin: 1em 0; page-break-inside: avoid; }
                .content-image { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .page-break { page-break-before: always; }
                .chapter-break { page-break-before: always; margin-top: 2em; }
            '''
            
            nav_css = epub.EpubItem(uid="nav-css", file_name="style/enhanced.css", media_type="text/css", content=enhanced_css)
            book.add_item(nav_css)
            
            html_content = self._build_structured_html(content, enhanced_css)
            if not html_content or len(html_content.strip()) < 100:
                logger.error("❌ Contenido HTML estructurado vacío")
                return False
            
            chapter = epub.EpubHtml(title='Contenido Principal', file_name='chapter_main.xhtml', lang='es')
            try:
                chapter.set_content(html_content.encode('utf-8'))
            except:
                chapter.content = html_content
            
            chapter.add_item(nav_css)
            book.add_item(chapter)

            images_added = 0
            for element in content['elements']:
                if element['type'] == 'image':
                    try:
                        img_item = epub.EpubImage()
                        img_item.file_name = f"images/{element['filename']}"
                        img_item.media_type = "image/png"
                        img_item.content = element['data']
                        book.add_item(img_item)
                        images_added += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Error agregando imagen {element['filename']}: {str(e)}")
            
            book.toc = (epub.Link("chapter_main.xhtml", "Contenido Principal", "main"),)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.spine = ['nav', chapter]
            
            epub.write_epub(str(output_path), book, {})
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ EPUB estructurado generado: {output_path} ({output_path.stat().st_size} bytes)")
                return True
            else:
                logger.error(f"❌ EPUB estructurado no se creó: {output_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error generando EPUB estructurado: {str(e)}")
            return False

    def clean_content(self, content: Dict) -> Dict:
        logger.info("🧹 Limpiando contenido...")
        text_blocks = content['text_blocks']
        total_pages = content['metadata']['total_pages']
        
        text_frequency = {}
        for block in text_blocks:
            text = block['text'].strip().lower()
            if text:
                text_frequency[text] = text_frequency.get(text, 0) + 1
        
        threshold = max(3, total_pages // 4)
        frequent_texts = {text for text, count in text_frequency.items() if count >= threshold}
        cleaned_blocks = []
        
        for block in text_blocks:
            text = block['text'].strip()
            text_lower = text.lower()
            
            if not text or len(text) < 2:
                continue
            if text_lower in frequent_texts:
                continue
            if (self._is_page_number(text) or self._is_header_footer(text, total_pages) or 
                self._is_watermark_or_noise(text)):
                continue
            
            cleaned_blocks.append(block)
        
        content['text_blocks'] = cleaned_blocks
        logger.info(f"✅ Contenido limpiado: {len(cleaned_blocks)} bloques válidos (eliminados {len(text_blocks) - len(cleaned_blocks)})")
        return content

    def _is_watermark_or_noise(self, text: str) -> bool:
        text_lower = text.lower().strip()
        patterns = ['wishesubs', 'domen', 'reversed', 'trans', 'dgf', 'scanlation', 'translator', 
                   'translation', 'group', 'team', 'sub', 'subs', 'página', 'page', 'p á g i n a',
                   'capítulo', 'chapter', 'vol', 'volume', 'www.', 'http', '.com', '.net', '.org']
        
        if len(text_lower) < 30:
            for pattern in patterns:
                if pattern in text_lower:
                    return True
        
        return text.isupper() and len(text) < 50

    def _is_page_number(self, text: str) -> bool:
        patterns = [r'^\d+$', r'^página\s*\d+', r'^\d+\s*de\s*\d+$']
        for pattern in patterns:
            if re.match(pattern, text.lower().strip()):
                return True
        return False

    def _is_header_footer(self, text: str, total_pages: int) -> bool:
        if len(text) > 100:
            return False
        patterns = [r'copyright|©', r'capítulo|chapter', r'www\.|http', r'página|page']
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                return True
        return False

    def generate_epub(self, content: Dict, output_path: Path) -> bool:
        logger.info(f"📚 Generando EPUB: {output_path.name}")
        try:
            book = epub.EpubBook()
            metadata = content['metadata']
            book.set_identifier('pdf-to-epub-' + output_path.stem)
            book.set_title(metadata['title'])
            book.set_language('es')
            book.add_author(metadata['author'])
            
            chapter_content = self._build_html_content(content)
            if not chapter_content or len(chapter_content.strip()) < 100:
                logger.error("❌ Contenido HTML vacío o muy pequeño")
                return False
            
            c1 = epub.EpubHtml(title='Contenido Principal', file_name='chapter_1.xhtml', lang='es')
            try:
                c1.set_content(chapter_content.encode('utf-8'))
            except:
                c1.content = chapter_content
            
            book.add_item(c1)
            
            images_added = 0
            for img in content['images']:
                try:
                    img_item = epub.EpubImage()
                    img_item.file_name = f"images/{img['filename']}"
                    img_item.media_type = "image/png"
                    img_item.content = img['data']
                    book.add_item(img_item)
                    images_added += 1
                except Exception as e:
                    logger.warning(f"⚠️  Error agregando imagen {img['filename']}: {str(e)}")
            
            book.toc = (epub.Link("chapter_1.xhtml", "Contenido Principal", "chapter_1"),)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.spine = ['nav', c1]
            
            epub.write_epub(str(output_path), book, {})
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ EPUB generado exitosamente: {output_path} ({output_path.stat().st_size} bytes)")
                return True
            else:
                logger.error(f"❌ Archivo EPUB no se creó correctamente: {output_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error generando EPUB: {str(e)}")
            return False

    def _build_structured_html(self, content: Dict, css_content: str) -> str:
        elements = content.get('elements', [])
        if not elements:
            return self._create_minimal_html("No hay contenido estructurado disponible")
        
        title = content["metadata"]["title"].strip()
        if not title or len(title) < 2:
            title = "Documento Convertido"
        
        html_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html>',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<head>',
            f'<title>{self._escape_html(title)}</title>',
            '<link rel="stylesheet" type="text/css" href="style/enhanced.css"/>',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>',
            '</head>',
            '<body>'
        ]
        
        if title != "Documento Convertido":
            html_parts.append(f'<h1 class="heading">{self._escape_html(title)}</h1>')
        
        current_page = 0
        element_count = 0
        
        for i, element in enumerate(elements):
            try:
                if element['page'] != current_page:
                    if current_page > 0:
                        html_parts.append('<div class="page-break"></div>')
                    current_page = element['page']
                
                if element['type'] == 'text':
                    text = element['content'].strip()
                    if not text:
                        continue
                    
                    if text.startswith('[Diálogo]:'):
                        css_class = "dialogue"
                        text = text.replace('[Diálogo]:', '').strip()
                    else:
                        css_class = self._determine_text_style(element, text)
                    
                    escaped_text = self._escape_html(text)
                    html_parts.append(f'<p class="{css_class}">{escaped_text}</p>')
                    element_count += 1
                    
                elif element['type'] == 'image':
                    filename = element['filename']
                    html_parts.append(f'''
                    <div class="image-container">
                        <img src="images/{filename}" alt="Imagen {element_count + 1}" class="content-image"/>
                    </div>''')
                    element_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Error procesando elemento {i}: {str(e)}")
                continue
        
        html_parts.extend(['</body>', '</html>'])
        return self._validate_html_content('\n'.join(html_parts))

    def _determine_text_style(self, element: Dict, text: str) -> str:
        font_info = element.get('font_info', {})
        font_size = font_info.get('size', 12)
        font_flags = font_info.get('flags', 0)
        
        if font_size > 16 or (font_flags & 2**4):
            return "heading"
        if font_size > 14:
            return "large-text"
        if (len(text) < 50 and 
            any(word in text.lower() for word in ['capítulo', 'prólogo', 'epílogo', 'volumen'])):
            return "heading"
        return "normal-text"

    def _build_html_content(self, content):
        text_blocks = content.get('text_blocks', [])
        if not text_blocks:
            return '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Documento sin contenido</title></head>
<body><p>Este documento no contiene texto procesable.</p></body>
</html>'''
        
        title = content["metadata"]["title"].strip()
        if not title or len(title) < 2:
            title = "Documento Convertido"
        
        html_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html>',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<head>',
            f'<title>{self._escape_html(title)}</title>',
            '<style>',
            'body { font-family: serif; line-height: 1.6; margin: 2em; }',
            'p { text-align: justify; margin-bottom: 1em; }',
            '.page-break { page-break-before: always; }',
            '</style>',
            '</head>',
            '<body>'
        ]
        
        if title != "Documento Convertido":
            html_parts.append(f'<h1>{self._escape_html(title)}</h1>')
        
        current_page = 0
        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
                
            if block['page'] != current_page:
                if current_page > 0:
                    html_parts.append('<div class="page-break"></div>')
                current_page = block['page']
            
            escaped_text = self._escape_html(text)
            html_parts.append(f'<p>{escaped_text}</p>')
        
        html_parts.extend(['</body>', '</html>'])
        return self._validate_html_content('\n'.join(html_parts))

    def _escape_html(self, text: str) -> str:
        if not text:
            return ""
        
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in ' \t\n')
        
        return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                   .replace('"', '&quot;').replace("'", '&#39;'))

    def _validate_html_content(self, html_content: str) -> str:
        if not html_content or len(html_content.strip()) < 50:
            return '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Documento</title></head>
<body><p>Contenido del documento.</p></body>
</html>'''
        
        try:
            from xml.etree.ElementTree import fromstring
            fromstring(html_content)
            return html_content
        except Exception as e:
            if '<!DOCTYPE html>' not in html_content:
                return self._create_minimal_html("Error en contenido original")
            return html_content

    def _create_minimal_html(self, content: str = "Documento procesado") -> str:
        escaped_content = self._escape_html(content)
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Documento Convertido</title>
    <meta charset="UTF-8"/>
</head>
<body>
    <h1>Documento Convertido</h1>
    <p>{escaped_content}</p>
</body>
</html>'''

    def convert_pdf(self, pdf_path: Path) -> bool:
        try:
            conversion_mode = self.mode if self.mode in ["text", "comic", "structured"] else self.detect_document_type(pdf_path)
            output_path = self.output_dir / f"{pdf_path.stem}.epub"
            
            if conversion_mode == "comic":
                content = self.extract_pages_as_images(pdf_path)
                return self.generate_comic_epub(content, output_path)
            elif conversion_mode == "structured":
                content = self.extract_structured_content(pdf_path)
                return self.generate_structured_epub(content, output_path)
            else:
                content = self.extract_text_and_images(pdf_path)
                content = self.clean_content(content)
                return self.generate_epub(content, output_path)
        except Exception as e:
            logger.error(f"❌ Error convirtiendo {pdf_path.name}: {str(e)}")
            return False

    def convert_all(self) -> Dict[str, int]:
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("⚠️  No se encontraron archivos PDF en el directorio input")
            return {"total": 0, "success": 0, "failed": 0}
        
        results = {"total": len(pdf_files), "success": 0, "failed": 0}
        
        with tqdm(pdf_files, desc="📚 Convirtiendo PDFs", unit="archivo") as pbar:
            for pdf_path in pbar:
                pbar.set_description(f"📖 {pdf_path.name}")
                
                if self.convert_pdf(pdf_path):
                    results["success"] += 1
                    pbar.set_postfix({"✅ Exitosos": results["success"]})
                else:
                    results["failed"] += 1
                    pbar.set_postfix({"❌ Fallidos": results["failed"]})
        
        return results

@click.command()
@click.option('--input-dir', '-i', default="input", help="Directorio con archivos PDF")
@click.option('--output-dir', '-o', default="output", help="Directorio para archivos EPUB")
@click.option('--single', '-s', help="Convertir un solo archivo PDF")
@click.option('--mode', '-m', default="auto", type=click.Choice(['auto', 'text', 'comic', 'structured']), 
              help="Modo: auto (detecta), text (texto), comic (páginas), structured (texto+imágenes)")
def main(input_dir: str, output_dir: str, single: str, mode: str):
    print("🧩 PDF to EPUB Converter")
    print("=" * 50)
    
    converter = PDFtoEPUBConverter(input_dir, output_dir, mode)
    
    if single:
        pdf_path = Path(single)
        if not pdf_path.exists():
            logger.error(f"❌ Archivo no encontrado: {single}")
            sys.exit(1)
        
        if converter.convert_pdf(pdf_path):
            print(f"✅ Conversión exitosa: {pdf_path.name}")
        else:
            print(f"❌ Error en conversión: {pdf_path.name}")
            sys.exit(1)
    else:
        results = converter.convert_all()
        print(f"\n📊 Resultados:")
        print(f"   📚 Total: {results['total']}")
        print(f"   ✅ Exitosos: {results['success']}")
        print(f"   ❌ Fallidos: {results['failed']}")
        
        if results['failed'] > 0:
            sys.exit(1)

if __name__ == "__main__":
    main()