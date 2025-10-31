
"""
Script temporal para arreglar la duplicación de imágenes
"""

with open('pdf-to-epub.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_line = "            overlapping_areas = []"
new_line = "            overlapping_areas = {}  # Diccionario agrupado por imagen {image_index: info}"

pos = content.find(old_line)
if pos != -1:
    content = content[:pos] + new_line + content[pos + len(old_line):]

old_block = """                        if overlapping_image and is_useful_text:
                            # Texto superpuesto útil - marcar área para renderizar como imagen unificada
                            combined_area = self._combine_rects(text_rect, overlapping_image['original_rect'])
                            overlapping_areas.append({
                                'text_rect': text_rect,
                                'image_area': overlapping_image['original_rect'],
                                'text': block_text,
                                'combined_area': combined_area,
                                'image_index': overlapping_image['index']
                            })
                            images_with_text.add(overlapping_image['index'])
                            logger.debug(f"🖼️  Área texto+imagen marcada en página {page_num+1}: '{block_text[:30]}...'")"""

new_block = """                        if overlapping_image and is_useful_text:
                            # Texto superpuesto útil - agrupar por imagen
                            img_index = overlapping_image['index']
                            
                            if img_index not in overlapping_areas:
                                # Primera vez que vemos esta imagen con texto
                                overlapping_areas[img_index] = {
                                    'text_rects': [text_rect],
                                    'image_area': overlapping_image['original_rect'],
                                    'texts': [block_text],
                                    'image_index': img_index
                                }
                            else:
                                # Ya tenemos texto en esta imagen, agregar más
                                overlapping_areas[img_index]['text_rects'].append(text_rect)
                                overlapping_areas[img_index]['texts'].append(block_text)
                            
                            images_with_text.add(img_index)
                            logger.debug(f"🖼️  Texto agrupado para imagen {img_index} en página {page_num+1}: '{block_text[:30]}...'")"""

pos = content.find(old_block)
if pos != -1:
    content = content[:pos] + new_block + content[pos + len(old_block):]

old_render = """            # Renderizar áreas de imagen+texto como imágenes unificadas
            for i, overlap_area in enumerate(overlapping_areas):
                try:
                    # Renderizar solo la región específica como imagen
                    combined_rect = overlap_area['combined_area']
                    
                    # Crear matriz de transformación para buena calidad
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom para mejor calidad
                    
                    # Renderizar solo la región específica
                    pix = page.get_pixmap(matrix=mat, clip=combined_rect)
                    img_data = pix.tobytes("png")
                    
                    page_elements.append({
                        'type': 'image',
                        'data': img_data,
                        'filename': f"combined_p{page_num+1:03d}_{i:02d}.png",
                        'page': page_num + 1,
                        'bbox': combined_rect,
                        'y_position': combined_rect[1],
                        'is_combined': True  # Marcar como imagen combinada
                    })
                    
                    logger.debug(f"✅ Región texto+imagen renderizada en página {page_num+1}")
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error renderizando región combinada en página {page_num+1}: {str(e)}")"""

new_render = """            # Renderizar UNA imagen combinada por cada imagen que tiene texto superpuesto
            for img_index, overlap_data in overlapping_areas.items():
                try:
                    # Combinar todos los rectángulos de texto con el área de la imagen
                    combined_rect = overlap_data['image_area']
                    for text_rect in overlap_data['text_rects']:
                        combined_rect = self._combine_rects(combined_rect, text_rect)
                    
                    # Crear matriz de transformación para buena calidad
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom para mejor calidad
                    
                    # Renderizar solo la región específica como imagen
                    pix = page.get_pixmap(matrix=mat, clip=combined_rect)
                    img_data = pix.tobytes("png")
                    
                    page_elements.append({
                        'type': 'image',
                        'data': img_data,
                        'filename': f"combined_p{page_num+1:03d}_img{img_index:02d}.png",
                        'page': page_num + 1,
                        'bbox': combined_rect,
                        'y_position': combined_rect[1],
                        'is_combined': True,  # Marcar como imagen combinada
                        'text_count': len(overlap_data['texts'])  # Para debug
                    })
                    
                    logger.debug(f"✅ Imagen combinada {img_index} con {len(overlap_data['texts'])} textos en página {page_num+1}")
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error renderizando imagen combinada {img_index} en página {page_num+1}: {str(e)}")"""

pos = content.find(old_render)
if pos != -1:
    content = content[:pos] + new_render + content[pos + len(old_render):]

with open('pdf-to-epub.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Archivo corregido para evitar duplicación de imágenes")