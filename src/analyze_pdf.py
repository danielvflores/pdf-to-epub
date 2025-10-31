#!/usr/bin/env python3
"""
Análisis de estructura de PDF para entender el layout
"""
import fitz

def analyze_pdf_structure(pdf_path: str, page_num: int = 10):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    print(f"=== ANÁLISIS PÁGINA {page_num + 1} ===")
    
    text_dict = page.get_text("dict")
    print(f"📄 Bloques de contenido: {len(text_dict['blocks'])}")
    
    # Analizar cada bloque
    for i, block in enumerate(text_dict["blocks"][:5]):
        if "lines" in block:
            bbox = block["bbox"]
            print(f"\n📝 Bloque {i+1} (TEXTO):")
            print(f"   Posición: x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]-bbox[0]:.0f}, h={bbox[3]-bbox[1]:.0f}")
            
            text_content = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text_content += span["text"] + " "
            
            print(f"   Texto: '{text_content[:100]}...'")
            print(f"   Font: {block['lines'][0]['spans'][0].get('font', 'N/A') if block['lines'] else 'N/A'}")
        else:
            bbox = block["bbox"]
            print(f"\n🖼️  Bloque {i+1} (IMAGEN):")
            print(f"   Posición: x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]-bbox[0]:.0f}, h={bbox[3]-bbox[1]:.0f}")
    
    images = page.get_images()
    print(f"\n🖼️  Total imágenes: {len(images)}")
    
    full_text = page.get_text()
    print(f"\n📄 Texto completo ({len(full_text)} chars):")
    print(f"'{full_text[:200]}...'")
    
    doc.close()

if __name__ == "__main__":
    analyze_pdf_structure("input/Rokudenashi Majutsu Koushi to Akashic Records Vol. 01 [Wishesubs].pdf")