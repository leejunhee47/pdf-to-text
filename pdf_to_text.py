import io
import os
from google.cloud import vision
from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF

def pdf_to_images(pdf_path):
    """PDF 페이지를 이미지로 변환"""
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI 해상도
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    return images

def detect_text(image):
    """Google Cloud Vision API를 사용하여 이미지에서 텍스트 추출"""
    # 이미지 크기 조정 함수 적용
    image = resize_if_needed(image, max_size=4000)
    
    client = vision.ImageAnnotatorClient()
    
    # 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr = img_byte_arr.getvalue()
    
    image = vision.Image(content=img_byte_arr)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    if texts:
        return texts[0].description
    return ""

def resize_if_needed(image, max_size=4000):
    """필요한 경우 이미지 크기 조정"""
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def main():
    # Google Cloud 인증 설정
    # 환경 변수 GOOGLE_APPLICATION_CREDENTIALS에 서비스 계정 키 파일 경로 설정 필요
    # 예: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "키 입력"
    pdf_path = "서유견문_etc.pdf"  # 입력 PDF 파일 경로
    output_file = "output_etc.txt"  # 출력 텍스트 파일 경로
    
    try:
        # PDF를 이미지로 변환
        images = pdf_to_images(pdf_path)
        
        # 각 이미지에서 텍스트 추출
        all_text = ""
        for i, image in enumerate(images):
            # 디버깅을 위해 이미지 저장
            image.save(f"debug_page/debug_page_{i+100}.jpg")
            print(f"페이지 {i+1} 처리 중...")
            text = detect_text(image)
            all_text += f"\n\n--- 페이지 {i+1} ---\n\n{text}"
        
        # 결과를 파일로 저장
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
            
        print(f"텍스트 추출이 완료되었습니다. 결과는 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
