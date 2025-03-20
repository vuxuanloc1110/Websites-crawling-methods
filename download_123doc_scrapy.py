import json

def load_urls_from_json(file_path):
    """Đọc danh sách URL từ file JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    return urls
def modify_urls(urls):
    """Thêm 'text.' vào trước domain của URL, dù là 123docz.com hay 123docz.net."""
    modified_urls = []
    for url in urls:
        modified_url = re.sub(r"https://(www\.)?(123docz\.(com|net))", "https://text.123docz.net", url)
        modified_urls.append(modified_url)
    return modified_urls
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Khởi tạo Selenium WebDriver (cấu hình phù hợp với hệ thống của bạn)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Chạy chế độ ẩn để tiết kiệm tài nguyên
driver = webdriver.Chrome(options=options)

DOWNLOAD_FOLDER = "downloaded_pages"
TEXT_FOLDER = "extracted_texts"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

def download_html(url):
    """Tải trang HTML và lưu vào thư mục."""
    driver.get(url)
    try:
        WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        print(f"✅ Tải xong: {url}")
    except:
        print(f"❌ Lỗi tải trang: {url}")
        return None

    time.sleep(3)  # Chờ để trang tải xong
    page_source = driver.page_source

    # Lưu HTML vào file
    file_name = re.sub(r'[\\/*?:"<>|]', '_', url.split('/')[-1]) + ".html"
    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(page_source)

    return page_source, file_name

def extract_text_from_html(html_content):
    """Trích xuất nội dung từ thẻ chứa bài viết chính."""
    soup = BeautifulSoup(html_content, "html.parser")
    content_div = soup.find("div", class_="vf_view_pc md:col-span-7 col-span-12 pb-16 relative")
    
    if not content_div:
        print("❌ Không tìm thấy nội dung chính!")
        return ""

    return content_div.get_text(separator="\n", strip=True)

def save_text(text, filename):
    """Lưu văn bản vào file."""
    text_path = os.path.join(TEXT_FOLDER, filename + ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Đã lưu văn bản tại: {text_path}")
    # Đọc danh sách URL từ file JSON
collected_urls = load_urls_from_json("collected_urls.json")

# Sửa URL để thêm 'text.'
modified_urls = modify_urls(collected_urls)

# Tải từng URL và xử lý nội dung
for url in modified_urls:
    print(f"📥 Đang tải: {url}")
    html_content, file_name = download_html(url)

    if html_content:
        text_content = extract_text_from_html(html_content)
        save_text(text_content, file_name)

# Đóng trình duyệt
driver.quit()
print("✅ Hoàn tất tải và xử lý nội dung!")
