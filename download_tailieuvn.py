from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import urllib.parse
import re
import json
import concurrent.futures
import time
from tqdm import tqdm  # For progress bar

# Selenium driver instance - create once and reuse
driver = None

def setup_driver():
    """Initialize the Chrome driver with optimized settings"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')  # Disable images
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_pdf_url(doc_url):
    """Extract PDF URL from a tailieu.vn document page"""
    global driver
    
    try:
        # Navigate to the document page
        driver.get(doc_url)
        
        # Wait for iframe with explicit wait instead of sleep
        wait = WebDriverWait(driver, 10)
        iframe = wait.until(
            EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src, 'viewfile') and contains(@src, 'readpdf2')]"))
        )
        
        iframe_src = iframe.get_attribute("src")
        if not iframe_src:
            return None, None
            
        # Parse URL to extract PDF link
        parsed_url = urllib.parse.urlparse(iframe_src)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if 'f' in query_params:
            pdf_url = query_params['f'][0]
            
            # Extract filename from URL
            filename_match = re.search(r'/([^/]+\.pdf)', pdf_url)
            if filename_match:
                filename = filename_match.group(1)
            else:
                # Use document URL to create a unique filename
                doc_id = doc_url.split('/')[-1]
                filename = f"tailieu_{doc_id}.pdf"
                
            return pdf_url, filename
    except Exception as e:
        print(f"❌ Error extracting PDF URL from {doc_url}: {e}")
    
    return None, None

def download_file(pdf_url, output_path):
    """Download a file with progress tracking and retry mechanism"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Stream download with a timeout
            with requests.get(pdf_url, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    print(f"❌ Failed to download (status code {response.status_code})")
                    retry_count += 1
                    time.sleep(2)
                    continue
                    
                # Get file size for progress bar if available
                file_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                with open(output_path, 'wb') as f:
                    if file_size:
                        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return True
        except Exception as e:
            print(f"❌ Download attempt {retry_count + 1} failed: {e}")
            retry_count += 1
            time.sleep(2)
    
    return False

def process_url(doc_url, output_folder):
    """Process a single URL to extract and download PDF"""
    try:
        pdf_url, filename = extract_pdf_url(doc_url)
        
        if not pdf_url or not filename:
            print(f"❌ Could not extract PDF URL from {doc_url}")
            return False
            
        output_path = os.path.join(output_folder, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"⏩ File already exists: {output_path}")
            return True
            
        # Download the PDF
        if download_file(pdf_url, output_path):
            print(f"✅ Successfully downloaded: {output_path}")
            return True
        else:
            print(f"❌ Failed to download PDF from {doc_url}")
            return False
    except Exception as e:
        print(f"❌ Error processing {doc_url}: {e}")
        return False

def download_pdf_from_tailieu(tailieu_urls, output_folder='download_tailieuvn', max_workers=1):
    """Download PDFs from tailieu.vn using parallel processing"""
    global driver
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the driver once
    driver = setup_driver()
    
    try:
        # Track stats
        total = len(tailieu_urls)
        successful = 0
        failed = 0
        
        print(f"🚀 Starting download of {total} PDFs with {max_workers} worker(s)...")
        
        # Process each URL (in sequence rather than parallel for stability)
        for i, url in enumerate(tailieu_urls):
            print(f"\n[{i+1}/{total}] Processing: {url}")
            if process_url(url, output_folder):
                successful += 1
            else:
                failed += 1
                
        print(f"\n📊 Download Summary:")
        print(f"   - Total URLs: {total}")
        print(f"   - Successfully downloaded: {successful}")
        print(f"   - Failed: {failed}")
            
    finally:
        # Clean up the driver
        if driver:
            driver.quit()

if __name__ == "__main__":
    # Read URLs from JSON file
    try:
        with open('tailieu_urls.json', 'r') as f:
            tailieu_urls = json.load(f)
        
        # You can adjust max_workers to control parallelism (use 1 for stability)
        download_pdf_from_tailieu(tailieu_urls, max_workers=1)
    except FileNotFoundError:
        print("❌ File tailieu_urls.json not found")
    except json.JSONDecodeError:
        print("❌ Error parsing tailieu_urls.json - invalid JSON format")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")