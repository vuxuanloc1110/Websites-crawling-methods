# lệnh để lấy tất cả sub urls 
# curl "http://web.archive.org/cdx/search/cdx?url=philosophy.vass.gov.vn/*&output=json&fl=original,timestamp&collapse=original" -o latest_sub_urls.json

import requests
import json
import os

# Đọc danh sách
with open("latest_sub_urls.json", "r") as f:
    data = json.load(f)

# Tạo thư mục lưu dữ liệu
os.makedirs("archived_pages", exist_ok=True)


for row in data:
    original_url, timestamp = row
    archived_url = f"https://web.archive.org/web/{timestamp}/{original_url}"

    # Tải trang về
    response = requests.get(archived_url)
    if response.status_code == 200:
        filename = original_url.replace("https://", "").replace("/", "_") + ".html"
        with open(f"archived_pages/{filename}", "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Đã tải: {archived_url}")
    else:
        print(f"❌ Không thể tải: {archived_url}")

print("🎉 Hoàn tất tải trang!")
