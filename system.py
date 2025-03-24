import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch, helpers

# Khởi tạo model
model = SentenceTransformer("minilm-custom-model/final")

# Kết nối Elasticsearch
ES_HOST = "http://localhost:9200"
INDEX_NAME = "documents"
es = Elasticsearch([ES_HOST])

# Thư mục chứa các file text
TEXT_FOLDER = "search_results/downloaded_texts"

# Hàm đọc nội dung từ file .txt
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")
        return ""

# Hàm chia nội dung thành từng đoạn
def split_text_into_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

# Hàm tạo index trên Elasticsearch
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "paragraph": {"type": "text"},
                    "paragraph_hash": {"type": "keyword"},  # 🔥 Thêm trường hash để hỗ trợ khử trùng
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    )
    print("✅ Đã tạo index Elasticsearch")

# Hàm tạo hash từ nội dung đoạn văn để khử trùng
def create_paragraph_hash(paragraph):
    # Loại bỏ khoảng trắng và chuyển về chữ thường để so sánh chính xác nội dung
    normalized_para = re.sub(r'\s+', '', paragraph.lower())
    return hash(normalized_para)

# Hàm lưu dữ liệu vào Elasticsearch
def index_documents():
    create_index()
    actions = []
    batch_size = 500
    indexed_count = 0
    skipped_count = 0
    paragraph_hashes = set()  # 🔥 Tập hợp lưu các hash để kiểm tra trùng lặp

    for filename in os.listdir(TEXT_FOLDER):
        file_path = os.path.join(TEXT_FOLDER, filename)
        text_content = read_text_file(file_path)

        if not text_content.strip():
            print(f"⚠️ Bỏ qua file rỗng: {filename}")
            continue

        paragraphs = split_text_into_paragraphs(text_content)

        for para in paragraphs:
            # 🔥 Kiểm tra và bỏ qua đoạn văn quá ngắn
            if len(para.split()) < 5:  # Bỏ qua đoạn văn có ít hơn 5 từ
                skipped_count += 1
                continue
                
            para_hash = create_paragraph_hash(para)
            
            # 🔥 Kiểm tra trùng lặp
            if para_hash in paragraph_hashes:
                skipped_count += 1
                continue
                
            paragraph_hashes.add(para_hash)
            
            embedding = model.encode(para).tolist()
            actions.append({
                "_index": INDEX_NAME,
                "_source": {
                    "filename": filename,
                    "paragraph": para,
                    "paragraph_hash": str(para_hash),  # Lưu hash để dễ truy vấn sau này
                    "embedding": embedding
                }
            })

            indexed_count += 1

            if len(actions) >= batch_size:
                helpers.bulk(es, actions)
                actions = []

    if actions:
        helpers.bulk(es, actions)

    print(f"✅ Đã index {indexed_count} đoạn văn vào Elasticsearch")
    print(f"⚠️ Đã bỏ qua {skipped_count} đoạn văn (quá ngắn hoặc trùng lặp)")
# Hàm tách câu từ đoạn văn
def split_into_sentences(text):
    text = re.sub(r'\.+', '.', text)  # Xoá dấu chấm dư thừa (.. -> .)
    sentences = re.split(r'\.\s+', text)  # Tách câu dựa trên dấu chấm và khoảng trắng
    sentences = [s.strip() for s in sentences if s.strip()]  # Loại bỏ khoảng trắng thừa
    return sentences

# Hàm tìm kiếm dựa trên từng câu của input
def hybrid_search_by_sentence(input_text, top_k=10, num_candidates=200):
    input_sentences = split_into_sentences(input_text)  # Tách input thành các câu nhỏ
    
    all_results = []
    
    for sentence in input_sentences:
        print(f"\n🔍 Tìm kiếm cho câu: \"{sentence}\"")
        input_embedding = model.encode(sentence).tolist()

        query = {
            "size": top_k * 2,
            "knn": {
                "field": "embedding",
                "query_vector": input_embedding,
                "k": top_k * 2,
                "num_candidates": num_candidates
            }
        }

        response = es.search(index=INDEX_NAME, body=query)
        results = response["hits"]["hits"]
        
        unique_results = {}

        for res in results:
            filename = res["_source"]["filename"]
            paragraph = res["_source"]["paragraph"]
            para_hash = res["_source"]["paragraph_hash"]
            para_embedding = res["_source"]["embedding"]
            
            score = util.cos_sim(
                torch.tensor(input_embedding, dtype=torch.float32),
                torch.tensor(para_embedding, dtype=torch.float32)
            ).item()

            if para_hash not in unique_results:
                unique_results[para_hash] = {
                    "filename": filename,
                    "paragraph": paragraph,
                    "score": score
                }

        sorted_results = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        all_results.extend(sorted_results)
        
        # Hiển thị kết quả cho từng câu input
        for i, result in enumerate(sorted_results, 1):
            print(f"#{i} 📄 File: {result['filename']}")
            print(f"🔹 Đoạn văn: {result['paragraph']}")
            print(f"📊 Độ tương đồng: {result['score']:.4f}\n")

    # 🔥 Tính độ đa dạng của kết quả tổng hợp
    unique_files = len(set(result["filename"] for result in all_results))
    print(f"\n📑 Tổng số lượng file duy nhất ở trong tất cả kết quả: {unique_files}/{len(all_results)}")
