import wikipediaapi
import json
import os
import re
from nltk.tokenize import word_tokenize
import requests
from urllib.parse import unquote
import time

# =========================
# Wikipedia Chunking Script
# =========================
# This script loads Wikipedia article URLs (fixed and/or random), downloads their content,
# cleans and chunks the text, and saves the resulting chunks with metadata for RAG retrieval.
# It supports command-line flags to select which URL sources to use.

# Chunking parameters
CHUNK_SIZE = 300  # Number of tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between consecutive chunks (tokens)

# Load fixed Wikipedia URLs (required for --use-fixed)
with open(os.path.join(os.getcwd(), 'data', 'fixed_urls.json'), 'r') as f:
    fixed_urls = json.load(f)

# Parse command-line arguments to select which URLs to process
import argparse
parser = argparse.ArgumentParser(description="Chunk Wikipedia articles from fixed and/or random URLs.")
parser.add_argument('--use-fixed', action='store_true', help='Include fixed_urls.json')
parser.add_argument('--use-random', action='store_true', help='Include random_urls.json')
args = parser.parse_args()

# Build the list of URLs to process based on flags
urls = []
if args.use_fixed:
    urls.extend(fixed_urls)
if args.use_random:
    random_urls_path = os.path.join(os.getcwd(), 'data', 'random_urls.json')
    if os.path.exists(random_urls_path):
        with open(random_urls_path, 'r', encoding='utf-8') as f:
            random_urls = json.load(f)
        urls.extend(random_urls)
    else:
        print(f"[WARN] random_urls.json not found, skipping random URLs.")
if not urls:
    raise ValueError("No URLs to process. Use --use-fixed and/or --use-random.")

# Initialize Wikipedia API client
wiki = wikipediaapi.Wikipedia(user_agent='HybridRAGStudent/1.0 (2024aa05851@wilp.bits-pilani.ac.in)', language='en')

# Clean Wikipedia text: remove references, templates, and extra whitespace

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2], etc.
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # Remove templates
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Split text into overlapping chunks of tokens for retrieval

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = word_tokenize(text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if len(chunk) < 50:
            break  # Skip very short trailing chunks
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

# Main chunking loop: download, clean, and chunk each Wikipedia article
all_chunks = []
chunk_id = 0

print(f"[LOG] Starting chunking for {len(urls)} Wikipedia articles...")
for i, url in enumerate(urls):
    if i % 25 == 0 and i > 0:
        print(f"[LOG] Chunked {i}/{len(urls)} articles...")
    # Try multiple title variants to maximize Wikipedia API hit rate
    raw_title = url.split('/wiki/')[-1]
    tried_titles = [raw_title]
    if raw_title and not raw_title[0].isupper():
        tried_titles.append(raw_title[0].upper() + raw_title[1:])
    if '_' in raw_title:
        tried_titles.append(raw_title.replace('_', ' '))
    page = None
    for t in tried_titles:
        try:
            candidate = wiki.page(t)
            if candidate.exists():
                page = candidate
                break
        except Exception as e:
            print(f"[ERROR] Exception during wiki.page/exists for '{t}': {e}")
            time.sleep(2)
    used_fallback = False
    fallback_title = None
    # Fallback: use Wikipedia search API if direct lookup fails
    if not page or not page.exists():
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={raw_title}&format=json"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Calling Wikipedia search API for URL: {url} | Query: {raw_title} (attempt {attempt+1})")
                headers = {"User-Agent": "HybridRAGStudent/1.0 (2024aa05851@wilp.bits-pilani.ac.in)"}
                resp = requests.get(search_url, timeout=8, headers=headers)
                time.sleep(1.0)  # Avoid rate limiting
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"[DEBUG] Raw search API response for URL: {url} | Query: {raw_title}\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                    search_results = data.get('query', {}).get('search', [])
                    if search_results:
                        print(f"[DEBUG] Search results for URL: {url} | Query: {raw_title}")
                        for idx, result in enumerate(search_results):
                            print(f"    [{idx+1}] {result['title']}")
                        top_title = search_results[0]['title']
                        try:
                            candidate = wiki.page(top_title)
                            if candidate.exists():
                                page = candidate
                                used_fallback = True
                                fallback_title = top_title
                        except Exception as e:
                            print(f"[ERROR] Exception during fallback wiki.page/exists for '{top_title}': {e}")
                        break  # Success, exit retry loop
                    else:
                        print(f"[DEBUG] No search results for URL: {url} | Query: {raw_title}")
                        break
                else:
                    print(f"[DEBUG] Wikipedia search API returned status {resp.status_code} for URL: {url} | Query: {raw_title}")
            except Exception as e:
                print(f"[WARN] Wikipedia search API failed for {raw_title} (attempt {attempt+1}): {e}")
                time.sleep(2.0)  # Wait longer before retrying
    if used_fallback:
        print(f"[LOG] Fallback used for URL: {url} | Fallback title: {fallback_title}")
    if not page or not page.exists():
        print(f"[WARN] Page does not exist for URL: {url} (tried: {tried_titles})")
        continue
    title = page.title
    try:
        text = clean_text(page.text)
    except Exception as e:
        print(f"[ERROR] Exception during page.text for '{title}': {e}")
        continue
    # Extract intro/summary (first paragraph or summary attribute)
    intro = ''
    try:
        if hasattr(page, 'summary') and page.summary:
            intro = clean_text(page.summary)
        else:
            intro = text.split('\n\n')[0]
    except Exception as e:
        print(f"[ERROR] Exception during intro extraction for '{title}': {e}")
        intro = ''
    # Compose first chunk: title + intro
    first_chunk = f"Title: {title}\nIntro: {intro}"
    all_chunks.append({
        'chunk_id': f'{chunk_id}',
        'url': url,
        'title': title,
        'chunk_index': 0,
        'text': first_chunk
    })
    chunk_id += 1
    # Chunk the rest of the article (excluding intro)
    rest_text = text[len(intro):].strip() if intro else text
    if rest_text:
        try:
            chunks = chunk_text(rest_text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f'{chunk_id}',
                    'url': url,
                    'title': title,
                    'chunk_index': idx + 1,  # +1 because 0 is the intro chunk
                    'text': chunk
                })
                chunk_id += 1
            print(f"[LOG] Processed: {title} ({len(chunks)+1} chunks)")
        except Exception as e:
            print(f"[ERROR] Exception during chunking for '{title}': {e}")
            print(f"[LOG] Processed: {title} (1 chunk - intro only)")
    else:
        print(f"[LOG] Processed: {title} (1 chunk - intro only)")
print(f"[LOG] Finished chunking all articles. Total chunks: {len(all_chunks)}")

# Save all chunks with metadata to JSON file for downstream retrieval
os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
with open(os.path.join(os.getcwd(), 'data', 'wikipedia_chunks.json'), 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Total chunks created: {len(all_chunks)}")
