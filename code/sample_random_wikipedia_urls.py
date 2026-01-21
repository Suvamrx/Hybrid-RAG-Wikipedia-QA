import json
import os
import random
import requests
import time
from typing import List

# Function to sample random Wikipedia article URLs
# Ensures each article has at least min_words words and is not in the exclude list
# Handles API errors, rate limiting, and logs progress

def get_random_wikipedia_urls(n: int, min_words: int = 200, exclude_urls: List[str] = None, max_attempts_per_url: int = 300) -> List[str]:
    """
    Sample n random Wikipedia article URLs with at least min_words words in the main text.
    Excludes any URLs in exclude_urls.
    Returns a list of valid Wikipedia article URLs.
    """
    S = set(exclude_urls) if exclude_urls else set()
    urls = []
    attempts = 0
    while len(urls) < n and attempts < n * max_attempts_per_url:
        # Request a random Wikipedia article URL
        resp = requests.get('https://en.wikipedia.org/wiki/Special:Random', allow_redirects=True, timeout=10, headers={"User-Agent": "HybridRAGStudent/1.0 (2024aa05851@wilp.bits-pilani.ac.in)"})
        url = resp.url
        # Skip if already sampled or in exclude list
        if url in S or url in urls:
            attempts += 1
            continue
        # Fetch article text using Wikipedia API to check length
        api_url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&redirects=1&titles={url.split("/wiki/")[-1]}'
        try:
            api_resp = requests.get(api_url, timeout=10, headers={"User-Agent": "HybridRAGStudent/1.0 (2024aa05851@wilp.bits-pilani.ac.in)"})
            if api_resp.status_code != 200:
                print(f"[WARN] API HTTP {api_resp.status_code} for {url}")
                attempts += 1
                time.sleep(2.0)
                continue
            try:
                data = api_resp.json()
            except Exception as e_json:
                print(f"[WARN] JSON decode failed for {url}: {e_json}\nResponse text: {api_resp.text[:200]}")
                attempts += 1
                time.sleep(2.0)
                continue
            pages = data.get('query', {}).get('pages', {})
            text = next(iter(pages.values())).get('extract', '')
            # Only keep articles with enough words
            if len(text.split()) >= min_words:
                urls.append(url)
                print(f"[LOG] Sampled: {url} ({len(text.split())} words)")
            else:
                print(f"[SKIP] Too short: {url} ({len(text.split())} words)")
        except Exception as e:
            print(f"[WARN] Failed to fetch or parse: {url} | {e}")
            attempts += 1
            time.sleep(2.0)
            continue
        attempts += 1
        time.sleep(1.0)  # Be polite to Wikipedia servers
    print(f"[LOG] Sampled {len(urls)} random Wikipedia URLs (target: {n})")
    return urls

# Main entry point for script execution
# Loads fixed URLs, samples random URLs, and saves them to disk

def main():
    # Ensure data directory exists
    os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
    fixed_urls_path = os.path.join(os.getcwd(), 'data', 'fixed_urls.json')
    random_urls_path = os.path.join(os.getcwd(), 'data', 'random_urls.json')
    # Load fixed URLs to avoid duplicates
    if os.path.exists(fixed_urls_path):
        with open(fixed_urls_path, 'r', encoding='utf-8') as f:
            fixed_urls = json.load(f)
    else:
        fixed_urls = []
    n_random = 300  # Number of random articles to sample (change as needed)
    random_urls = get_random_wikipedia_urls(n_random, min_words=200, exclude_urls=fixed_urls, max_attempts_per_url=300)
    # Save sampled random URLs to file
    with open(random_urls_path, 'w', encoding='utf-8') as f:
        json.dump(random_urls, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Saved {len(random_urls)} random Wikipedia URLs to {random_urls_path}")

if __name__ == '__main__':
    main()
