import wikipediaapi
import random
import json

# =====================================
# Fixed Wikipedia URL Collection Script
# =====================================
# This script samples Wikipedia articles from diverse topics and saves 200 fixed URLs for evaluation.
# It uses the Wikipedia API and ensures articles are of sufficient length and not duplicated.

# Diverse topics to sample from
TOPICS = [
    "Science", "Technology", "History", "Geography", "Mathematics", "Art", "Music", "Literature", "Sports", "Politics",
    "Biology", "Physics", "Chemistry", "Medicine", "Engineering", "Economics", "Philosophy", "Psychology", "Sociology", "Education",
    "Environment", "Astronomy", "Computer Science", "Business", "Law", "Religion", "Culture", "Film", "Architecture", "Languages"
]

wiki = wikipediaapi.Wikipedia(user_agent='HybridRAGStudent/1.0 (2024aa05851@wilp.bits-pilani.ac.in)', language='en')

# Helper to get random articles from a category
# Returns a list of articles with title, URL, and word count
def get_articles_from_category(category, min_words=200, max_articles=20):
    cat = wiki.page(f"Category:{category}")
    articles = []
    for title, page in cat.categorymembers.items():
        if page.ns == 0 and len(page.text.split()) >= min_words:
            articles.append({
                "title": page.title,
                "url": page.fullurl,
                "word_count": len(page.text.split())
            })
        if len(articles) >= max_articles:
            break
    return articles

fixed_articles = []
used_titles = set()
random.shuffle(TOPICS)

# Main loop: sample articles from each topic until 200 unique articles are collected
for topic in TOPICS:
    print(f"Sampling articles from topic: {topic} (collected so far: {len(fixed_articles)})")
    articles = get_articles_from_category(topic)
    for article in articles:
        if article["title"] not in used_titles:
            fixed_articles.append(article)
            used_titles.add(article["title"])
        if len(fixed_articles) >= 200:
            break
    if len(fixed_articles) >= 200:
        break
    print(f"  -> After topic '{topic}': {len(fixed_articles)} articles collected.")

# Save only URLs to data/fixed_urls.json for downstream use
import os
data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(data_dir, exist_ok=True)
fixed_urls = [a["url"] for a in fixed_articles]
with open(os.path.join(data_dir, "fixed_urls.json"), "w") as f:
    json.dump(fixed_urls, f, indent=2)

print(f"Collected {len(fixed_urls)} fixed Wikipedia URLs.")
