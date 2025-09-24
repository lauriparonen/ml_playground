import requests
from bs4 import BeautifulSoup

URL = "https://www.gutenberg.org/browse/scores/top#books-last30"

f"""
individual book texts utf-8 are at 
https://www.gutenberg.org/cache/epub/book_id/pgbook_id.txt

the link in the li element in the URL list is like:
<a href="/ebooks/1513">Romeo and Juliet by William Shakespeare (83336)</a>
where book_id = 1513
"""

#%% Crawler

def get_top_k_books(url, k=20):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    books_section = soup.find("h2", {"id": "books-last30"}).find_next("ol")
    links = books_section.find_all("a")[:k]

    book_list = []
    for a in links:
        href = a["href"]
        book_id = href.split("/")[-1]
        title = a.text
        txt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        book_list.append((title, txt_url))
    return book_list

def download_books(book_list):
    texts = []
    for title, url in book_list:
        r = requests.get(url)
        r.encoding = "utf-8"
        texts.append(r.text)
    return texts

#%% Data processing pipeline

import nltk
from nltk.stem import  WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")

def strip_gutenberg(text):
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    if start != -1:
        text = text[start:]
        text = text[text.find("\n")+1:]  

    end = text.find(end_marker)
    if end != -1:
        text = text[:end]

    return text

def tag_to_wordnet(tag):
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("R"): return wordnet.ADV
    return None

def process_text(text):
    # cleanup header/footer
    clean = strip_gutenberg(text)
    # tokenize 
    # - don't include punctuation
    # - only use words longer than 1 letter except for edge cases "a" and "I"
    tokens = [w.lower() for w in nltk.word_tokenize(clean) \
          if (w.isalpha() and (len(w) > 1 or w.lower() in {"a", "i"}))] 
    nltktext = nltk.Text(tokens)
    # lowercase
    lower = [w.lower() for w in nltktext]
    # pos-tagging
    tagged = nltk.pos_tag(lower)
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in tagged:
        wn_tag = tag_to_wordnet(tag)
        if wn_tag:
            lemmas.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmas.append(word)
    return lemmas

#%% Vocabulary

import numpy as np
from collections import Counter

def build_unified_vocab(processed_docs):
    vocab = []
    for doc in processed_docs:
        vocab.extend(doc)
    return np.unique(vocab)

def top_words(processed_docs, n=100):
    counter = Counter()
    for doc in processed_docs:
        counter.update(doc)
    return counter.most_common(n)

#%% Running the whole program

def top_k_downloader(url, k=20):
    books = get_top_k_books(url, k)
    texts = download_books(books)
    processed = [process_text(t) for t in texts] 

    print("downloaded books:")
    for title, url in books:
        print(f"- {title}: {url}")

    vocab = build_unified_vocab(processed)
    top100 = top_words(processed)

    # save top 100 words to file
    with open("top_100.txt", "w") as f:
        for w, f in top100:
            f.write(f"{w} {f}\n")

    return books, vocab

top_k_downloader(URL, k=20)