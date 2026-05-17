import math
from collections import Counter

class RAGPipeline: 

    def chunking(self, document, chunk_size = 250, overlap = 50):
        words = document.split()
        start = 0
        allChunks = []
        while start < len(words):
            end = start + chunk_size
            allChunks.append(" ".join(words[start:end]))
            start = end
        return allChunks

    def create_vocab(self, all_chunks):
        vocab = set()
        for chunk in all_chunks: 
            vocab.update(chunk.lower().split())
        return vocab

    def compute_tf(self, text, vocab):
        words = text.lower().split()
        word_count = Counter(text)
        total = len(text)
        return [word_count.get(word, 0) / total  for word in words]

    def compute_idf(self, all_chunks, vocab):
        length = len(all_chunks)
        idf = []
        for word in vocab:
            doc_count = sum(1 for chunk in all_chunks if word in chunk.lower().split())
            idf.append(math.log((doc_count + 1)/(length + 1)) + 1)
        return idf
    
    def tf_idf(self, text, vocab, idf):
        tf = self.compute_tf(text, vocab)
        return [i*j for i, j in zip(tf, idf)]
    
    def cosine_similarity(self, a, b):
        dot_product = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(y*y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot_product / ( mag_a * mag_b)
    
    def search(self, query_embed, stored, top_k): 
        scores = []
        for i, embed in enumerate(stored):
            sim = self.cosine_similarity(query_embed, embed)
            scores.append((i, sim))
        
        scores.sort(key = lambda x:x[1], reverse=True)
        return scores[:top_k]



document = 'TF (Term Frequency) measures how often a word appears inside one chunk, while IDF (Inverse Document Frequency) measures how rare that word is across all chunks. For example, if your chunks are "python python class", "python object", and "dog cat", then in the first chunk the TF of "python" is 2/3 because it appears 2 times out of 3 words. To compute IDF for "python", we look at all chunks and see that it appears in 2 out of 3 chunks, so its IDF is lower because it is relatively common. The word "class" appears in only 1 out of 3 chunks, so it gets a higher IDF because it is rarer and more informative. In short, TF tells you how important a word is in a specific chunk, and IDF tells you how unique that word is across the entire collection of chunks.'

rag = RAGPipeline()
all_chunks = rag.chunking(document, 10, 3)
vocab = rag.create_vocab(all_chunks)

idf = rag.compute_idf(all_chunks, vocab)
embeddings = [
    rag.tf_idf(chunk, vocab, idf) for chunk in all_chunks
]

user_query = "What is Term frequency"
query_embeddings = rag.tf_idf(user_query, vocab, idf)
scores = rag.search(query_embeddings, embeddings, 2)

retrieved = [(all_chunks[i], score) for i, score in scores]

print(retrieved)
