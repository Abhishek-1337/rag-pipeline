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


        



      

rag = RAGPipeline()
chunks = rag.chunking("A document is a document, you can't say a sentence is a document", 2, 1)

vocab = rag.create_vocab(chunks)
print(vocab)
