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
      


rag = RAGPipeline()
chunks = rag.chunking("A document is a document, you can't say a sentence is a document", 2, 1)
print(chunks)
