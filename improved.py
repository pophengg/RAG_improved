"""Improved retrieval, inheriting the baseline final generator unchanged."""
from pathlib import Path
import json
import hashlib
from langchain_core.documents import Document
from opensearchpy.helpers import bulk
from baseline import AnthropicStyleContextualRAG
from retrieval import markdown_chunks, stable_id, weighted_rrf


class ImprovedRAG(AnthropicStyleContextualRAG):
    def __init__(self, *args, rerank=True, **kwargs):
        # The overridden index initializer separates both experiments.
        super().__init__(*args, **kwargs)
        self.rerank_enabled = rerank
        self.reranker = None
        self.cache_path = Path('.cache/context.json')
        self.context_cache = (json.loads(self.cache_path.read_text(encoding='utf-8'))
                              if self.cache_path.exists() else {})

    def _create_or_update_indices(self):
        self.vector_index_name = 'improved-vector-index'
        self.bm25_index_name = 'improved-bm25-index'
        shared = {'content': {'type': 'text', 'analyzer': 'thai'},
                  'contextualized_content': {'type': 'text', 'analyzer': 'thai'},
                  'heading': {'type': 'text', 'analyzer': 'thai'},
                  'source': {'type': 'keyword'}, 'doc_id': {'type': 'keyword'},
                  'chunk_id': {'type': 'keyword'}}
        for name, vector in [(self.vector_index_name, True), (self.bm25_index_name, False)]:
            if self.opensearch_client.indices.exists(index=name):
                continue
            properties = dict(shared)
            settings = {}
            if vector:
                settings['index.knn'] = True
                properties['embedding'] = {
                    'type': 'knn_vector', 'dimension': self.embedding_dim,
                    'method': {'name': 'hnsw', 'engine': 'lucene',
                               'space_type': 'cosinesimil'}}
            self.opensearch_client.indices.create(index=name, body={
                'settings': settings, 'mappings': {'properties': properties}})

    def load_documents(self, md_paths):
        documents = []
        for filename in sorted(md_paths):
            path = Path(filename)
            full = path.read_text(encoding='utf-8')
            for ordinal, (content, heading) in enumerate(markdown_chunks(
                    full, self.embed_model.tokenizer)):
                documents.append(Document(page_content=content, metadata={
                    'source': path.name, 'full_document': full,
                    'heading': heading, 'ordinal': ordinal}))
        return documents

    def generate_context(self, document_content, chunk_content):
        key = hashlib.sha256((document_content + '\0' + chunk_content).encode()).hexdigest()
        if key not in self.context_cache:
            context = super().generate_context(document_content, chunk_content)
            if context.startswith('Qwen API error:') or not context:
                raise RuntimeError('Context generation failed; abort indexing')
            self.context_cache[key] = context
            self.cache_path.parent.mkdir(exist_ok=True)
            self.cache_path.write_text(json.dumps(self.context_cache, ensure_ascii=False), encoding='utf-8')
        return self.context_cache[key]

    def add_documents_with_context(self, docs):
        # Bounded batches; embeddings are generated together, not one request per chunk.
        for start in range(0, len(docs), 16):
            batch = docs[start:start + 16]
            contexts = [self.generate_context(d.metadata['full_document'], d.page_content) for d in batch]
            texts = [f"{d.metadata['heading']}\n{c}\n{d.page_content}" for d, c in zip(batch, contexts)]
            # Detect model truncation instead of silently dropping evidence.
            if any(len(self.embed_model.tokenizer.encode(t)) > self.embed_model.max_seq_length for t in texts):
                raise ValueError('Contextual chunk exceeds embedding model token budget')
            vectors = self.embed_model.encode(texts, normalize_embeddings=True, batch_size=16)
            actions = []
            for doc, context, vector in zip(batch, contexts, vectors):
                identifier = stable_id(doc.metadata['source'], doc.metadata['ordinal'], doc.page_content)
                record = {'content': doc.page_content, 'contextualized_content': context,
                          'heading': doc.metadata['heading'], 'source': doc.metadata['source'],
                          'doc_id': identifier, 'chunk_id': str(doc.metadata['ordinal'])}
                actions.extend([
                    {'_index': self.vector_index_name, '_id': identifier,
                     '_source': {**record, 'embedding': vector.tolist()}},
                    {'_index': self.bm25_index_name, '_id': identifier, '_source': record}])
            bulk(self.opensearch_client, actions, raise_on_error=True)
        for name in (self.vector_index_name, self.bm25_index_name):
            self.opensearch_client.indices.refresh(index=name)
        return len(docs)

    def sparse_search(self, query, k=30):
        response = self.opensearch_client.search(index=self.bm25_index_name, body={
            'size': k, 'query': {'multi_match': {'query': query,
                'fields': ['content^2', 'heading^2', 'contextualized_content'], 'type': 'best_fields'}}})
        return [(h['_id'], h['_score'], h['_source']) for h in response['hits']['hits']]

    def hybrid_search(self, query, k=5, rrf_k=60):
        if k <= 0:
            raise ValueError('k must be positive')
        candidates = max(30, k * 6)
        fused = weighted_rrf([self.sparse_search(query, candidates),
                              self.dense_search(query, candidates)], constant=rrf_k)
        # Remove duplicate evidence before running the more expensive reranker.
        seen, pool = set(), []
        for item in fused:
            content = item[2]['content'].strip()
            if content not in seen:
                seen.add(content)
                item[2]['rrf_score'] = item[1]
                pool.append(item)
        pool = pool[:candidates]
        if self.rerank_enabled and pool:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024)
            scores = self.reranker.predict([(query, p[2]['content']) for p in pool])
            pool = [(p[0], float(score), {**p[2], 'rerank_score': float(score)})
                    for p, score in zip(pool, scores)]
            pool.sort(key=lambda item: (-item[1], item[0]))
        return pool[:k]

    def truncate_context(self, context, max_tokens=4000):
        # Conservative UTF-8 byte budget works for Thai without whitespace;
        # avoids adding another generation tokenizer. Not an exact Qwen token count.
        return context.encode('utf-8')[:max_tokens].decode('utf-8', errors='ignore')
