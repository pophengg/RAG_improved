"""Integration of improved retrieval with deterministic fake model/backend."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from retrieval import markdown_chunks, stable_id, weighted_rrf


def load_class():
    tree = ast.parse(Path('improved.py').read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {'AnthropicStyleContextualRAG': object, 'Path': Path,
                 'json': json, 'hashlib': hashlib, 'markdown_chunks': markdown_chunks,
                 'stable_id': stable_id, 'weighted_rrf': weighted_rrf, 'bulk': Mock()}
    exec(compile(tree, 'improved.py', 'exec'), namespace)
    return namespace['ImprovedRAG'], namespace


class IntegrationTests(unittest.TestCase):
    def test_reranking_changes_order_and_preserves_fusion_score(self):
        cls, _ = load_class()
        rag = cls.__new__(cls)
        rag.sparse_search = Mock(return_value=[('a', 99, {'content': 'wrong'}),
                                              ('b', 1, {'content': 'correct'})])
        rag.dense_search = Mock(return_value=[('a', 1, {'content': 'wrong'})])
        rag.rerank_enabled = True
        rag.reranker = SimpleNamespace(predict=Mock(return_value=[-2, 5]))
        hits = rag.hybrid_search('question', k=1)
        self.assertEqual(hits[0][0], 'b')
        self.assertEqual(hits[0][1], 5)
        self.assertIn('rrf_score', hits[0][2])
        rag.sparse_search.assert_called_once_with('question', 30)

    def test_index_passes_full_document_and_batches_normalized_embeddings(self):
        cls, namespace = load_class()
        rag = cls.__new__(cls)
        rag.generate_context = Mock(return_value='context')
        rag.embed_model = SimpleNamespace(
            encode=Mock(return_value=[SimpleNamespace(tolist=lambda: [0.1, 0.2])]),
            tokenizer=SimpleNamespace(encode=lambda text: list(text)), max_seq_length=1000)
        rag.opensearch_client = SimpleNamespace(indices=SimpleNamespace(refresh=Mock()))
        rag.vector_index_name, rag.bm25_index_name = 'vector', 'bm25'
        doc = SimpleNamespace(page_content='chunk', metadata={
            'full_document': 'full document', 'source': 'a.md', 'heading': 'Title', 'ordinal': 0})
        self.assertEqual(rag.add_documents_with_context([doc]), 1)
        rag.generate_context.assert_called_once_with('full document', 'chunk')
        self.assertTrue(rag.embed_model.encode.call_args.kwargs['normalize_embeddings'])
        actions = namespace['bulk'].call_args.args[1]
        self.assertEqual(actions[0]['_id'], actions[1]['_id'])
        self.assertEqual(actions[0]['_source']['source'], 'a.md')

    def test_thai_context_byte_budget(self):
        cls, _ = load_class()
        result = cls.truncate_context(None, 'ข้อความภาษาไทย' * 500, max_tokens=100)
        self.assertLessEqual(len(result.encode('utf-8')), 100)


if __name__ == '__main__':
    unittest.main()
