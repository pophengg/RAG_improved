import ast
import unittest
from pathlib import Path
from retrieval import markdown_chunks, weighted_rrf, metrics, stable_id


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class RetrievalTests(unittest.TestCase):
    def test_thai_without_spaces_is_bounded_and_preserved(self):
        text = 'ข้อความภาษาไทยไม่มีช่องว่าง' * 20
        chunks = list(markdown_chunks(text, CharTokenizer(), 64, 8))
        self.assertTrue(all(len(c) <= 64 for c, _ in chunks))
        restored = chunks[0][0] + ''.join(c[8:] for c, _ in chunks[1:])
        self.assertEqual(restored, text)

    def test_heading_path(self):
        chunks = list(markdown_chunks('# A\n## B\ntext\n## C\nnext', CharTokenizer()))
        self.assertEqual([h for _, h in chunks], ['A > B', 'A > C'])

    def test_invalid_chunk_configuration(self):
        with self.assertRaises(ValueError):
            list(markdown_chunks('text', CharTokenizer(), 10, 10))

    def test_fusion_merges_sources_and_uses_one_based_rank(self):
        hits = weighted_rrf([[('a', 999, {'content': 'x'})],
                             [('a', 0.2, {'contextualized_content': 'y'})]])
        self.assertAlmostEqual(hits[0][1], 2 / 61)
        self.assertEqual(hits[0][2], {'content': 'x', 'contextualized_content': 'y'})

    def test_duplicate_ranker_hits_do_not_inflate_score(self):
        hits = weighted_rrf([[('a', 1, {}), ('a', 2, {})]], weights=(1,))
        self.assertAlmostEqual(hits[0][1], 1 / 61)

    def test_metrics_deduplicate_sources(self):
        result = metrics(['a', 'a', 'b'], {'a', 'b'}, 2)
        self.assertEqual(result, {'recall': 1, 'mrr': 1, 'ndcg': 1})

    def test_metrics_known_ranking(self):
        result = metrics(['irrelevant', 'correct'], {'correct'}, 2)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['mrr'], 0.5)
        self.assertAlmostEqual(result['ndcg'], 0.6309297535714575)

    def test_ids_stable_and_source_specific(self):
        self.assertEqual(stable_id('a', 1, 'text'), stable_id('a', 1, 'text'))
        self.assertNotEqual(stable_id('a', 1, 'text'), stable_id('b', 1, 'text'))

    def test_generation_code_remains_identical_to_upstream(self):
        def method(path, name):
            tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
            return ast.dump(next(n for n in ast.walk(tree)
                                 if isinstance(n, ast.FunctionDef) and n.name == name))
        for name in ('generate_response', 'call_qwen_api'):
            self.assertEqual(method('baseline.py', name), method('upstream/authenticRAG.py', name))
        improved = ast.parse(Path('improved.py').read_text(encoding='utf-8'))
        methods = {n.name for n in ast.walk(improved) if isinstance(n, ast.FunctionDef)}
        self.assertNotIn('generate_response', methods)
        self.assertNotIn('call_qwen_api', methods)


if __name__ == '__main__':
    unittest.main()
