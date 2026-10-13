import argparse
import json
from pathlib import Path
from baseline import AnthropicStyleContextualRAG
from improved import ImprovedRAG


class BaselineRAG(AnthropicStyleContextualRAG):
    def _create_or_update_indices(self):
        self.vector_index_name = 'baseline-vector-index'
        self.bm25_index_name = 'baseline-bm25-index'
        super()._create_or_update_indices()


def make_rag(mode, host='localhost', port=9200, rerank=True):
    if mode == 'baseline':
        return BaselineRAG(host, port)
    return ImprovedRAG(host, port, rerank=rerank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['index', 'search', 'ask'])
    parser.add_argument('--mode', choices=['baseline', 'improved'], default='baseline')
    parser.add_argument('--corpus', default='corpus_input')
    parser.add_argument('--question')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9200)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--no-rerank', action='store_true')
    args = parser.parse_args()
    if args.k <= 0 or (args.action != 'index' and not args.question):
        parser.error('Positive k and a question for search/ask are required')
    paths = sorted(Path(args.corpus).glob('*.md'))
    if args.action == 'index' and not paths:
        parser.error('No Markdown documents found')
    if len({p.name for p in paths}) != len(paths):
        parser.error('Document filenames must be unique')
    rag = make_rag(args.mode, args.host, args.port, not args.no_rerank)
    if args.action == 'index':
        # Rebuild only this experiment's two indices to avoid stale chunks.
        for name in (rag.vector_index_name, rag.bm25_index_name):
            rag.opensearch_client.indices.delete(index=name)
        rag._create_or_update_indices()
        docs = rag.load_documents([str(p) for p in paths])
        rag.add_documents_with_context(docs)
        manifest = {f'doc_{i}': Path(d.metadata['source']).name for i, d in enumerate(docs)}
        Path(f'{args.mode}-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
        print(f'Indexed {len(docs)} chunks ({args.mode})')
    elif args.action == 'search':
        hits = rag.hybrid_search(args.question, k=args.k)
        print(json.dumps([{'id': i, 'score': s, 'document': {key: val for key, val in d.items() if key != 'embedding'}}
                          for i, s, d in hits], ensure_ascii=False, indent=2))
    else:
        # Both versions inherit precisely the same Qwen generator and prompt.
        print(rag.search_for_question(args.question, k=args.k))


if __name__ == '__main__':
    main()
