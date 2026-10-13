"""Evaluate retrieval only; never call the final generator."""
import argparse
import json
import statistics
import time
from pathlib import Path
from retrieval import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', default='eval/questions.json')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--output', default='results/retrieval.json')
    p.add_argument('--no-rerank', action='store_true')
    args = p.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    if args.k <= 0 or not labels or any(not q['relevant_sources'] for q in labels):
        p.error('Need positive k and nonempty relevance labels')
    from run import make_rag
    report = {'k': args.k, 'scope': 'document-level retrieval, synthetic corpus',
              'rerank': not args.no_rerank, 'modes': {}}
    for mode in ('baseline', 'improved'):
        rag = make_rag(mode, rerank=not args.no_rerank)
        manifest = json.loads(Path(f'{mode}-manifest.json').read_text(encoding='utf-8'))
        # Exclude first model load from timed measurements.
        rag.hybrid_search(labels[0]['question'], k=args.k)
        rows = []
        for q in labels:
            start = time.perf_counter()
            hits = rag.hybrid_search(q['question'], k=args.k)
            elapsed = time.perf_counter() - start
            sources = [d.get('source') or manifest[i] for i, _, d in hits]
            rows.append({'question': q['question'], 'sources': sources,
                         'seconds': elapsed, **metrics(sources, set(q['relevant_sources']), args.k)})
        report['modes'][mode] = {'queries': rows, 'mean': {
            key: statistics.mean(row[key] for row in rows)
            for key in ('recall', 'mrr', 'ndcg', 'seconds')}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['modes'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
