"""Pure retrieval helpers; no network or model downloads on import."""
import hashlib
import math
import re


def markdown_chunks(text, tokenizer, size=384, overlap=64):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > overlap >= 0')
    heading = []
    sections = re.split(r'(?m)^(#{1,6} .+)$', text)
    for part in sections:
        if re.fullmatch(r'#{1,6} .+', part):
            level = len(part) - len(part.lstrip('#'))
            heading = heading[:level - 1] + [part.lstrip('# ').strip()]
            continue
        if not part.strip():
            continue
        # Slice the original text using tokenizer offsets: no decode-induced corruption.
        offsets = tokenizer(part, add_special_tokens=False,
                            return_offsets_mapping=True)['offset_mapping']
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            content = part[offsets[start][0]:offsets[end - 1][1]].strip()
            if content:
                yield content, ' > '.join(heading)
            if end == len(offsets):
                break


def stable_id(source, ordinal, content):
    return hashlib.sha256(f'{source}\0{ordinal}\0{content}'.encode()).hexdigest()


def weighted_rrf(rankings, weights=(1.0, 1.0), constant=60):
    if constant <= 0 or len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError('Invalid RRF configuration')
    scores, sources = {}, {}
    for results, weight in zip(rankings, weights):
        seen = set()
        for rank, (identifier, _, source) in enumerate(results, 1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0) + weight / (constant + rank)
            sources.setdefault(identifier, {}).update(source)
    return [(key, value, sources[key]) for key, value in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def metrics(ranked_sources, relevant, k):
    if not relevant or k <= 0:
        raise ValueError('Nonempty labels and positive k required')
    unique = list(dict.fromkeys(ranked_sources))[:k]
    gains = [int(source in relevant) for source in unique]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {'recall': sum(gains) / len(relevant),
            'mrr': next((1 / (i + 1) for i, g in enumerate(gains) if g), 0),
            'ndcg': dcg / ideal}
