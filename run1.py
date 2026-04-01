"""
Part 1: Classical retrieval baselines.

Implements:
- BM25 (sparse)
- msmarco-MiniLM (dense)
- UAE-large-v1 (dense)

The script evaluates Recall@1 and Recall@5 on the query split provided in
the local JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "run1.py requires `torch` and `transformers` in the active Python "
        "environment."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PA3 Part 1 retrieval baselines")
    parser.add_argument("--queries-path", type=str, default="data/test_queries.json")
    parser.add_argument("--tools-path", type=str, default="data/tools.json")
    parser.add_argument(
        "--msmarco-model",
        type=str,
        default="sentence-transformers/msmarco-MiniLM-L6-cos-v5",
    )
    parser.add_argument(
        "--uae-model",
        type=str,
        default="WhereIsAI/UAE-Large-V1",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--uae-trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=str, default="part1_metrics.json")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tool_text(tool_name: str, description: str) -> str:
    return f"tool_id: {tool_name}\ntool description: {description}"


def prepare_dataset(
    queries_path: str,
    tools_path: str,
) -> Tuple[List[dict], List[str], List[str], Dict[str, int]]:
    queries = load_json(queries_path)
    tools = load_json(tools_path)

    if not isinstance(queries, list):
        raise ValueError(f"{queries_path} must contain a list of queries.")
    if not isinstance(tools, dict):
        raise ValueError(f"{tools_path} must contain a tool->description mapping.")

    tool_ids = list(tools.keys())
    tool_texts = [build_tool_text(tool_id, tools[tool_id]) for tool_id in tool_ids]
    tool_index = {tool_id: idx for idx, tool_id in enumerate(tool_ids)}

    for sample in queries:
        gold_tool = sample["gold_tool_name"]
        if gold_tool not in tool_index:
            raise ValueError(f"Gold tool {gold_tool!r} not present in tools file.")

    return queries, tool_ids, tool_texts, tool_index


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Retriever:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = len(documents)
        self.doc_tokens = [simple_tokenize(doc) for doc in documents]
        self.doc_lengths = np.array([len(doc) for doc in self.doc_tokens], dtype=np.float32)
        self.avg_doc_len = float(self.doc_lengths.mean()) if self.num_docs else 0.0

        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._build_index()

    def _build_index(self) -> None:
        doc_freq = Counter()
        for doc_id, tokens in enumerate(self.doc_tokens):
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.postings[term].append((doc_id, count))
            doc_freq.update(term_counts.keys())

        for term, freq in doc_freq.items():
            # Standard BM25 idf with +1 for numerical stability.
            self.idf[term] = math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> np.ndarray:
        query_terms = simple_tokenize(query)
        scores = np.zeros(self.num_docs, dtype=np.float32)
        if not query_terms or self.avg_doc_len == 0.0:
            return scores

        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, term_freq in self.postings[term]:
                denom = term_freq + self.k1 * (
                    1.0 - self.b + self.b * (self.doc_lengths[doc_id] / self.avg_doc_len)
                )
                scores[doc_id] += idf * (term_freq * (self.k1 + 1.0) / denom)
        return scores


def mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def safe_max_length(tokenizer: AutoTokenizer, requested_max_length: int) -> int:
    tokenizer_max_length = getattr(tokenizer, "model_max_length", requested_max_length)
    if tokenizer_max_length is None or tokenizer_max_length > 100_000:
        return requested_max_length
    return min(requested_max_length, int(tokenizer_max_length))


def load_encoder(
    model_name: str,
    local_files_only: bool,
    trust_remote_code: bool,
    device: str,
) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(f"Tokenizer for {model_name} has no pad/eos/unk token.")

    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
    trust_remote_code: bool = False,
) -> torch.Tensor:
    tokenizer, model = load_encoder(
        model_name=model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    final_max_length = safe_max_length(tokenizer, max_length)

    embeddings: List[torch.Tensor] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Encoding {Path(model_name).name}",
    ):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=final_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)
            token_embeddings = (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs[0]
            )
            pooled = mean_pool(token_embeddings, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


def evaluate_rankings(
    score_matrix: np.ndarray,
    gold_indices: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    ranked_indices = np.argsort(-score_matrix, axis=1)
    hits_at_1 = (ranked_indices[:, 0] == gold_indices).mean()
    hits_at_5 = (ranked_indices[:, :5] == gold_indices[:, None]).any(axis=1).mean()
    metrics = {
        "recall@1": float(hits_at_1),
        "recall@5": float(hits_at_5),
    }
    return metrics, ranked_indices


def print_metrics_table(results: Dict[str, Dict[str, float]]) -> None:
    header = f"{'Method':<18} {'Recall@1':>10} {'Recall@5':>10}"
    print("\n" + header)
    print("-" * len(header))
    for method, metrics in results.items():
        print(f"{method:<18} {metrics['recall@1']:>10.4f} {metrics['recall@5']:>10.4f}")


def run_bm25(
    queries: Sequence[dict],
    tool_texts: Sequence[str],
    tool_index: Dict[str, int],
) -> Tuple[Dict[str, float], np.ndarray]:
    retriever = BM25Retriever(tool_texts)
    score_rows = [
        retriever.score(sample["text"])
        for sample in tqdm(queries, desc="Scoring BM25")
    ]
    score_matrix = np.stack(score_rows, axis=0)
    gold_indices = np.array([tool_index[sample["gold_tool_name"]] for sample in queries])
    return evaluate_rankings(score_matrix, gold_indices)


def run_dense(
    model_name: str,
    query_texts: Sequence[str],
    tool_texts: Sequence[str],
    gold_indices: np.ndarray,
    args: argparse.Namespace,
    query_prefix: str = "",
    trust_remote_code: bool = False,
) -> Tuple[Dict[str, float], np.ndarray]:
    encoded_queries = [query_prefix + text for text in query_texts]

    query_embeddings = encode_texts(
        texts=encoded_queries,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tool_embeddings = encode_texts(
        texts=tool_texts,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        trust_remote_code=trust_remote_code,
    )

    score_matrix = (query_embeddings @ tool_embeddings.T).cpu().numpy()
    return evaluate_rankings(score_matrix, gold_indices)


def save_summary(
    output_path: str,
    results: Dict[str, Dict[str, float]],
    num_queries: int,
    num_tools: int,
) -> None:
    payload = {
        "num_queries": num_queries,
        "num_tools": num_tools,
        "metrics": results,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved metrics to {output_path}")


def main() -> int:
    args = parse_args()
    args.device = resolve_device(args.device)

    queries, tool_ids, tool_texts, tool_index = prepare_dataset(
        queries_path=args.queries_path,
        tools_path=args.tools_path,
    )

    query_texts = [sample["text"] for sample in queries]
    gold_indices = np.array([tool_index[sample["gold_tool_name"]] for sample in queries])

    print(f"Queries: {len(queries)}")
    print(f"Tools: {len(tool_ids)}")
    print(f"Device: {args.device}")

    results: Dict[str, Dict[str, float]] = {}

    bm25_metrics, _ = run_bm25(
        queries=queries,
        tool_texts=tool_texts,
        tool_index=tool_index,
    )
    results["BM25"] = bm25_metrics

    dense_jobs = [
        (
            "msmarco-MiniLM",
            args.msmarco_model,
            "",
            False,
        ),
        (
            "UAE-large-v1",
            args.uae_model,
            "Represent this sentence for searching relevant passages: ",
            args.uae_trust_remote_code,
        ),
    ]

    for method_name, model_name, query_prefix, trust_remote_code in dense_jobs:
        try:
            metrics, _ = run_dense(
                model_name=model_name,
                query_texts=query_texts,
                tool_texts=tool_texts,
                gold_indices=gold_indices,
                args=args,
                query_prefix=query_prefix,
                trust_remote_code=trust_remote_code,
            )
            results[method_name] = metrics
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            print(f"\n{method_name} failed: {exc}")

    print_metrics_table(results)
    save_summary(
        output_path=args.output_json,
        results=results,
        num_queries=len(queries),
        num_tools=len(tool_ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
