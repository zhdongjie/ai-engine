import argparse
import asyncio
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

INVALID_PROXY_TARGETS = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "http://localhost:9",
    "https://localhost:9",
}


def _sanitize_invalid_proxy_env() -> None:
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        proxy_value = os.environ.get(proxy_key)
        if proxy_value and proxy_value.strip().lower() in INVALID_PROXY_TARGETS:
            os.environ.pop(proxy_key, None)


_sanitize_invalid_proxy_env()

from ai_engine.chains.common.query_transformer import transform_queries
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.infra.db.knowledge_corpus import knowledge_corpus
from ai_engine.utils.retrieval_utils import (
    assess_retrieval_quality,
    collect_candidate_documents,
    compress_context_documents,
    extract_relevant_segments,
    get_reranked_docs,
    resolve_retrieval_runtime_config,
    select_top_documents,
    summarize_retrieved_documents,
)


def _load_cases(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    cases: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "query" not in payload:
                raise ValueError(f"Missing `query` in dataset line {line_number}")
            cases.append(payload)
    return cases


def _load_report(report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"Baseline report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_expected_values(case: Dict[str, Any], field_name: str) -> set[str]:
    return {
        str(item).strip().lower()
        for item in case.get(field_name, [])
        if str(item).strip()
    }


def _is_case_failed(result: Dict[str, Any]) -> bool:
    retrieval_quality = result.get("retrieval_quality", {})
    if not retrieval_quality.get("is_confident", False):
        return True

    for field_name in ("source_hit", "header_hit", "source_key_hit"):
        field_value = result.get(field_name)
        if field_value is False:
            return True

    return False


def _safe_average(values: List[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


async def _run_case(case: Dict[str, Any], diagnostics_limit: int) -> Dict[str, Any]:
    biz_type = case.get("biz_type", "normal_chat")
    user_lang = case.get("lang", "zh")
    prompt_data = get_prompt_config(biz_type)
    runtime_config = resolve_retrieval_runtime_config(prompt_data.get("retrieval_config", {}))

    queries = [case["query"]]
    if runtime_config["enable_query_transform"]:
        queries = transform_queries(user_input=case["query"], history=case.get("history", []), config={})

    candidate_docs = await collect_candidate_documents(
        queries=queries,
        search_k=runtime_config["search_k"],
        lexical_k=runtime_config["lexical_k"],
        user_lang=user_lang,
        enable_lexical_retrieval=runtime_config["enable_lexical_retrieval"],
    )

    reranked_docs = await asyncio.to_thread(get_reranked_docs, case["query"], candidate_docs)
    retrieval_quality = assess_retrieval_quality(reranked_docs)

    anchor_docs = reranked_docs
    if runtime_config["enable_context_compression"] and anchor_docs:
        anchor_docs = select_top_documents(anchor_docs, runtime_config["max_context_chunks"])

    parent_context_used = False
    if runtime_config["enable_small_to_big_retrieval"] and anchor_docs:
        expanded_docs = knowledge_corpus.expand_to_parent_context(
            anchor_docs,
            max_parent_chunks=runtime_config["small_to_big_max_parent_chunks"],
            fallback_window_size=runtime_config["small_to_big_fallback_window_size"],
        )
        parent_context_used = bool(expanded_docs)
    elif runtime_config["enable_context_enrichment"] and anchor_docs:
        expanded_docs = knowledge_corpus.expand_with_neighbors(anchor_docs, runtime_config["context_window_size"])
    else:
        expanded_docs = anchor_docs

    rse_summary = {
        "segment_count": 0,
        "retained_doc_count": len(expanded_docs),
        "dropped_doc_count": 0,
        "selected_segment_scores": [],
        "applied": False,
    }
    final_docs = expanded_docs
    if runtime_config["enable_relevant_segment_extraction"] and final_docs:
        final_docs, rse_summary = extract_relevant_segments(
            final_docs,
            similarity_threshold=runtime_config["rse_similarity_threshold"],
            segment_score_threshold=runtime_config["rse_segment_score_threshold"],
            window_size=runtime_config["rse_window_size"],
            max_segments=runtime_config["rse_max_segments"],
        )

    if runtime_config["enable_context_compression"] and final_docs:
        final_docs = compress_context_documents(
            final_docs,
            max_chunks=runtime_config["max_context_chunks"],
            max_characters=runtime_config["max_context_characters"],
        )

    expected_sources = _normalize_expected_values(case, "expected_sources")
    expected_headers = _normalize_expected_values(case, "expected_headers")
    expected_source_keys = _normalize_expected_values(case, "expected_source_keys")

    actual_sources = {
        str(doc.metadata.get("file_name", "")).lower()
        for doc in final_docs
    }
    actual_headers = {
        str(doc.metadata.get("header_path", "")).lower()
        for doc in final_docs
        if doc.metadata.get("header_path")
    }
    actual_source_keys = {
        str(doc.metadata.get("source_key", "")).lower()
        for doc in final_docs
    }

    source_hits = len(expected_sources & actual_sources)
    header_hits = len(expected_headers & actual_headers)
    source_key_hits = len(expected_source_keys & actual_source_keys)

    return {
        "case_id": str(case.get("case_id", "")),
        "query": case["query"],
        "biz_type": biz_type,
        "lang": user_lang,
        "tags": [str(tag) for tag in case.get("tags", []) if str(tag).strip()],
        "queries": queries,
        "retrieval_quality": retrieval_quality,
        "parent_context_used": parent_context_used,
        "rse_summary": rse_summary,
        "expected_sources": sorted(expected_sources),
        "expected_headers": sorted(expected_headers),
        "expected_source_keys": sorted(expected_source_keys),
        "actual_sources": sorted(actual_sources),
        "actual_headers": sorted(actual_headers),
        "actual_source_keys": sorted(actual_source_keys),
        "source_hit": source_hits > 0 if expected_sources else None,
        "header_hit": header_hits > 0 if expected_headers else None,
        "source_key_hit": source_key_hits > 0 if expected_source_keys else None,
        "source_recall": (source_hits / len(expected_sources)) if expected_sources else None,
        "header_recall": (header_hits / len(expected_headers)) if expected_headers else None,
        "source_key_recall": (source_key_hits / len(expected_source_keys)) if expected_source_keys else None,
        "final_doc_count": len(final_docs),
        "failed": False,
        "top_docs": summarize_retrieved_documents(final_docs, diagnostics_limit),
    }


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(results)
    source_cases = [item for item in results if item["source_hit"] is not None]
    header_cases = [item for item in results if item["header_hit"] is not None]
    source_key_cases = [item for item in results if item["source_key_hit"] is not None]
    failed_cases = [item for item in results if item.get("failed", False)]
    confident_cases = [item for item in results if item.get("retrieval_quality", {}).get("is_confident", False)]
    weak_cases = [item for item in results if not item.get("retrieval_quality", {}).get("is_confident", False)]

    return {
        "total_cases": total_cases,
        "failed_cases": len(failed_cases),
        "failure_rate": (len(failed_cases) / total_cases) if total_cases else 0,
        "confident_rate": (len(confident_cases) / total_cases) if total_cases else 0,
        "weak_retrieval_rate": (len(weak_cases) / total_cases) if total_cases else 0,
        "source_hit_rate": (
            sum(1 for item in source_cases if item["source_hit"]) / len(source_cases)
            if source_cases else None
        ),
        "header_hit_rate": (
            sum(1 for item in header_cases if item["header_hit"]) / len(header_cases)
            if header_cases else None
        ),
        "source_key_hit_rate": (
            sum(1 for item in source_key_cases if item["source_key_hit"]) / len(source_key_cases)
            if source_key_cases else None
        ),
        "avg_source_recall": (
            _safe_average([item["source_recall"] for item in source_cases])
        ),
        "avg_header_recall": (
            _safe_average([item["header_recall"] for item in header_cases])
        ),
        "avg_source_key_recall": (
            _safe_average([item["source_key_recall"] for item in source_key_cases])
        ),
        "avg_final_doc_count": (
            sum(item["final_doc_count"] for item in results) / total_cases
            if total_cases else 0
        ),
        "avg_query_count": (
            sum(len(item["queries"]) for item in results) / total_cases
            if total_cases else 0
        ),
        "avg_top_rerank_score": (
            _safe_average([item.get("retrieval_quality", {}).get("top_score") for item in results])
        ),
        "avg_score_gap": (
            _safe_average([item.get("retrieval_quality", {}).get("score_gap") for item in results])
        ),
        "parent_context_usage_rate": (
            sum(1 for item in results if item.get("parent_context_used")) / total_cases
            if total_cases else 0
        ),
        "rse_applied_rate": (
            sum(1 for item in results if item.get("rse_summary", {}).get("applied")) / total_cases
            if total_cases else 0
        ),
    }


def _build_group_summary(results: List[Dict[str, Any]], field_name: str) -> Dict[str, Dict[str, Any]]:
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        raw_value = result.get(field_name)
        group_key = str(raw_value).strip() if raw_value not in (None, "") else "unknown"
        grouped_results.setdefault(group_key, []).append(result)

    return {
        group_key: _build_summary(group_items)
        for group_key, group_items in sorted(grouped_results.items())
    }


def _collect_failures(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = []
    for result in results:
        if not result.get("failed", False):
            continue
        failure_item = copy.deepcopy(result)
        failures.append(failure_item)
    return failures


def _build_summary_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key, current_value in current.items():
        baseline_value = baseline.get(key)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[key] = round(current_value - baseline_value, 6)
    return deltas


def _build_comparison(current_report: Dict[str, Any], baseline_report: Dict[str, Any]) -> Dict[str, Any]:
    current_summary = current_report.get("summary", {})
    baseline_summary = baseline_report.get("summary", {})

    return {
        "baseline_report": baseline_report.get("dataset", ""),
        "current_total_cases": current_summary.get("total_cases", 0),
        "baseline_total_cases": baseline_summary.get("total_cases", 0),
        "summary_delta": _build_summary_delta(current_summary, baseline_summary),
        "group_delta": {
            "by_biz_type": {
                group_key: _build_summary_delta(
                    current_report.get("groups", {}).get("by_biz_type", {}).get(group_key, {}),
                    baseline_report.get("groups", {}).get("by_biz_type", {}).get(group_key, {}),
                )
                for group_key in sorted(
                    set(current_report.get("groups", {}).get("by_biz_type", {}).keys())
                    | set(baseline_report.get("groups", {}).get("by_biz_type", {}).keys())
                )
            },
            "by_lang": {
                group_key: _build_summary_delta(
                    current_report.get("groups", {}).get("by_lang", {}).get(group_key, {}),
                    baseline_report.get("groups", {}).get("by_lang", {}).get(group_key, {}),
                )
                for group_key in sorted(
                    set(current_report.get("groups", {}).get("by_lang", {}).keys())
                    | set(baseline_report.get("groups", {}).get("by_lang", {}).keys())
                )
            },
        },
    }


async def _run(
    dataset_path: Path,
    output_path: Path | None,
    diagnostics_limit: int,
    failures_output_path: Path | None,
    compare_path: Path | None,
) -> None:
    db_manager.init_db()
    cases = _load_cases(dataset_path)
    try:
        results = []
        for case in cases:
            case_result = await _run_case(case, diagnostics_limit)
            case_result["failed"] = _is_case_failed(case_result)
            results.append(case_result)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "summary": _build_summary(results),
            "groups": {
                "by_biz_type": _build_group_summary(results, "biz_type"),
                "by_lang": _build_group_summary(results, "lang"),
            },
            "failures": _collect_failures(results),
            "results": results,
        }

        if compare_path is not None:
            baseline_report = _load_report(compare_path)
            report["comparison"] = _build_comparison(report, baseline_report)

        report_text = json.dumps(report, ensure_ascii=False, indent=2)
        if output_path is not None:
            output_path.write_text(report_text, encoding="utf-8")

        if failures_output_path is not None:
            failures_output_path.write_text(
                json.dumps(report["failures"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(report_text)
    finally:
        db_manager.close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current retrieval stack with a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the JSONL evaluation dataset")
    parser.add_argument("--output", help="Optional path to write the evaluation report")
    parser.add_argument("--failures-output", help="Optional path to write failed-case diagnostics")
    parser.add_argument("--compare", help="Optional baseline report path used for summary comparison")
    parser.add_argument("--diagnostics-limit", type=int, default=5, help="Number of retrieved chunks kept per case")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    failures_output_path = Path(args.failures_output).resolve() if args.failures_output else None
    compare_path = Path(args.compare).resolve() if args.compare else None
    asyncio.run(
        _run(
            dataset_path,
            output_path,
            args.diagnostics_limit,
            failures_output_path,
            compare_path,
        )
    )


if __name__ == "__main__":
    main()
