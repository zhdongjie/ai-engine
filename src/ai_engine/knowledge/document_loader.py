import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import DefaultDict, List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.knowledge.processors.factory import get_processor


def _build_header_path(metadata: dict) -> str:
    headers = [
        metadata.get("Header 1", "").strip(),
        metadata.get("Header 2", "").strip(),
        metadata.get("Header 3", "").strip(),
    ]
    return " > ".join(header for header in headers if header)


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.SEMANTIC_CHUNK_SIZE,
        chunk_overlap=settings.SEMANTIC_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\u3002", "\uff1f", "\uff01", ".", "!", "?", ";", "\uff1b", " ", ""],
    )


def _split_into_semantic_units(text: str, fallback_splitter: RecursiveCharacterTextSplitter) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= settings.SEMANTIC_CHUNK_SIZE:
            units.append(paragraph)
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[\u3002\uff01\uff1f.!?])\s+", paragraph)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [paragraph]

        sentence_buffer = ""
        for sentence in sentences:
            candidate = sentence if not sentence_buffer else f"{sentence_buffer} {sentence}"
            if len(candidate) <= settings.SEMANTIC_CHUNK_SIZE:
                sentence_buffer = candidate
                continue

            if sentence_buffer:
                units.append(sentence_buffer)
                sentence_buffer = ""

            if len(sentence) <= settings.SEMANTIC_CHUNK_SIZE:
                sentence_buffer = sentence
                continue

            units.extend(chunk.strip() for chunk in fallback_splitter.split_text(sentence) if chunk.strip())

        if sentence_buffer:
            units.append(sentence_buffer)

    return units


def _get_overlap_tail(text: str, max_length: int) -> str:
    if max_length <= 0 or len(text) <= max_length:
        return text.strip()

    tail = text[-max_length:].strip()
    sentence_parts = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\u3002\uff01\uff1f.!?])\s+", tail)
        if sentence.strip()
    ]
    if not sentence_parts:
        return tail
    return " ".join(sentence_parts[-2:]).strip()


def _semantic_split_documents(docs: List[Document], fallback_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    semantic_docs: List[Document] = []

    for doc in docs:
        base_metadata = deepcopy(doc.metadata)
        text = doc.page_content.strip()
        if not text:
            continue

        units = _split_into_semantic_units(text, fallback_splitter)
        if not units:
            continue

        local_docs: List[Document] = []
        current_text = ""
        for unit in units:
            candidate = unit if not current_text else f"{current_text}\n\n{unit}"
            if len(candidate) <= settings.SEMANTIC_CHUNK_SIZE:
                current_text = candidate
                continue

            if current_text:
                local_docs.append(Document(page_content=current_text, metadata=deepcopy(base_metadata)))
                overlap_text = _get_overlap_tail(current_text, settings.SEMANTIC_CHUNK_OVERLAP)
                current_text = f"{overlap_text}\n\n{unit}".strip() if overlap_text else unit
            else:
                local_docs.append(Document(page_content=unit, metadata=deepcopy(base_metadata)))
                current_text = ""

            if len(current_text) > settings.SEMANTIC_CHUNK_SIZE:
                oversized_chunks = [
                    chunk.strip()
                    for chunk in fallback_splitter.split_text(current_text)
                    if chunk.strip()
                ]
                for oversized_chunk in oversized_chunks[:-1]:
                    local_docs.append(Document(page_content=oversized_chunk, metadata=deepcopy(base_metadata)))
                current_text = oversized_chunks[-1] if oversized_chunks else ""

        if current_text:
            if local_docs and len(current_text) < settings.SEMANTIC_CHUNK_MIN_SIZE:
                local_docs[-1].page_content = f"{local_docs[-1].page_content}\n\n{current_text}".strip()
            else:
                local_docs.append(Document(page_content=current_text, metadata=deepcopy(base_metadata)))

        semantic_docs.extend(local_docs)

    return semantic_docs


def _split_documents_for_ingestion(
    docs: List[Document],
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> tuple[List[Document], str]:
    if not docs:
        return [], "empty"

    if settings.ENABLE_SEMANTIC_CHUNKING:
        semantic_docs = _semantic_split_documents(docs, fallback_splitter)
        if semantic_docs:
            return semantic_docs, "semantic"

    return fallback_splitter.split_documents(docs), "fixed"


def _enrich_chunk_metadata(
    docs: List[Document],
    biz_type: str,
    file_name: str,
    source_type: str,
    chunk_strategy: str,
) -> None:
    source_key = f"{biz_type}:{file_name}"
    total_chunks = len(docs)

    for index, doc in enumerate(docs):
        header_path = _build_header_path(doc.metadata)
        doc.metadata.update(
            {
                "biz_type": biz_type,
                "file_name": file_name,
                "source_type": source_type,
                "source_key": source_key,
                "chunk_index": index,
                "chunk_total": total_chunks,
                "header_path": header_path,
                "chunk_strategy": chunk_strategy,
            }
        )

        if header_path:
            doc.page_content = f"[Section] {header_path}\n{doc.page_content}"


def _group_documents_by_source(docs: List[Document]) -> DefaultDict[str, List[Document]]:
    grouped_docs: DefaultDict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        source = str(doc.metadata.get("source", "unknown"))
        grouped_docs[source].append(doc)
    return grouped_docs


def _extract_lead_sentence(text: str, max_length: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    normalized = re.sub(r"^\[Section\]\s+[^\n]+\s*", "", normalized).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\u3002\uff01\uff1f.!?])\s+", normalized)
        if sentence.strip()
    ]
    lead_sentence = sentences[0] if sentences else normalized
    return lead_sentence[:max_length].strip()


def _build_augmentation_questions(doc: Document) -> List[str]:
    if len(doc.page_content.strip()) < settings.DOCUMENT_AUGMENTATION_MIN_CHARS:
        return []

    metadata = doc.metadata or {}
    header_path = str(metadata.get("header_path", "")).strip()
    file_name = Path(str(metadata.get("file_name", "unknown"))).stem.strip()
    title = header_path or file_name or "document"
    lead_sentence = _extract_lead_sentence(doc.page_content, settings.DOCUMENT_AUGMENTATION_MAX_CHARS)

    candidates: List[str] = []
    if title:
        candidates.extend(
            [
                f"What is {title}?",
                f"How does {title} work?",
                f"What are the key rules of {title}?",
            ]
        )
    if file_name and header_path and file_name != header_path:
        candidates.append(f"What conditions apply to {header_path} in {file_name}?")
    if lead_sentence:
        candidates.append(lead_sentence)

    questions: List[str] = []
    seen = set()
    for candidate in candidates:
        clean_candidate = candidate.strip()
        if not clean_candidate or clean_candidate in seen:
            continue
        seen.add(clean_candidate)
        questions.append(clean_candidate)
        if len(questions) >= settings.DOCUMENT_AUGMENTATION_MAX_QUESTIONS:
            break

    return questions


def _apply_document_augmentation(docs: List[Document]) -> None:
    if not settings.ENABLE_DOCUMENT_AUGMENTATION:
        return

    for doc in docs:
        questions = _build_augmentation_questions(doc)
        if not questions:
            continue

        doc.metadata["augmentation_questions"] = json.dumps(questions, ensure_ascii=False)
        augmentation_block = "\n".join(f"- {question}" for question in questions)
        doc.page_content = f"{doc.page_content}\n[Augmented Questions]\n{augmentation_block}".strip()


def load_documents() -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = _build_text_splitter()

    all_docs: List[Document] = []
    knowledge_root = Path(settings.knowledge_dir)

    if not knowledge_root.exists():
        logger.error(f"Knowledge directory does not exist: {knowledge_root}")
        return []

    for biz_dir in knowledge_root.iterdir():
        if not biz_dir.is_dir():
            continue

        biz_type = biz_dir.name
        logger.info(f"Parsing business knowledge directory: [{biz_type}]")
        processor = get_processor(biz_type)

        for markdown_path in biz_dir.glob("**/*.md"):
            try:
                content = markdown_path.read_text(encoding="utf-8")

                extracted_meta = {}
                if processor:
                    content, extracted_meta = processor.process(content, markdown_path)

                markdown_splits = markdown_splitter.split_text(content)
                final_splits, chunk_strategy = _split_documents_for_ingestion(markdown_splits, text_splitter)
                _enrich_chunk_metadata(
                    final_splits,
                    biz_type=biz_type,
                    file_name=markdown_path.name,
                    source_type="markdown",
                    chunk_strategy=chunk_strategy,
                )
                for doc in final_splits:
                    doc.metadata.update(extracted_meta)
                _apply_document_augmentation(final_splits)
                all_docs.extend(final_splits)
            except Exception as exc:
                logger.error(f"Failed to parse markdown file {markdown_path}: {exc}")

        try:
            txt_loader = DirectoryLoader(
                str(biz_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            pdf_loader = DirectoryLoader(
                str(biz_dir),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            )

            raw_docs = txt_loader.load() + pdf_loader.load()

            for raw_doc in raw_docs:
                source_path = Path(str(raw_doc.metadata.get("source", "")))
                content = raw_doc.page_content
                extracted_meta = {}
                if processor:
                    content, extracted_meta = processor.process(content, source_path)
                raw_doc.page_content = content
                raw_doc.metadata.update(extracted_meta)

            grouped_docs = _group_documents_by_source(raw_docs)
            for source, source_docs in grouped_docs.items():
                file_name = os.path.basename(source or "unknown")
                split_docs, chunk_strategy = _split_documents_for_ingestion(source_docs, text_splitter)
                _enrich_chunk_metadata(
                    split_docs,
                    biz_type=biz_type,
                    file_name=file_name,
                    source_type="file",
                    chunk_strategy=chunk_strategy,
                )
                _apply_document_augmentation(split_docs)

                for split_doc in split_docs:
                    if split_doc.page_content.strip():
                        all_docs.append(split_doc)
        except Exception as exc:
            logger.error(f"Failed to parse TXT/PDF documents for [{biz_type}]: {exc}")

    return all_docs
