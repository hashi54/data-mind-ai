import re
from typing import List, Dict, Any
from pathlib import Path


class DocumentChunker:
    """Chunks structured text and markdown documents into semantically coherent segments."""

    @staticmethod
    def chunk_markdown(
        text: str,
        metadata: Dict[str, Any],
        max_chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        # Split primarily by markdown headers or double newlines
        sections = re.split(r"(?=\n##\s+)", text)
        chunks = []
        chunk_idx = 1

        for sec in sections:
            clean_sec = sec.strip()
            if not clean_sec:
                continue

            if len(clean_sec) <= max_chunk_size:
                chunks.append({
                    "chunk_id": f"{metadata.get('document_name', 'doc')}_c{chunk_idx}",
                    "content": clean_sec,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "char_length": len(clean_sec),
                    },
                })
                chunk_idx += 1
            else:
                # Sub-chunk by paragraphs
                paragraphs = clean_sec.split("\n\n")
                curr_buffer = ""
                for p in paragraphs:
                    if len(curr_buffer) + len(p) <= max_chunk_size:
                        curr_buffer += (p + "\n\n")
                    else:
                        if curr_buffer.strip():
                            chunks.append({
                                "chunk_id": f"{metadata.get('document_name', 'doc')}_c{chunk_idx}",
                                "content": curr_buffer.strip(),
                                "metadata": {
                                    **metadata,
                                    "chunk_index": chunk_idx,
                                    "char_length": len(curr_buffer.strip()),
                                },
                            })
                            chunk_idx += 1
                        curr_buffer = p + "\n\n"
                
                if curr_buffer.strip():
                    chunks.append({
                        "chunk_id": f"{metadata.get('document_name', 'doc')}_c{chunk_idx}",
                        "content": curr_buffer.strip(),
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_idx,
                            "char_length": len(curr_buffer.strip()),
                        },
                    })
                    chunk_idx += 1

        return chunks
