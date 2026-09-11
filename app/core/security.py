import re
from typing import Tuple
from app.core.logging import logger

# Disallowed SQL patterns for safety guardrails
DANGEROUS_SQL_PATTERNS = [
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE)\b",
    r"\b(EXEC|EXECUTE|XP_|SP_)\b",
    r";\s*--",
    r";\s*(DROP|DELETE|INSERT|UPDATE)",
]


def validate_read_only_sql(query: str) -> Tuple[bool, str]:
    """
    Strict security check to ensure generated SQL is read-only.
    Blocks any data modification, DDL, or command execution.
    """
    clean_query = query.strip()
    
    # Remove leading comments or markdown code fences if present
    clean_query = re.sub(r"^```(sql)?", "", clean_query, flags=re.IGNORECASE).strip()
    clean_query = re.sub(r"```$", "", clean_query).strip()
    
    if not clean_query:
        return False, "Query is empty."
    
    # Check if query starts with SELECT or WITH (for CTEs) or EXPLAIN
    if not re.match(r"^(SELECT|WITH|EXPLAIN)\b", clean_query, re.IGNORECASE):
        logger.warning(f"Blocked non-SELECT query: {clean_query[:100]}")
        return False, "Security violation: Only read-only SELECT and WITH statements are permitted."
    
    # Check for dangerous DDL/DML keywords
    for pattern in DANGEROUS_SQL_PATTERNS:
        match = re.search(pattern, clean_query, re.IGNORECASE)
        if match:
            matched_kw = match.group(0)
            logger.warning(f"Dangerous SQL keyword detected: {matched_kw} in query: {clean_query[:100]}")
            return False, f"Security violation: Statement contains prohibited keyword or construct '{matched_kw}'."
            
    return True, clean_query


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames to prevent path traversal."""
    # Strip directory components and dangerous characters
    safe_name = re.sub(r"[^\w\.\-]", "_", filename)
    return safe_name
