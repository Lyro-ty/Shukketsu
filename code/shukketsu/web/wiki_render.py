"""Server-side Markdown rendering for wiki articles."""

import markdown

_MD = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])


def render_markdown(text: str) -> str:
    """Render Markdown text to HTML.

    Uses tables, fenced_code, and toc extensions. Resets internal state
    between calls to prevent leakage.
    """
    if not text:
        return ""
    _MD.reset()
    return _MD.convert(text)
