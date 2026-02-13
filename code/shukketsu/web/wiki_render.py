"""Server-side Markdown rendering for wiki articles."""

import markdown  # type: ignore[import-untyped]
import nh3

_MD = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])

# Tags and attributes allowed after sanitization.
# Covers standard Markdown output (headings, lists, code, tables, links).
_ALLOWED_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "br",
    "hr",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "li",
    "a",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "div",
    "span",
    "img",
}

_ALLOWED_ATTRS: dict[str, set[str]] = {
    "a": {"href", "title", "target"},
    "img": {"src", "alt", "title"},
    "th": {"align"},
    "td": {"align"},
    "div": {"class", "id"},
    "span": {"class"},
}


def render_markdown(text: str) -> str:
    """Render Markdown text to sanitized HTML.

    Uses tables, fenced_code, and toc extensions. Output is sanitized
    with nh3 to strip dangerous tags (script, iframe, etc.) since the
    result is injected via Jinja2's ``|safe`` filter.
    """
    if not text:
        return ""
    _MD.reset()
    raw_html: str = _MD.convert(text)
    return nh3.clean(raw_html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS)
