"""Tests for server-side Markdown rendering."""

from code.shukketsu.web.wiki_render import render_markdown


class TestRenderMarkdown:
    def test_headings(self) -> None:
        html = render_markdown("# Title\n\n## Subtitle")
        assert "<h1" in html
        assert "<h2" in html
        assert "Title" in html

    def test_bold_and_italic(self) -> None:
        html = render_markdown("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_tables(self) -> None:
        md = "| Head |\n|------|\n| Cell |"
        html = render_markdown(md)
        assert "<table>" in html
        assert "<th>" in html
        assert "Cell" in html

    def test_fenced_code(self) -> None:
        md = "```python\nprint('hello')\n```"
        html = render_markdown(md)
        assert "<pre>" in html
        assert "<code" in html
        assert "print" in html

    def test_empty_string(self) -> None:
        assert render_markdown("") == ""

    def test_multiple_calls_independent(self) -> None:
        """State shouldn't leak between calls (reset works)."""
        html1 = render_markdown("# First")
        html2 = render_markdown("# Second")
        assert "First" in html1
        assert "Second" in html2
        assert "First" not in html2

    def test_xss_script_tag_stripped(self) -> None:
        """Inline <script> tags are stripped by nh3 sanitization."""
        html = render_markdown("Hello <script>alert('xss')</script> world")
        assert "<script>" not in html
        assert "alert" not in html
        assert "Hello" in html

    def test_xss_iframe_stripped(self) -> None:
        """Iframe injection is stripped."""
        html = render_markdown('<iframe src="http://evil.com"></iframe>')
        assert "<iframe" not in html

    def test_xss_event_handler_stripped(self) -> None:
        """Event handler attributes are stripped from allowed tags."""
        html = render_markdown('<a href="#" onclick="alert(1)">click</a>')
        assert "onclick" not in html
        assert "click" in html

    def test_safe_link_preserved(self) -> None:
        """Normal markdown links are preserved after sanitization."""
        html = render_markdown("[Wowhead](https://www.wowhead.com)")
        assert "href" in html
        assert "wowhead.com" in html
