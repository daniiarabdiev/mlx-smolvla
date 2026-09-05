"""Keep the PyPI README usable without a repository-relative base URL."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "daniiarabdiev/mlx-smolvla"
RELEASE_TAG = "v0.1.2"


def heading_slugs(markdown):
    """GitHub-style anchors for ATX headings, excluding fenced examples."""
    markdown = re.sub(r"(?ms)^```.*?^```[^\n]*$", "", markdown)
    seen = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", markdown):
        heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\-\s]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        yield f"{slug}-{count}" if count else slug


def readme_links():
    markdown = (ROOT / "README.md").read_text()
    inline = re.findall(r"\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)", markdown)
    references = re.findall(r"(?m)^\s*\[[^]]+\]:\s*<?([^\s>]+)", markdown)
    html = re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", markdown)
    return inline + references + html


def test_readme_links_are_absolute_and_release_sources_exist():
    links = readme_links()
    assert links, "README link extraction unexpectedly found no links"
    for target in links:
        url = urlsplit(target)
        assert url.scheme in {"https", "http", "mailto"}, f"Relative README link: {target}"
        source_path = None
        if url.netloc == "github.com" and url.path.startswith(f"/{REPOSITORY}/"):
            parts = url.path.split("/")
            if parts[3] in {"blob", "tree"}:
                assert parts[4] == RELEASE_TAG, f"Unpinned release source: {target}"
                source_path = "/".join(parts[5:])
        elif url.netloc == "raw.githubusercontent.com" and url.path.startswith(f"/{REPOSITORY}/"):
            parts = url.path.split("/")
            assert parts[3] == RELEASE_TAG, f"Unpinned release image: {target}"
            source_path = "/".join(parts[4:])
        if source_path is not None:
            local = ROOT / unquote(source_path)
            assert local.exists(), f"Missing release source: {target}"
            if url.fragment:
                assert local.is_file(), f"Anchor does not target a file: {target}"
                assert unquote(url.fragment) in set(heading_slugs(local.read_text())), target
