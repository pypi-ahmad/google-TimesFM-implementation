"""Check internal Markdown links and banned domains (no network).

This is a fast hygiene gate for a learning repo: broken relative links are
beginner-hostile and tend to rot silently.

It intentionally avoids third-party deps and does not fetch anything over
the network. It checks:
- relative links resolve to files on disk
- anchors (e.g. file.md#section) point to an existing heading when possible
- banned domains are not referenced (e.g. internal Google-only URLs)

Run:
    uv run python scripts/check_markdown_links.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BANNED_SUBSTRINGS = (
    "pantheon.corp.google.com",
)

# Markdown inline links: [text](target)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _github_slugify(heading: str) -> str:
    """Best-effort GitHub-style heading slug for local anchor checks."""
    s = heading.strip().lower()
    s = re.sub(r"[`*_]+", "", s)
    s = re.sub(r"[^a-z0-9\-\s]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def _collect_heading_anchors(md_text: str) -> set[str]:
    anchors: set[str] = set()
    for line in md_text.splitlines():
        if not line.startswith("#"):
            continue
        m = re.match(r"^#+\s+(.*)$", line)
        if not m:
            continue
        anchors.add(_github_slugify(m.group(1)))
    return anchors


def _iter_markdown_files(project_root: Path) -> Iterable[Path]:
    globs = (
        "README.md",
        "HANDBOOK.md",
        "docs/**/*.md",
        "examples/**/*.md",
        "exercises/**/*.md",
    )
    for pat in globs:
        yield from project_root.glob(pat)


@dataclass(frozen=True)
class LinkIssue:
    file: Path
    target: str
    reason: str


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def _split_anchor(target: str) -> tuple[str, str | None]:
    if "#" not in target:
        return target, None
    path, anchor = target.split("#", 1)
    return path, anchor or None


def check_file(project_root: Path, md_path: Path) -> list[LinkIssue]:
    issues: list[LinkIssue] = []
    text = md_path.read_text(encoding="utf-8")

    for banned in BANNED_SUBSTRINGS:
        if banned in text:
            issues.append(LinkIssue(md_path, banned, "banned_domain"))

    heading_anchors = _collect_heading_anchors(text)

    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip()
        if not target or _is_external(target):
            continue
        if target.startswith("#"):
            anchor = target[1:]
            if anchor and anchor not in heading_anchors:
                issues.append(LinkIssue(md_path, target, "missing_anchor_in_same_file"))
            continue
        if target.startswith(("file://", "vscode://")):
            issues.append(LinkIssue(md_path, target, "disallowed_uri_scheme"))
            continue

        rel_path, anchor = _split_anchor(target)
        if not rel_path:
            continue

        resolved = (md_path.parent / rel_path).resolve()
        try:
            resolved.relative_to(project_root.resolve())
        except ValueError:
            issues.append(LinkIssue(md_path, target, "link_escapes_project_root"))
            continue

        if not resolved.exists():
            issues.append(LinkIssue(md_path, target, "missing_target_file"))
            continue

        if anchor:
            if resolved.suffix.lower() != ".md":
                # We only anchor-check Markdown targets.
                continue
            try:
                other_text = resolved.read_text(encoding="utf-8")
            except OSError:
                continue
            other_anchors = _collect_heading_anchors(other_text)
            if anchor not in other_anchors:
                issues.append(LinkIssue(md_path, target, "missing_anchor_in_target_file"))

    return issues


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    issues: list[LinkIssue] = []

    md_files = sorted(set(_iter_markdown_files(project_root)))
    for p in md_files:
        issues.extend(check_file(project_root=project_root, md_path=p))

    if issues:
        for it in issues:
            rel = it.file.relative_to(project_root)
            print(f"{rel}: {it.reason}: {it.target}")
        raise SystemExit(1)

    print(f"OK: checked {len(md_files)} markdown files")


if __name__ == "__main__":
    main()
