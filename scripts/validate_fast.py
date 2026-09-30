#!/usr/bin/env python3
"""Bounded, dependency-free L0 checks. See tests/README.md for guarantees/limits."""

import argparse
import ast
from pathlib import Path
import re
import sys
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SURFACES = ("src/icgs", "scripts", "tests", "docs", ".agents")
HARNESS_NAMES = {"docs", "scripts", "tests", ".agents", "AGENTS.md"}
AUTHORITY = "docs/decisions/0001-harness-boundary.md"


def files(root, suffix):
    """Only repository-owned surfaces; never scan environments or downloaded data."""
    found = set(root.glob("*" + suffix))
    for directory in SURFACES:
        base = root / directory
        if base.is_dir():
            found.update(base.rglob("*" + suffix))
    return sorted(p for p in found if p.is_file() and not p.is_symlink()
                  and "__pycache__" not in p.parts)


def check_syntax(root):
    errors = []
    if not (root / "src/icgs").is_dir():
        return ["SYNTAX: missing src/icgs/ runtime tree; select the repository root"]
    for path in files(root, ".py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            compile(tree, str(path), "exec")  # Detect e.g. top-level return; never execute.
        except (SyntaxError, UnicodeError, OSError) as error:
            errors.append(f"SYNTAX {path.relative_to(root)}: {error}")
    return errors


def prose(text):
    """Remove fenced examples while retaining line numbers."""
    result, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            result.append("")
        else:
            result.append(line if fence is None else "")
    return "\n".join(result)


def anchors(text):
    counts, result = {}, set()
    for line in prose(text).splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            value = re.sub(r"[^\w\- ]", "", heading.group(1).lower()).replace(" ", "-")
            count = counts.get(value, 0)
            counts[value] = count + 1
            result.add(value if count == 0 else f"{value}-{count}")
    return result


def check_links(root):
    """Check simple Markdown links/images/definitions and HTML src/href paths.

    Not a full CommonMark engine: use simple inline or defined reference links,
    ATX headings, and URL-encoded spaces in maintained documents.
    """
    errors = []
    if not (root / "AGENTS.md").is_file():
        errors.append("LINK: missing AGENTS.md; restore the harness entrypoint")
    for path in files(root, ".md"):
        text = prose(path.read_text(encoding="utf-8"))
        targets = re.findall(r"!?\[[^\]\n]*\]\(\s*<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)", text)
        targets += re.findall(r"^\s*\[[^\]]+\]:\s*<?([^\s>]+)>?", text, re.MULTILINE)
        targets += re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", text)
        definitions = {" ".join(label.lower().split()) for label in
                       re.findall(r"^\s*\[([^\]]+)\]:", text, re.MULTILINE)}
        for label, reference in re.findall(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]", text):
            key = " ".join((reference or label).lower().split())
            if key not in definitions:
                errors.append(f"LINK {path.relative_to(root)}: undefined reference {key!r}; define its target")
        for target in targets:
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            dest = (path.parent / unquote(url.path)).resolve() if url.path else path
            prefix = f"LINK {path.relative_to(root)} -> {target}:"
            if not dest.is_relative_to(root.resolve()):
                errors.append(f"{prefix} outside repository; use a repository-relative link")
            elif not dest.exists():
                errors.append(f"{prefix} target missing; correct the link or restore its owner")
            elif url.fragment and dest.suffix == ".md":
                if unquote(url.fragment) not in anchors(dest.read_text(encoding="utf-8")):
                    errors.append(f"{prefix} heading missing; correct the anchor")
    return errors


def harness_literal(value):
    """Conservative detection of literal path/module targets, not arbitrary prose."""
    if not isinstance(value, str) or any(c.isspace() for c in value):
        return False
    value = value.replace("\\", "/")
    while value.startswith(("./", "../")):
        value = value.split("/", 1)[1]
    return (value in HARNESS_NAMES
            or value.startswith(tuple(name + "/" for name in HARNESS_NAMES))
            or value.startswith(("docs.", "scripts.", "tests.")))


def check_boundary(root):
    errors = []
    if not (root / "src/icgs").is_dir():
        return ["HARNESS_BOUNDARY: missing src/icgs/ runtime tree; select the repository root"]
    paths = sorted((root / "src/icgs").rglob("*.py"))
    if (root / "setup.py").is_file():
        paths.append(root / "setup.py")
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError, OSError) as error:
            errors.append(f"HARNESS_BOUNDARY {path.relative_to(root)}: cannot inspect: {error}")
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [node.module] if node.module else [alias.name for alias in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                targets = [node.value]
            for target in targets:
                if harness_literal(target):
                    errors.append(
                        f"HARNESS_BOUNDARY {path.relative_to(root)}:{node.lineno}: {target!r}; "
                        f"runtime must not depend on harness ({AUTHORITY}); "
                        "keep operational data/code in a runtime-owned location"
                    )
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository to inspect (default: script's repository)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    failed = False
    for label, check in (("Python syntax", check_syntax), ("Local documentation links", check_links),
                         ("Static harness boundary", check_boundary)):
        try:
            errors = check(root)
        except (OSError, UnicodeError, ValueError) as error:
            errors = [f"cannot complete {label}: {error}"]
        print(f"{'FAIL' if errors else 'PASS'}: {label}", flush=True)
        for error in errors:
            print(f"  {error}", flush=True)
        failed |= bool(errors)
    if root == ROOT:
        suite = unittest.TestSuite([
            unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
            for pattern in ("test_harness.py", "test_architecture.py", "test_src_package.py")
        ])
        if not suite.countTestCases():
            print("FAIL: no harness tests discovered")
            failed = True
        else:
            result = unittest.TextTestRunner(verbosity=1).run(suite)
            failed |= not result.wasSuccessful() or bool(result.skipped)
            print(f"{'FAIL' if not result.wasSuccessful() or result.skipped else 'PASS'}: harness self-tests")
    else:
        print("NOT RUN: harness self-tests (--root inspects a fixture; run default command for self-tests)")
    print("NOT RUN: L1 CPU smoke, L2 model, L3 simulator, L4 benchmark (L0 only)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
