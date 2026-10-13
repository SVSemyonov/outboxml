"""Collect and validate release metadata for the PyPI release workflow.

Checks that the version from pyproject.toml is newer than the latest release on PyPI
and not tagged, and extracts its release notes from the changelog.
Writes ``version`` and ``tag`` to ``$GITHUB_OUTPUT`` when run in GitHub Actions.

Usage:
    python .github/scripts/release_metadata.py --notes-out release_notes.md
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import Version


def read_project(pyproject: Path) -> tuple[str, str]:
    with open(pyproject, "rb") as f:
        project = tomllib.load(f)["project"]
    return project["name"], project["version"]


def pypi_versions(name: str) -> list[Version]:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            releases = json.load(response)["releases"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # project is not on PyPI yet
            return []
        raise
    return [Version(v) for v in releases]


def tag_exists(tag: str) -> bool:
    result = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                            capture_output=True)
    return result.returncode == 0


def extract_release_notes(changelog: Path, version: str) -> str:
    """Return the body of the ``## [version]`` section of a Keep a Changelog file."""
    text = changelog.read_text(encoding="utf-8")
    header = re.compile(rf"^## \[{re.escape(version)}\][^\n]*$", re.MULTILINE)
    match = header.search(text)
    if not match:
        return ""
    next_section = re.compile(r"^## \[", re.MULTILINE).search(text, match.end())
    body = text[match.end():next_section.start() if next_section else len(text)]
    return body.strip().strip("-").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("docs/CHANGELOG.md"))
    parser.add_argument("--notes-out", type=Path, default=Path("release_notes.md"))
    args = parser.parse_args()

    name, version = read_project(args.pyproject)
    tag = f"v{version}"
    errors = []

    published = pypi_versions(name)
    latest = max(published, default=None)
    if latest is not None and Version(version) <= latest:
        errors.append(f"{name} {version} is not newer than {latest} on PyPI: bump the version in {args.pyproject}")
    if tag_exists(tag):
        errors.append(f"git tag {tag} already exists: bump the version in {args.pyproject}")

    notes = extract_release_notes(args.changelog, version)
    if not notes:
        errors.append(f"no release notes for [{version}] in {args.changelog}: add a '## [{version}] - YYYY-MM-DD' section")

    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1

    args.notes_out.write_text(notes + "\n", encoding="utf-8")
    print(f"Release {name} {version} ({tag})")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"version={version}\ntag={tag}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
