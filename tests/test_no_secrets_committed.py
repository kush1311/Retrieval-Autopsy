"""No file that git would commit may contain a credential.

This is the test that would have caught the actual incident. Two live API keys sat in
`.env` on disk while `.gitignore` did cover `.env` — so nothing was ever exposed through
git, and precisely nothing in the repository verified that. The protection was one edit
away from vanishing silently.

Deliberately checks the *ignore rules* rather than the filesystem: the risk is not "is
there a secret on this machine" (there is, in `.env`, correctly) but "would a secret reach
a commit". Those are different questions and only the second one matters here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Prefixes for the providers this project actually talks to. Anchored and length-bounded
#: rather than a generic "high entropy string" heuristic, which fires on every content
#: hash, chunk ID and trace ID in the corpus — of which there are thousands here.
SECRET_PATTERNS = {
    "groq": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "openai-project": re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    "openai-legacy": re.compile(r"sk-[A-Za-z0-9]{32,}"),
    "anthropic": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "aws": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    ).stdout


@pytest.fixture(scope="module")
def committable_files() -> list[Path]:
    """Everything git would include: tracked, plus untracked-and-not-ignored."""
    tracked = _git("ls-files").splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    if not tracked and not untracked:
        pytest.skip("not a git checkout")
    return [REPO_ROOT / p for p in {*tracked, *untracked} if p.strip()]


def test_no_credentials_in_anything_git_would_commit(committable_files) -> None:
    found: list[str] = []
    for path in committable_files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            match = pattern.search(text)
            if match:
                rel = path.relative_to(REPO_ROOT).as_posix()
                # Report the prefix only. A test failure message ends up in CI logs, and
                # printing the whole key would leak it into a second place.
                found.append(f"{rel}: {label} key starting {match.group(0)[:12]}…")
    assert not found, "credentials would be committed:\n  " + "\n  ".join(found)


def test_dotenv_is_ignored_and_the_example_is_not() -> None:
    """Both halves matter. `.env` ignored keeps secrets out; `.env.example` *not* ignored
    is what makes the project usable — an ignored template is an absent template."""
    ignored = _git("check-ignore", "-v", ".env", ".env.local", ".env.backup")
    for name in (".env", ".env.local", ".env.backup"):
        assert name in ignored, f"{name} is NOT gitignored"

    example = _git("check-ignore", "-v", ".env.example")
    assert not example.strip(), (
        ".env.example is gitignored, so a fresh clone has no template to copy"
    )


def test_the_example_template_has_empty_credential_values() -> None:
    """A template with a working key in it is not a template, it is a leak with
    documentation."""
    path = REPO_ROOT / ".env.example"
    offenders = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip().endswith(("_KEY", "_TOKEN", "_SECRET")) and value.strip():
            offenders.append(f"line {i}: {name.strip()} has a value")
    assert not offenders, ".env.example must ship empty credentials: " + "; ".join(offenders)
