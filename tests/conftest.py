"""Test-suite-wide setup.

**The suite is pinned to the offline provider, regardless of the developer's shell.**

``default_config()`` reads ``AUTOPSY_PROVIDER`` from the environment, and several test
modules call it at import time. Someone with ``AUTOPSY_PROVIDER=groq`` exported — which
is the normal state while working on the live path — would otherwise have `pytest` make
real API calls: slow, quota-burning, network-dependent, and non-deterministic. A test
suite whose results depend on which variables happen to be exported is not a test suite.

This runs before any test module is imported, which is the only point early enough to
matter.
"""

from __future__ import annotations

import os

os.environ["AUTOPSY_PROVIDER"] = "offline"
os.environ["AUTOPSY_EMBEDDER"] = "concept"
# Belt and braces: if a code path ever reaches a live client despite the above, it
# should fail loudly on a missing key rather than quietly spending someone's quota.
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(key, None)

# The offline provider's defaults are tuned for concept-coverage scores, not for
# whatever the ambient provider would have selected.
os.environ.pop("AUTOPSY_GATE_THRESHOLD", None)
os.environ.pop("AUTOPSY_GATE_READS", None)
os.environ.pop("AUTOPSY_EMBED_MODEL", None)

# Never let the suite touch the shared vector store. Embedded Qdrant holds an exclusive
# file lock, so with the API server running `pytest` would fail on a lock rather than on
# anything under test — and a test run that depends on whether a server happens to be up
# is not a test run.
os.environ["AUTOPSY_VECTOR_BACKEND"] = "local"


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config():
    """The offline config, resolved once."""
    from autopsy.config import default_config

    return default_config()


@pytest.fixture(scope="session")
def tiny_index(tiny_config):
    """A two-tenant in-memory corpus.

    Built here rather than read from ``corpus/index``, which would make these tests
    depend on whichever provider the developer last ran ``ingest`` with — a dense index
    against a concept-embedder run is a vector-kind mismatch, and unrelated files start
    failing for reasons that have nothing to do with the code under test.
    """
    from autopsy.ingest import Document, build_index

    docs = [
        Document(
            tenant_id="tenant_kelvin",
            doc_id="errors.md",
            markdown=(
                "# Kelvin errors\n\n"
                "## KLV-4021\n\nKLV-4021 means the pellshale widget threshold was "
                "exhausted while accepting incoming batches. The operation is abandoned "
                "and retried at the next scheduled attempt.\n\n"
                "## KLV-4022\n\nKLV-4022 means the murrvale widget threshold was "
                "exhausted while merging segments, which is a distinct fault with a "
                "different remedy.\n"
            ),
        ),
        Document(
            tenant_id="tenant_atlas",
            doc_id="errors.md",
            markdown=(
                "# Atlas errors\n\n"
                "## ATS-4021\n\nATS-4021 means the corrindale widget threshold was "
                "exceeded during checkpointing, and the checkpoint is retried on the "
                "next interval.\n"
            ),
        ),
        Document(
            tenant_id="tenant_global",
            doc_id="policy.md",
            markdown=(
                "# Shared policy\n\n## Support\n\nEvery tenant may read this shared "
                "policy document, which describes the escalation path and applies "
                "uniformly across the catalogue.\n"
            ),
        ),
    ]
    index, _ = build_index(docs, cfg=tiny_config, label="fixture")
    return index
