"""Pipeline stages. Every one is optional and driven entirely by config."""

from autopsy.stages.base import Context, Stage, State, ensure_candidate, execute, stage_span
from autopsy.stages.embed import EmbedStage
from autopsy.stages.expand import ExpandStage, finalize_context
from autopsy.stages.fuse import FuseStage, FusedEventEmitter, rrf_score
from autopsy.stages.gate import GateStage, read_signal
from autopsy.stages.generate import GenerateStage
from autopsy.stages.rerank import RerankStage
from autopsy.stages.retrieve_dense import RetrieveDenseStage
from autopsy.stages.retrieve_sparse import RetrieveSparseStage
from autopsy.stages.rewrite import RewriteStage

__all__ = [
    "Context",
    "EmbedStage",
    "ExpandStage",
    "FuseStage",
    "FusedEventEmitter",
    "GateStage",
    "GenerateStage",
    "RerankStage",
    "RetrieveDenseStage",
    "RetrieveSparseStage",
    "RewriteStage",
    "Stage",
    "State",
    "ensure_candidate",
    "execute",
    "finalize_context",
    "read_signal",
    "rrf_score",
    "stage_span",
]
