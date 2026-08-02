import { useState } from "react";

import type { Candidate, StageRecord } from "../lib/trace";

/**
 * Panel A — retrieval competition.
 *
 * Two ranked columns resolving into one fused list. A chunk that ranked 7th lexically
 * and 2nd semantically visibly winning the fusion is the whole point of the panel.
 *
 * **Rank is position; score is text.** BM25 is unbounded (0–20+ here), cosine is 0–1,
 * and RRF lands around 0.03. Those three cannot share an axis, and plotting them
 * together produces a chart that looks quantitative and means nothing. The only
 * quantity all three legs agree on is ordinal position, so position is what encodes
 * rank and the raw numbers are rendered as monospace text beside it.
 */

const PER_LEG = 8;

interface Props {
  lexical: Candidate[];
  semantic: Candidate[];
  fused: Candidate[];
  gate: number | null;
  gateValue: number | null;
  gateReads: string | null;
  rerank: StageRecord | undefined;
  highlighted: string | null;
  onHover: (chunkId: string | null) => void;
}

function label(candidate: Candidate): string {
  const head = candidate.heading_path[candidate.heading_path.length - 1];
  return head ?? candidate.doc_id;
}

function num(value: number | null, digits = 2): string {
  return value === null ? "—" : value.toFixed(digits);
}

function Leg({
  title,
  scale,
  items,
  scoreOf,
  rankOf,
  highlighted,
  onHover,
}: {
  title: string;
  scale: string;
  items: Candidate[];
  scoreOf: (c: Candidate) => number | null;
  rankOf: (c: Candidate) => number | null;
  highlighted: string | null;
  onHover: (id: string | null) => void;
}) {
  const shown = items.slice(0, PER_LEG);
  const hidden = items.length - shown.length;
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-2xs uppercase tracking-widest text-dim">{title}</span>
        <span className="text-2xs text-dim/70">{scale}</span>
      </div>
      <ol className="space-y-px">
        {shown.map((candidate) => (
          <li
            key={candidate.chunk_id}
            onMouseEnter={() => onHover(candidate.chunk_id)}
            onMouseLeave={() => onHover(null)}
            className={`flex gap-2 px-1.5 py-1 cursor-default ${
              highlighted === candidate.chunk_id ? "bg-win/15 text-bright" : "hover:bg-raised"
            }`}
          >
            <span className="tabular text-dim w-4 text-right shrink-0">{rankOf(candidate)}</span>
            <span className="truncate flex-1 min-w-0">{label(candidate)}</span>
            <span className="tabular text-dim shrink-0">{num(scoreOf(candidate))}</span>
          </li>
        ))}
        {items.length === 0 && <li className="px-1.5 py-1 text-dim italic">leg not run</li>}
      </ol>
      {hidden > 0 && (
        <div className="px-1.5 pt-1 text-2xs text-dim">
          +{hidden} more — hidden because forty rows is not a ranking, it is a wall
        </div>
      )}
    </div>
  );
}

export default function Competition({
  lexical,
  semantic,
  fused,
  gate,
  gateValue,
  gateReads,
  rerank,
  highlighted,
  onHover,
}: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const admitted = fused.filter((c) => c.rejected_by !== "gate");
  const gateRejected = fused.filter((c) => c.rejected_by === "gate");
  const shown = admitted.slice(0, 10);
  const hidden = admitted.length - shown.length;

  const top1 = fused[0]?.fused_score ?? null;
  const top2 = fused[1]?.fused_score ?? null;
  const margin = top1 !== null && top2 !== null ? top1 - top2 : null;

  return (
    <section className="panel flex flex-col min-h-0">
      <div className="panel-title flex justify-between">
        <span>A · retrieval competition</span>
        <span className="text-dim/70 normal-case tracking-normal">
          {fused.length} candidates
        </span>
      </div>

      <div className="p-3 grid grid-cols-2 gap-4">
        <Leg
          title="bm25 · lexical"
          scale="unbounded"
          items={lexical}
          scoreOf={(c) => c.lexical_score}
          rankOf={(c) => c.lexical_rank}
          highlighted={highlighted}
          onHover={onHover}
        />
        <Leg
          title="vector · semantic"
          scale="0–1"
          items={semantic}
          scoreOf={(c) => c.semantic_score}
          rankOf={(c) => c.semantic_rank}
          highlighted={highlighted}
          onHover={onHover}
        />
      </div>

      <div className="px-3 text-center text-dim text-2xs select-none">↘ &nbsp; ↙</div>

      <div className="p-3 pt-2 flex-1 min-h-0 overflow-y-auto">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-2xs uppercase tracking-widest text-dim">fused · rrf k=60</span>
          <span className="text-2xs text-dim/70 tabular">
            margin {margin === null ? "—" : margin.toFixed(4)}
          </span>
        </div>

        <ol className="space-y-px">
          {shown.map((candidate) => {
            const isExpansion = candidate.inclusion_reason === "neighbor_expansion";
            const promoted = candidate.inclusion_reason === "rerank_promoted";
            return (
              <li key={candidate.chunk_id}>
                <div
                  onMouseEnter={() => onHover(candidate.chunk_id)}
                  onMouseLeave={() => onHover(null)}
                  onClick={() =>
                    setExpanded(expanded === candidate.chunk_id ? null : candidate.chunk_id)
                  }
                  className={`flex gap-2 px-1.5 py-1 cursor-pointer ${
                    highlighted === candidate.chunk_id ? "bg-win/15" : "hover:bg-raised"
                  } ${candidate.in_context ? "text-bright" : "text-dim"}`}
                >
                  <span className="tabular text-dim w-5 text-right shrink-0">
                    {candidate.fused_rank ?? "·"}
                  </span>
                  <span
                    className={`truncate flex-1 min-w-0 ${
                      isExpansion ? "italic text-body/70" : ""
                    }`}
                  >
                    {label(candidate)}
                  </span>
                  {promoted && <span className="text-win shrink-0">↑rerank</span>}
                  {isExpansion && <span className="text-dim shrink-0">±neighbour</span>}
                  {candidate.rerank_score !== null && (
                    <span className="tabular text-win/80 shrink-0">
                      {candidate.rerank_score.toFixed(0)}
                    </span>
                  )}
                  <span className="tabular text-dim shrink-0 w-14 text-right">
                    {num(candidate.fused_score, 4)}
                  </span>
                </div>
                {expanded === candidate.chunk_id && (
                  <pre className="mx-1.5 my-1 p-2 bg-raised border border-line text-2xs whitespace-pre-wrap text-body/80 max-h-40 overflow-y-auto">
                    {candidate.heading_path.join(" › ")}
                    {"\n\n"}
                    {candidate.text}
                  </pre>
                )}
              </li>
            );
          })}
        </ol>

        {hidden > 0 && (
          <div className="px-1.5 pt-1 text-2xs text-dim">+{hidden} further candidates</div>
        )}

        {gate !== null && (
          <div className="mt-3">
            <div className="rule" />
            <div className="flex justify-between text-2xs text-caution/90 pt-1 px-1.5 tabular">
              <span>
                gate {gate.toFixed(2)} · reads {gateReads ?? "—"}
              </span>
              <span>observed {gateValue === null ? "—" : gateValue.toFixed(3)}</span>
            </div>
          </div>
        )}

        {gateRejected.length > 0 && (
          <ol className="mt-1 space-y-px opacity-60">
            {gateRejected.slice(0, 6).map((candidate) => (
              <li key={candidate.chunk_id} className="flex gap-2 px-1.5 py-1 text-reject">
                <span className="tabular w-5 text-right shrink-0">{candidate.fused_rank}</span>
                <span className="truncate flex-1 min-w-0 line-through">{label(candidate)}</span>
                <span className="shrink-0">rejected · gate</span>
              </li>
            ))}
          </ol>
        )}

        {/* Showing the reranker *declining* is the differentiator. Almost no public
            demo shows a system deciding not to spend money, and the numbers behind
            that decision are more interesting than a reranked list. */}
        {rerank && (
          <div className="mt-3 p-2 bg-raised border border-line text-2xs">
            <span className={rerank.skipped ? "text-dim" : "text-win"}>
              rerank {rerank.skipped ? "declined" : "fired"}
            </span>
            {rerank.skip_reason && <span className="text-dim"> — {rerank.skip_reason}</span>}
            {!rerank.skipped && (
              <span className="text-dim tabular">
                {" "}
                — scored {String((rerank.detail as { scored?: number }).scored ?? "?")} candidates
                {rerank.cost_usd > 0 && ` · $${rerank.cost_usd.toFixed(5)}`}
              </span>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
