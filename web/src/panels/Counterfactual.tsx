import type { Trace } from "../lib/trace";

/**
 * Panel D — counterfactual diff.
 *
 * The outcome category and the computed explanation, side by side with the two
 * answers. The explanation is derived from the two traces, never generated: "`c_12`
 * fell from fused rank 1 to rank 9 and dropped out of context" is checkable against
 * the data on screen, and a model-written narration of a diff it cannot verify would
 * put a plausible sentence exactly where a checkable one belongs.
 */

const OUTCOME: Record<string, { label: string; className: string }> = {
  identical: { label: "identical", className: "text-dim" },
  equivalent: { label: "equivalent", className: "text-dim" },
  improved: { label: "improved", className: "text-win" },
  degraded: { label: "degraded", className: "text-caution" },
  now_refuses: { label: "now refuses", className: "text-caution" },
  now_answers: { label: "now answers", className: "text-caution" },
  now_wrong: { label: "now wrong — but flagged", className: "text-caution" },
  now_confident_wrong: { label: "▲ now confidently wrong", className: "text-reject" },
  error: { label: "variant failed", className: "text-reject" },
};

interface Props {
  ablation: string | null;
  baseline: Trace | null;
  variant: Trace | null;
  outcome: string | null;
  explanation: string | null;
  droppedFromContext: string[];
  rankDelta: Record<string, number>;
  loading: boolean;
}

function Side({ title, trace }: { title: string; trace: Trace | null }) {
  return (
    <div className="min-w-0 flex-1">
      <div className="text-2xs uppercase tracking-widest text-dim mb-1">{title}</div>
      <p className="text-body/90 leading-relaxed max-h-40 overflow-y-auto">
        {trace ? trace.answer.text : <span className="text-dim italic">—</span>}
      </p>
      {trace && (
        <div className="mt-1 text-2xs text-dim tabular">
          {trace.answer.status} · {trace.candidates.filter((c) => c.in_context).length} chunks in
          context
        </div>
      )}
    </div>
  );
}

export default function Counterfactual({
  ablation,
  baseline,
  variant,
  outcome,
  explanation,
  droppedFromContext,
  rankDelta,
  loading,
}: Props) {
  const verdict = outcome ? OUTCOME[outcome] ?? OUTCOME.equivalent : null;
  const moved = Object.entries(rankDelta)
    .filter(([, delta]) => delta !== 0)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 4);

  return (
    <section className="panel flex flex-col min-h-[13rem]">
      <div className="panel-title flex justify-between">
        <span>D · counterfactual</span>
        {verdict && (
          <span className={`normal-case tracking-normal ${verdict.className}`}>
            {verdict.label}
          </span>
        )}
      </div>

      <div className="p-3 flex-1">
        {!ablation && (
          <p className="text-dim italic">
            pick an ablation above to run the same query with a stage removed
          </p>
        )}
        {ablation && loading && <p className="text-dim italic">running {ablation}…</p>}

        {ablation && !loading && (
          <>
            <div className="flex gap-4">
              <Side title="baseline" trace={baseline} />
              <div className="w-px bg-line shrink-0" />
              <Side title={ablation} trace={variant} />
            </div>

            {explanation && (
              <p className="mt-3 pt-3 border-t border-line text-body/80 leading-relaxed">
                {explanation}
              </p>
            )}

            {(moved.length > 0 || droppedFromContext.length > 0) && (
              <div className="mt-2 text-2xs text-dim tabular space-y-0.5">
                {moved.map(([chunkId, delta]) => (
                  <div key={chunkId}>
                    <span className="text-body/70">{chunkId}</span> rank{" "}
                    <span className={delta > 0 ? "text-reject" : "text-win"}>
                      {delta > 0 ? `+${delta}` : delta}
                      {delta === 999 && " (off the list)"}
                    </span>
                  </div>
                ))}
                {droppedFromContext.length > 0 && (
                  <div>{droppedFromContext.length} chunk(s) dropped out of context</div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
