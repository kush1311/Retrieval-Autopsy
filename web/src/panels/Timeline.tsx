import type { StageRecord } from "../lib/trace";

/**
 * Panel B — stage timeline.
 *
 * **Skipped stages render greyed, never omitted.** Absence is information: a pipeline
 * that looks shorter under an ablation gives exactly the wrong impression, because the
 * stage did not disappear — it declined to run, and the reason it gives is usually the
 * most interesting thing on the screen.
 */

const ORDER = [
  "rewrite",
  "embed",
  "retrieve_dense",
  "retrieve_sparse",
  "fuse",
  "gate",
  "rerank",
  "expand",
  "generate",
];

interface Props {
  stages: StageRecord[];
  totalMs: number;
  totalCost: number;
}

export default function Timeline({ stages, totalMs, totalCost }: Props) {
  const byName = new Map(stages.map((s) => [s.name, s]));
  const slowest = Math.max(1, ...stages.filter((s) => !s.skipped).map((s) => s.ms));

  return (
    <section className="panel">
      <div className="panel-title flex justify-between">
        <span>B · pipeline</span>
        <span className="text-dim/70 normal-case tracking-normal tabular">
          {totalMs.toFixed(0)}ms · ${totalCost.toFixed(5)}
        </span>
      </div>
      <div className="flex overflow-x-auto">
        {ORDER.map((name) => {
          const stage = byName.get(name);
          const pending = stage === undefined;
          const skipped = stage?.skipped ?? false;
          const width = stage && !skipped ? Math.max(6, (stage.ms / slowest) * 100) : 0;
          return (
            <div
              key={name}
              title={stage?.skip_reason ?? stage?.error ?? name}
              className={`group relative flex-1 min-w-[92px] border-r border-line last:border-r-0 px-2 py-2 ${
                pending ? "opacity-30" : skipped ? "opacity-45" : ""
              }`}
            >
              <div className="flex items-baseline justify-between gap-1">
                <span className={`truncate ${skipped ? "text-dim" : "text-bright"}`}>
                  {name}
                </span>
                {stage?.cache && (
                  <span
                    className={`text-2xs shrink-0 ${
                      stage.cache === "hit" ? "text-win" : "text-dim"
                    }`}
                  >
                    {stage.cache}
                  </span>
                )}
              </div>

              <div className="mt-1 h-0.5 bg-line">
                <div
                  className={`h-0.5 ${skipped ? "bg-dim/40" : "bg-win"}`}
                  style={{ width: `${width}%` }}
                />
              </div>

              <div className="mt-1 flex justify-between text-2xs text-dim tabular">
                <span>{pending ? "—" : skipped ? "skipped" : `${stage!.ms.toFixed(1)}ms`}</span>
                {stage && stage.tokens_in + stage.tokens_out > 0 && (
                  <span>{(stage.tokens_in + stage.tokens_out).toLocaleString()}t</span>
                )}
              </div>

              {stage?.error && <div className="mt-1 text-2xs text-reject">error</div>}

              {stage?.skip_reason && (
                <div className="pointer-events-none absolute bottom-full left-0 z-20 mb-1 hidden w-72 border border-line bg-raised p-2 text-2xs text-body group-hover:block">
                  {stage.skip_reason}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
