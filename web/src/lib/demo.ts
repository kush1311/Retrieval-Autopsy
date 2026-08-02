import { parseTrace, type StreamEvent, type Trace } from "./trace";

/**
 * Demo mode: replay pre-recorded traces with realistic timing, no key and no backend.
 *
 * Anyone evaluating this work will click the deployed link rather than clone the repo,
 * and a cold-start demo that asks for an API key gets closed in two seconds. The
 * traces are the real thing — produced by the same pipeline, frozen by
 * `python -m autopsy.cli demo` — so what the demo shows and what the eval measures
 * cannot diverge.
 */

export interface DemoEntry {
  query: string;
  tenant: string;
  baseline: string;
  variants: Record<string, string>;
}

const files = import.meta.glob("../demo/traces/*.json", { eager: true, import: "default" });

function byId(): Map<string, Trace> {
  const map = new Map<string, Trace>();
  for (const [path, value] of Object.entries(files)) {
    if (path.endsWith("index.json")) continue;
    const trace = parseTrace(value);
    map.set(trace.trace_id, trace);
  }
  return map;
}

export function loadDemo(): { entries: DemoEntry[]; traces: Map<string, Trace> } {
  const index = Object.entries(files).find(([p]) => p.endsWith("index.json"));
  const entries = (index ? (index[1] as DemoEntry[]) : []) ?? [];
  return { entries, traces: byId() };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Re-emit a finished trace as the event sequence that produced it.
 *
 * Timing comes from each stage's recorded `ms`, scaled up: the offline pipeline
 * finishes in ~15ms, and replaying it truthfully would flash the whole thing in one
 * frame. The scale factor is honest about being a presentation choice — the numbers
 * shown in the timeline are the real recorded ones, only the pacing is stretched.
 */
export async function replay(
  trace: Trace,
  onEvent: (event: StreamEvent) => void,
  scale = 14,
  signal?: { cancelled: boolean },
): Promise<void> {
  const lexical = trace.candidates
    .filter((c) => c.lexical_rank !== null)
    .sort((a, b) => (a.lexical_rank ?? 0) - (b.lexical_rank ?? 0));
  const semantic = trace.candidates
    .filter((c) => c.semantic_rank !== null)
    .sort((a, b) => (a.semantic_rank ?? 0) - (b.semantic_rank ?? 0));
  const fused = trace.candidates
    .filter((c) => c.fused_rank !== null)
    .sort((a, b) => (a.fused_rank ?? 0) - (b.fused_rank ?? 0));

  const gateStage = trace.stages.find((s) => s.name === "gate");
  const gateDetail = (gateStage?.detail ?? {}) as { threshold?: number; value?: number; reads?: string };

  for (const stage of trace.stages) {
    if (signal?.cancelled) return;
    await sleep(Math.max(90, stage.ms * scale));
    onEvent({
      type: "stage",
      name: stage.name,
      ms: stage.ms,
      skipped: stage.skipped,
      skip_reason: stage.skip_reason,
      cache: stage.cache,
      cost_usd: stage.cost_usd,
    });
    if (stage.name === "retrieve_dense" && semantic.length) {
      onEvent({ type: "candidates", leg: "semantic", items: semantic });
    }
    if (stage.name === "retrieve_sparse" && lexical.length) {
      onEvent({ type: "candidates", leg: "lexical", items: lexical });
    }
    if (stage.name === "gate") {
      onEvent({
        type: "fused",
        items: fused,
        gate: gateDetail.threshold ?? null,
        gate_reads: gateDetail.reads ?? null,
        gate_value: gateDetail.value ?? null,
      });
    }
  }

  // Type the answer out, so the panel fills the way a real generation does.
  const words = trace.answer.text.split(" ");
  for (let i = 0; i < words.length; i += 3) {
    if (signal?.cancelled) return;
    await sleep(45);
    onEvent({ type: "answer_delta", text: words.slice(i, i + 3).join(" ") + " " });
  }
  onEvent({ type: "done", trace_id: trace.trace_id, trace });
}
