import { useCallback, useEffect, useRef, useState } from "react";

import {
  DEMO_ONLY,
  getMeta,
  runCounterfactual,
  streamQuery,
  type CounterfactualResponse,
  type Meta,
} from "./lib/api";
import { loadDemo, replay, type DemoEntry } from "./lib/demo";
import type { Candidate, StageRecord, StreamEvent, Trace } from "./lib/trace";
import Answer from "./panels/Answer";
import Competition from "./panels/Competition";
import Counterfactual from "./panels/Counterfactual";
import Timeline from "./panels/Timeline";

/**
 * A single view with four panels. No routing, no top-level tabs.
 *
 * It is a debugger, not a dashboard: it inspects one request completely. Cross-request
 * aggregation is a different product, and the moment this grows a "runs" list it stops
 * being the thing that makes one query legible.
 */

interface Live {
  stages: StageRecord[];
  lexical: Candidate[];
  semantic: Candidate[];
  fused: Candidate[];
  gate: number | null;
  gateValue: number | null;
  gateReads: string | null;
  streaming: string;
  trace: Trace | null;
}

const EMPTY: Live = {
  stages: [],
  lexical: [],
  semantic: [],
  fused: [],
  gate: null,
  gateValue: null,
  gateReads: null,
  streaming: "",
  trace: null,
};

function reduce(state: Live, event: StreamEvent): Live {
  switch (event.type) {
    case "stage":
      return {
        ...state,
        stages: [
          ...state.stages.filter((s) => s.name !== event.name),
          {
            name: event.name,
            ms: event.ms,
            tokens_in: 0,
            tokens_out: 0,
            cost_usd: event.cost_usd,
            cache: event.cache,
            skipped: event.skipped,
            skip_reason: event.skip_reason,
            error: null,
            detail: {},
          },
        ],
      };
    case "candidates":
      return event.leg === "lexical"
        ? { ...state, lexical: event.items }
        : { ...state, semantic: event.items };
    case "fused":
      return {
        ...state,
        fused: event.items,
        gate: event.gate,
        gateValue: event.gate_value,
        gateReads: event.gate_reads,
      };
    case "answer_delta":
      return { ...state, streaming: state.streaming + event.text };
    case "done":
      // The final trace supersedes everything accumulated during streaming: the
      // events carry partial records, the trace carries the authoritative ones.
      return {
        ...state,
        trace: event.trace,
        stages: event.trace.stages,
        fused: event.trace.candidates,
      };
    default:
      return state;
  }
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [demo] = useState(() => (DEMO_ONLY ? loadDemo() : null));
  const [query, setQuery] = useState("what does KLV-4021 mean");
  const [tenant, setTenant] = useState("tenant_kelvin");
  const [ablation, setAblation] = useState<string | null>(null);
  const [live, setLive] = useState<Live>(EMPTY);
  const [cf, setCf] = useState<CounterfactualResponse | null>(null);
  const [cfLoading, setCfLoading] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const cancel = useRef<{ cancelled: boolean }>({ cancelled: false });

  useEffect(() => {
    if (DEMO_ONLY) return;
    getMeta().then(setMeta).catch((e: Error) => setError(e.message));
  }, []);

  const run = useCallback(async () => {
    cancel.current.cancelled = true;
    cancel.current = { cancelled: false };
    const token = cancel.current;

    setError(null);
    setLive(EMPTY);
    setCf(null);
    setBusy(true);

    const onEvent = (event: StreamEvent) => {
      if (token.cancelled) return;
      if (event.type === "error") {
        setError(event.message);
        setBusy(false);
        return;
      }
      setLive((prev) => reduce(prev, event));
      if (event.type === "done") setBusy(false);
    };

    if (DEMO_ONLY && demo) {
      const entry =
        demo.entries.find((e) => e.query === query) ?? demo.entries[0];
      const traceId = ablation ? entry?.variants[ablation] ?? entry?.baseline : entry?.baseline;
      const trace = traceId ? demo.traces.get(traceId) : undefined;
      if (!trace) {
        setError("no pre-recorded trace for that combination in demo mode");
        setBusy(false);
        return;
      }
      await replay(trace, onEvent, 14, token);
      return;
    }

    streamQuery({ query, tenant_id: tenant, ablations: ablation ? [ablation] : [] }, onEvent, (m) => {
      setError(m);
      setBusy(false);
    });
  }, [ablation, demo, query, tenant]);

  useEffect(() => {
    void run();
    // Intentionally on mount only — re-running on every keystroke would fire a
    // pipeline run per character.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleAblation = useCallback(
    async (name: string | null) => {
      setAblation(name);
      setCf(null);
      if (!name || DEMO_ONLY) return;
      setCfLoading(true);
      try {
        setCf(
          await runCounterfactual({ query, tenant_id: tenant, ablations: [name] }),
        );
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setCfLoading(false);
      }
    },
    [query, tenant],
  );

  const demoEntry: DemoEntry | undefined = demo?.entries.find((e) => e.query === query);

  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-line bg-panel">
        <div className="flex flex-wrap items-center gap-2 p-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void run()}
            spellCheck={false}
            className="flex-1 min-w-[16rem] bg-ink border border-line px-2 py-1.5 text-bright outline-none focus:border-win/60"
            placeholder="ask something with an exact identifier in it…"
          />
          <select
            value={tenant}
            onChange={(e) => setTenant(e.target.value)}
            className="bg-ink border border-line px-2 py-1.5 text-body outline-none"
          >
            {(meta?.tenants ?? [tenant]).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <select
            value={ablation ?? ""}
            onChange={(e) => void toggleAblation(e.target.value || null)}
            className="bg-ink border border-line px-2 py-1.5 text-body outline-none"
          >
            <option value="">baseline</option>
            {(meta?.ablations ?? Object.keys(demoEntry?.variants ?? {}).map((n) => ({ name: n, expected: "" })))
              .map((a) => (
                <option key={a.name} value={a.name} title={a.expected}>
                  {a.name}
                </option>
              ))}
          </select>
          <button
            onClick={() => void run()}
            disabled={busy}
            className="border border-win/50 text-win px-3 py-1.5 hover:bg-win/10 disabled:opacity-40"
          >
            {busy ? "running…" : "run"}
          </button>
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 pb-2 text-2xs text-dim">
          {DEMO_ONLY && (
            <span className="text-caution">
              demo mode · replaying pre-recorded traces, no API key required
            </span>
          )}
          {meta && (
            <>
              <span>corpus {meta.corpus}</span>
              <span>{meta.chunks} chunks</span>
              <span>provider {meta.provider}</span>
              {meta.provider === "offline" && (
                <span className="text-caution">
                  offline simulator — deterministic stand-in, not a language model
                </span>
              )}
            </>
          )}
          {error && <span className="text-reject">{error}</span>}
        </div>
      </header>

      {/* Four panels on desktop, one stacked column under 1024px. LinkedIn traffic is
          majority mobile and a four-panel grid collapses into unreadable slivers —
          shipping a broken layout to most of the people who click the link is worse
          than shipping no layout at all. */}
      <main className="flex-1 grid gap-2 p-2 grid-cols-1 lg:grid-cols-2 min-h-0">
        <div className="lg:row-span-2 min-h-0 flex flex-col">
          <Competition
            lexical={live.lexical}
            semantic={live.semantic}
            fused={live.fused}
            gate={live.gate}
            gateValue={live.gateValue}
            gateReads={live.gateReads}
            rerank={live.stages.find((s) => s.name === "rerank")}
            highlighted={highlighted}
            onHover={setHighlighted}
          />
        </div>

        <Answer
          answer={live.trace?.answer ?? null}
          streaming={live.streaming}
          highlighted={highlighted}
          onHover={setHighlighted}
        />

        <Counterfactual
          ablation={ablation}
          baseline={cf?.baseline ?? null}
          variant={cf?.variant ?? live.trace ?? null}
          outcome={cf?.outcome ?? null}
          explanation={cf?.explanation ?? null}
          droppedFromContext={cf?.dropped_from_context ?? []}
          rankDelta={cf?.rank_delta ?? {}}
          loading={cfLoading}
        />
      </main>

      <footer className="p-2">
        <Timeline
          stages={live.stages}
          totalMs={live.trace?.totals.ms ?? 0}
          totalCost={live.trace?.totals.cost_usd ?? 0}
        />
      </footer>
    </div>
  );
}
