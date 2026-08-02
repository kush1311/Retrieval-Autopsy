import {
  StreamEventSchema,
  TraceSchema,
  type StreamEvent,
  type Trace,
} from "./trace";

export interface Meta {
  tenants: string[];
  ablations: { name: string; expected: string }[];
  corpus: string;
  provider: string;
  chunks: number;
}

/** Static build with no backend: replay pre-recorded traces instead of calling one. */
export const DEMO_ONLY = import.meta.env.VITE_DEMO_ONLY === "1";

const API = import.meta.env.VITE_API_BASE ?? "/api";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${path}: ${body.slice(0, 300)}`);
  }
  return (await response.json()) as T;
}

export const getMeta = () => json<Meta>("/meta");

export interface QueryRequest {
  query: string;
  tenant_id: string;
  ablations: string[];
  history?: string[];
}

export async function runQuery(req: QueryRequest): Promise<Trace> {
  // Parsed, not cast. The wire is a boundary, and `as Trace` on a response that has
  // drifted from the schema produces undefined-shaped objects three components deep,
  // far from the thing that actually changed.
  return TraceSchema.parse(await json<unknown>("/query", {
    method: "POST",
    body: JSON.stringify(req),
  }));
}

export interface CounterfactualResponse {
  baseline: Trace;
  variant: Trace;
  outcome: string;
  explanation: string;
  dropped_from_context: string[];
  rank_delta: Record<string, number>;
}

export async function runCounterfactual(req: QueryRequest): Promise<CounterfactualResponse> {
  const raw = await json<CounterfactualResponse>("/counterfactual", {
    method: "POST",
    body: JSON.stringify(req),
  });
  return {
    ...raw,
    baseline: TraceSchema.parse(raw.baseline),
    variant: TraceSchema.parse(raw.variant),
  };
}

/**
 * Stream one query, calling `onEvent` as each stage completes.
 *
 * Progressive fill is most of why this reads as an instrument: the reranker's decision
 * lands before the answer does. Waiting for the full run and rendering once shows
 * nothing for four seconds and then everything at once.
 */
export function streamQuery(
  req: QueryRequest,
  onEvent: (event: StreamEvent) => void,
  onError: (message: string) => void,
): () => void {
  const url = new URL(
    import.meta.env.VITE_WS_BASE ?? "/stream",
    window.location.href.replace(/^http/, "ws"),
  );
  const socket = new WebSocket(url.toString());
  let closed = false;

  socket.onopen = () => socket.send(JSON.stringify(req));
  socket.onmessage = (message) => {
    const parsed = StreamEventSchema.safeParse(JSON.parse(message.data as string));
    if (!parsed.success) {
      onError(`unrecognised event from server: ${parsed.error.issues[0]?.message ?? ""}`);
      return;
    }
    onEvent(parsed.data);
    if (parsed.data.type === "done" || parsed.data.type === "error") {
      closed = true;
      socket.close();
    }
  };
  socket.onerror = () => {
    if (!closed) onError("websocket error — is the API running on :8000?");
  };

  return () => {
    closed = true;
    socket.close();
  };
}
