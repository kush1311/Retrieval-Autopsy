// GENERATED FILE — do not edit.
//
// Regenerate with:  python -m autopsy.cli schema --ts web/src/lib/trace.ts
//
// Source of truth is autopsy/trace.py. Zod schemas rather than bare types because
// demo mode loads pre-recorded trace JSON straight off disk, and nothing has
// validated those since the moment they were written.

import { z } from "zod";


export const InclusionReasonSchema = z.enum(["fused_top_k", "rerank_promoted", "neighbor_expansion"]);
export type InclusionReason = z.infer<typeof InclusionReasonSchema>;

export const RejectedBySchema = z.enum(["gate", "top_k", "rerank", "tenant_filter"]);
export type RejectedBy = z.infer<typeof RejectedBySchema>;

export const AnswerStatusSchema = z.enum(["grounded", "refused", "ungrounded"]);
export type AnswerStatus = z.infer<typeof AnswerStatusSchema>;

export const CacheStateSchema = z.enum(["hit", "miss"]);
export type CacheState = z.infer<typeof CacheStateSchema>;

export const CandidateSchema = z.object({
  chunk_id: z.string(),
  doc_id: z.string(),
  heading_path: z.array(z.string()).default([]),
  text: z.string(),
  ordinal: z.number(),
  tenant_id: z.string(),
  lexical_rank: z.number().nullable().default(null),
  lexical_score: z.number().nullable().default(null),
  semantic_rank: z.number().nullable().default(null),
  semantic_score: z.number().nullable().default(null),
  fused_rank: z.number().nullable().default(null),
  fused_score: z.number().nullable().default(null),
  rerank_score: z.number().nullable().default(null),
  final_rank: z.number().nullable().default(null),
  in_context: z.boolean().default(false),
  inclusion_reason: InclusionReasonSchema.nullable().default(null),
  rejected_by: RejectedBySchema.nullable().default(null),
});
export type Candidate = z.infer<typeof CandidateSchema>;

export const StageRecordSchema = z.object({
  name: z.string(),
  ms: z.number().default(0.0),
  tokens_in: z.number().default(0),
  tokens_out: z.number().default(0),
  cost_usd: z.number().default(0.0),
  cache: CacheStateSchema.nullable().default(null),
  skipped: z.boolean().default(false),
  skip_reason: z.string().nullable().default(null),
  error: z.string().nullable().default(null),
  detail: z.record(z.string(), z.unknown()).default({}),
});
export type StageRecord = z.infer<typeof StageRecordSchema>;

export const SpanSchema = z.object({
  start: z.number(),
  end: z.number(),
  chunk_ids: z.array(z.string()).default([]),
  supported: z.boolean().default(false),
});
export type Span = z.infer<typeof SpanSchema>;

export const AnswerSchema = z.object({
  text: z.string(),
  status: AnswerStatusSchema,
  spans: z.array(SpanSchema).default([]),
  citations: z.array(z.string()).default([]),
  refusal_reason: z.string().nullable().default(null),
  hedged: z.boolean().default(false),
});
export type Answer = z.infer<typeof AnswerSchema>;

export const TotalsSchema = z.object({
  ms: z.number().default(0.0),
  cost_usd: z.number().default(0.0),
  llm_calls: z.number().default(0),
  tokens_in: z.number().default(0),
  tokens_out: z.number().default(0),
});
export type Totals = z.infer<typeof TotalsSchema>;

export const TraceSchema = z.object({
  schema_version: z.string().default("1.0.0"),
  trace_id: z.string(),
  created_at: z.string(),
  query: z.string(),
  rewritten_query: z.string().nullable().default(null),
  tenant_id: z.string(),
  session_id: z.string().nullable().default(null),
  ablations: z.array(z.string()).default([]),
  config_hash: z.string(),
  config: z.record(z.string(), z.unknown()),
  versions: z.record(z.string(), z.string()),
  candidates: z.array(CandidateSchema).default([]),
  stages: z.array(StageRecordSchema).default([]),
  answer: AnswerSchema,
  totals: TotalsSchema.default({ms: 0.0, cost_usd: 0.0, llm_calls: 0, tokens_in: 0, tokens_out: 0}),
});
export type Trace = z.infer<typeof TraceSchema>;

export const StageEventSchema = z.object({
  type: z.literal("stage").default("stage"),
  name: z.string(),
  ms: z.number().default(0.0),
  skipped: z.boolean().default(false),
  skip_reason: z.string().nullable().default(null),
  cache: CacheStateSchema.nullable().default(null),
  cost_usd: z.number().default(0.0),
});
export type StageEvent = z.infer<typeof StageEventSchema>;

export const CandidatesEventSchema = z.object({
  type: z.literal("candidates").default("candidates"),
  leg: z.enum(["lexical", "semantic"]),
  items: z.array(CandidateSchema),
});
export type CandidatesEvent = z.infer<typeof CandidatesEventSchema>;

export const FusedEventSchema = z.object({
  type: z.literal("fused").default("fused"),
  items: z.array(CandidateSchema),
  gate: z.number().nullable().default(null),
  gate_reads: z.string().nullable().default(null),
  gate_value: z.number().nullable().default(null),
});
export type FusedEvent = z.infer<typeof FusedEventSchema>;

export const AnswerDeltaEventSchema = z.object({
  type: z.literal("answer_delta").default("answer_delta"),
  text: z.string(),
});
export type AnswerDeltaEvent = z.infer<typeof AnswerDeltaEventSchema>;

export const DoneEventSchema = z.object({
  type: z.literal("done").default("done"),
  trace_id: z.string(),
  trace: TraceSchema,
});
export type DoneEvent = z.infer<typeof DoneEventSchema>;

export const ErrorEventSchema = z.object({
  type: z.literal("error").default("error"),
  message: z.string(),
});
export type ErrorEvent = z.infer<typeof ErrorEventSchema>;

export const StreamEventSchema = z.discriminatedUnion("type", [
  StageEventSchema,
  CandidatesEventSchema,
  FusedEventSchema,
  AnswerDeltaEventSchema,
  DoneEventSchema,
  ErrorEventSchema,
]);
export type StreamEvent = z.infer<typeof StreamEventSchema>;

/** Parse a trace from an untrusted source (a demo file, a paste, an old run). */
export function parseTrace(raw: unknown): Trace {
  return TraceSchema.parse(raw);
}
