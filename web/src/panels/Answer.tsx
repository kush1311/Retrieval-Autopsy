import type { Answer as AnswerType } from "../lib/trace";

/**
 * Panel C — the answer, with attribution.
 *
 * Hovering a clause highlights its source chunk in panel A; clauses with no supporting
 * chunk are flagged inline. That flag is the grounding audit, and it is the panel a
 * compliance or support user would actually open — "which sentence came from where"
 * is a different question from "is the answer good", and only one of them is
 * checkable.
 *
 * The panel reserves its height up front. Generation finishes two to four seconds
 * after retrieval, and a panel that grows from nothing shoves the entire layout
 * downward mid-demo.
 */

const STATUS: Record<string, { text: string; className: string }> = {
  grounded: { text: "grounded", className: "text-win" },
  refused: { text: "refused", className: "text-caution" },
  ungrounded: { text: "ungrounded · unsupported claims", className: "text-reject" },
};

interface Props {
  answer: AnswerType | null;
  streaming: string;
  highlighted: string | null;
  onHover: (chunkId: string | null) => void;
}

export default function Answer({ answer, streaming, highlighted, onHover }: Props) {
  const status = answer ? STATUS[answer.status] ?? STATUS.grounded : null;

  return (
    <section className="panel flex flex-col min-h-[13rem]">
      <div className="panel-title flex justify-between">
        <span>C · answer</span>
        {status && <span className={`normal-case tracking-normal ${status.className}`}>
          {status.text}
        </span>}
      </div>

      <div className="p-3 flex-1 leading-relaxed">
        {!answer && !streaming && (
          <p className="text-dim italic">awaiting generation…</p>
        )}

        {!answer && streaming && <p className="text-body">{streaming}</p>}

        {answer && answer.spans.length === 0 && <p className="text-body">{answer.text}</p>}

        {answer &&
          answer.spans.map((span, index) => {
            const text = answer.text.slice(span.start, span.end);
            const active = span.chunk_ids.some((id) => id === highlighted);
            return (
              <span
                key={`${span.start}-${index}`}
                onMouseEnter={() => onHover(span.chunk_ids[0] ?? null)}
                onMouseLeave={() => onHover(null)}
                className={`${
                  span.supported
                    ? active
                      ? "bg-win/20 text-bright"
                      : "hover:bg-win/10"
                    : "underline decoration-reject decoration-wavy underline-offset-4 text-reject"
                } cursor-default`}
                title={
                  span.supported
                    ? `source: ${span.chunk_ids.join(", ")}`
                    : "no supporting chunk — this clause is not grounded in any retrieved source"
                }
              >
                {text}{" "}
              </span>
            );
          })}
      </div>

      {answer && (
        <div className="px-3 py-2 border-t border-line flex flex-wrap gap-x-4 gap-y-1 text-2xs text-dim">
          <span>
            {answer.citations.length} citation{answer.citations.length === 1 ? "" : "s"}
          </span>
          <span>
            {answer.spans.filter((s) => !s.supported).length} unattributed clause
            {answer.spans.filter((s) => !s.supported).length === 1 ? "" : "s"}
          </span>
          {answer.hedged && <span className="text-caution">hedged</span>}
          {answer.refusal_reason && <span>· {answer.refusal_reason}</span>}
        </div>
      )}
    </section>
  );
}
