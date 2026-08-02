# Atlas connections and pooling

## Connection limits

```yaml
max_connections: 200
superuser_reserved_connections: 5
idle_session_timeout_seconds: 0    # 0 disables
statement_timeout_seconds: 0
```

Each connection is a separate backend process with its own memory. Raising
`max_connections` is not free: the per-connection overhead of work memory and catalog
caches means a server tuned for 200 connections can thrash at 2000 even though the
query mix has not changed.

When the limit is reached, new connections are refused with `ATL-4101`. The slots
reserved by `superuser_reserved_connections` are held back from ordinary users
specifically so an operator can still get in and see what is happening.

## Pooling

Atlas expects a connection pooler in front of it for any application with more than a
handful of workers. Three pooling modes are conventional:

- **session** — a client holds a server connection for the life of its session. Safe
  with everything, saves the least.
- **transaction** — a server connection is assigned per transaction. The usual choice.
  Session-scoped state does not survive: temporary tables, `SET` outside a
  transaction, prepared statements, and advisory locks all break.
- **statement** — a server connection per statement. Multi-statement transactions are
  impossible.

Applications moving from session to transaction pooling most often break on prepared
statements, because the failure is intermittent — it depends on which server
connection the pooler happens to hand out.

## Timeouts

`statement_timeout_seconds` cancels a query that runs too long. It applies to the
statement, not to the transaction, so a transaction issuing many short statements can
stay open indefinitely. Use `idle_in_transaction_session_timeout_seconds` to bound
that separately; an idle-in-transaction session holds locks and prevents cleanup of
row versions, which is a far more damaging state than a slow query.

## Throttling

Atlas has no admission-control or rate-limiting facility. There is no setting that
queues or slows incoming queries when the server is loaded; connections are either
accepted or refused at `max_connections`. Rate limiting belongs in the pooler or the
application.
