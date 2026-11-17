# Monetization (szl-cram)

Design principle: the engine is free and open; the metered value is the edge service and
the enterprise design work. Billing rails are owner-side and fail-closed.

## Tiers

| Tier | Price | What |
| --- | --- | --- |
| Free | $0 | In-browser solver (client-side, unlimited), 20 edge solves/day/IP, Apache-2.0 core library |
| Pro API (private beta) | TBD by owner | Unmetered edge solves, learned-energy endpoints, batch + certificate export, API key |
| Enterprise | custom | Designed energies for your constraint domain, certificates in your governance, support |

## Implementation state (all receipted in-repo)

- **Edge metering is live in code:** `worker/src/index.mjs` keeps an in-memory daily
  counter per client IP (`FREE_DAILY_QUOTA`, default 20), returns 429 with
  `retry-after` when exceeded, exposes `/usage` so clients can see their meter, and
  separates pro tier via `x-szl-api-key` checked against the `PRO_API_KEYS` worker
  secret (absent = no pro access, fail-closed).
- **In-memory metering resets on worker restart** and does not shard across colos:
  acceptable for a beta funnel, documented honestly. Durable metering = Workers KV/D1
  (owner-gated infra addition).
- **The worker never sees a payment.** It only authorizes API keys. This is deliberate:
  the edge stays minimal-audit-surface.

## Owner-side flips (choose rails)

1. **Stripe** (recommended for Pro beta): create product/prices, issue API keys to
   beta customers manually, set `PRO_API_KEYS` worker secret from the Stripe webhook
   output. The estate Stripe connector can be enabled in-session on request.
2. **x402 (per-call crypto metering)**: aligns with the estate's x402 lane; wire the
   worker's `/solve` behind x402 paywall for anonymous metered access. Owner-gated
   (x402 doctrine pending).
3. **Enterprise contracts:** direct, stephen@szlholdings.com.

Nothing in this document fabricates revenue. As of this writing: zero paying customers,
rails not yet wired; metering and tier separation are code-complete and receipt-bound.
