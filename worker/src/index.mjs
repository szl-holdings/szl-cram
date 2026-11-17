// SZL CRAM edge solver — classical energy reasoning with certificates.
//
// A JavaScript port of the hand-set energy engine (1 unique weight: same digit
// in a peer pair costs 1, zero trained parameters). Serverless relaxation:
// conflicted cells resample from softmax(-beta * U), chains in parallel,
// energy-threshold early stop. Every answer ships the certificate
// {answer, E_final, threshold, margin, sweeps}.
//
// Monetization scaffold (honest, fail-closed):
//   - free tier: in-memory per-client daily quota (FREE_DAILY_QUOTA, default 20)
//   - pro tier:  x-szl-api-key header checked against PRO_API_KEYS (comma list,
//                owner-set as a Worker secret; absent -> everyone is free tier)
//   - /usage:    daily counters, for the metering pipeline (owner-side billing
//                integration: Stripe or x402; see docs/MONETIZATION.md)
// This worker never sees a payment, a price, or a settlement.

const N = 9, BOX = 3, C = 9, CELLS = 81;

function buildPeers() {
  const peers = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const s = new Set();
      for (let k = 0; k < N; k++) { s.add(r * N + k); s.add(k * N + c); }
      const br = Math.floor(r / BOX) * BOX, bc = Math.floor(c / BOX) * BOX;
      for (let rr = br; rr < br + BOX; rr++)
        for (let cc = bc; cc < bc + BOX; cc++) s.add(rr * N + cc);
      s.delete(r * N + c);
      peers.push([...s]);
    }
  }
  return peers;
}
const PEERS = buildPeers();

function conflicts(board) {
  let e = 0;
  for (let i = 0; i < CELLS; i++) {
    if (board[i] < 0) continue;
    for (const j of PEERS[i]) if (j > i && board[j] === board[i]) e++;
  }
  return e;
}

function isValid(board) {
  return board.every(v => v >= 0) && conflicts(board) === 0;
}

// Focused Gibbs relaxation with beta anneal + restarts. Deterministic per seed.
function solve(puzzle, { chains = 4, maxSweeps = 200, seed = 0 } = {}) {
  let s = seed >>> 0 || 1;
  const rnd = () => { // xorshift32
    s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  const clueMask = puzzle.map(v => v >= 0);
  let best = null, bestE = Infinity, sweeps = 0;
  for (let ch = 0; ch < chains; ch++) {
    const board = puzzle.slice();
    for (let i = 0; i < CELLS; i++) if (!clueMask[i]) board[i] = Math.floor(rnd() * C);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      sweeps++;
      const beta = 1.0 + (3.0 * sweep) / maxSweeps;
      let updated = false;
      for (const i of shuffled(CELLS, rnd)) {
        if (clueMask[i]) continue;
        const counts = new Array(C).fill(0);
        for (const j of PEERS[i]) if (board[j] >= 0) counts[board[j]]++;
        if (counts[board[i]] === 0) continue;        // not conflicted: focused
        updated = true;
        const logits = counts.map(u => -beta * u - 0.5 * rnd());
        const mx = Math.max(...logits);
        const ex = logits.map(l => Math.exp(l - mx));
        const tot = ex.reduce((a, b) => a + b, 0);
        let pick = rnd() * tot, v = C - 1;
        for (let k = 0; k < C; k++) { pick -= ex[k]; if (pick <= 0) { v = k; break; } }
        board[i] = v;
      }
      const e = conflicts(board);
      if (e < bestE) { bestE = e; best = board.slice(); }
      if (e === 0) break;
      if (!updated) { // stuck: random restart of free cells
        for (let i = 0; i < CELLS; i++) if (!clueMask[i]) board[i] = Math.floor(rnd() * C);
      }
    }
  }
  const valid = isValid(best);
  return {
    answer: best,
    certificate: {
      answer: best,
      E_final: bestE,
      threshold: 0.5,
      margin: 0.5 - bestE,
      sweeps,
      valid,
      model: 'handset-1w',
      n: 9, box: 3,
    },
  };
}

function shuffled(n, rnd) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ------------------------------------------------------------------ metering

const quota = new Map(); // clientKey -> { day, count }
const revenue = new Map(); // day -> requests-by-tier { free, pro }
const today = () => new Date().toISOString().slice(0, 10);

function clientKey(request) {
  return (request.headers.get('cf-connecting-ip') || 'unknown').slice(0, 64);
}

function tierOf(request, env) {
  const keys = (env.PRO_API_KEYS || '').split(',').map(k => k.trim()).filter(Boolean);
  const presented = request.headers.get('x-szl-api-key') || '';
  return keys.length > 0 && keys.includes(presented) ? 'pro' : 'free';
}

function allowAndMeter(request, env) {
  const tier = tierOf(request, env);
  if (tier === 'pro') { bump(revenue, 'pro'); return { ok: true, tier }; }
  const key = clientKey(request);
  const d = today();
  let rec = quota.get(key);
  if (!rec || rec.day !== d) { rec = { day: d, count: 0 }; quota.set(key, rec); }
  const cap = parseInt(env.FREE_DAILY_QUOTA || '20', 10);
  if (rec.count >= cap) return { ok: false, tier };
  rec.count++;
  bump(revenue, 'free');
  return { ok: true, tier };
}

function bump(map, k) {
  const d = today();
  const rec = map.get(d) || { free: 0, pro: 0 };
  rec[k]++;
  map.set(d, rec);
}

// ------------------------------------------------------------------ routing

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === 'GET' && url.pathname === '/health') {
      return Response.json({ ok: true, engine: 'handset-1w', params: 0 });
    }
    if (request.method === 'GET' && url.pathname === '/usage') {
      return Response.json({ day: today(), requests: [...revenue.entries()] });
    }
    if (request.method === 'GET' && url.pathname === '/pricing') {
      return Response.json({
        free: { quota: env.FREE_DAILY_QUOTA || '20', scope: 'client-side + limited API' },
        pro: { status: 'private-beta', contact: 'stephen@szlholdings.com' },
        docs: 'https://github.com/szl-holdings/szl-cram/blob/main/docs/MONETIZATION.md',
      });
    }
    if (request.method === 'POST' && url.pathname === '/solve') {
      const gate = allowAndMeter(request, env);
      if (!gate.ok) {
        return Response.json(
          { error: 'free daily quota exceeded', upgrade: 'docs/MONETIZATION.md', tier: gate.tier },
          { status: 429 });
      }
      let body;
      try { body = await request.json(); } catch { return bad('body must be JSON'); }
      const puzzle = body.puzzle;
      if (!Array.isArray(puzzle) || puzzle.length !== CELLS) return bad('puzzle must be 81 cells');
      if (!puzzle.every(v => Number.isInteger(v) && v >= -1 && v <= 8)) return bad('cells must be ints in -1..8');
      const t0 = Date.now();
      const result = solve(puzzle, {
        chains: Math.min(8, Math.max(1, body.chains || 4)),
        maxSweeps: Math.min(1000, Math.max(10, body.max_sweeps || 200)),
        seed: Number.isInteger(body.seed) ? body.seed : 0,
      });
      return Response.json({
        certificate: result.certificate,
        solve_ms: Date.now() - t0,
        tier: gate.tier,
      });
    }
    return new Response('not found', { status: 404 });
  },
};

function bad(msg) {
  return Response.json({ error: msg }, { status: 400 });
}
