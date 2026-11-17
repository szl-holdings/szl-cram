import assert from 'node:assert/strict';
import { test } from 'node:test';
import worker from '../src/index.mjs';

const env = {};
const easy = '530070000600195000098000060800060003400803001700020000060000280000419005000080079'
  .split('')
  .map(ch => (ch === '0' ? -1 : Number(ch) - 1));

async function call(method, path, body, headers = {}) {
  const req = new Request(`https://worker.example${path}`, {
    method, headers, body: body ? JSON.stringify(body) : undefined,
  });
  return worker.fetch(req, env);
}

test('health is ok', async () => {
  const r = await call('GET', '/health');
  const j = await r.json();
  assert.equal(j.ok, true);
  assert.equal(j.params, 0);
});

test('solves the classic example with a certificate', async () => {
  const r = await call('POST', '/solve', { puzzle: easy, seed: 1 });
  const j = await r.json();
  assert.equal(r.status, 200);
  assert.equal(j.certificate.valid, true);
  assert.equal(j.certificate.E_final, 0);
  assert.equal(j.certificate.model, 'handset-1w');
  for (const key of ['answer', 'E_final', 'threshold', 'margin', 'sweeps']) {
    assert.ok(key in j.certificate, key);
  }
});

test('rejects malformed puzzles', async () => {
  const r = await call('POST', '/solve', { puzzle: [1, 2, 3] });
  assert.equal(r.status, 400);
});

test('free quota is enforced and metered', async () => {
  const free = { 'cf-connecting-ip': '203.0.113.9' };
  for (let i = 0; i < 20; i++) await call('POST', '/solve', { puzzle: easy, seed: 1 }, free);
  const r = await call('POST', '/solve', { puzzle: easy, seed: 1 }, free);
  assert.equal(r.status, 429);
  const u = await call('GET', '/usage');
  const j = await u.json();
  assert.ok(j.requests.length >= 1);
  assert.ok(j.requests[0][1].free >= 21);
});
