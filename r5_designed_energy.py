"""R5: hand-set graph-coloring energy for estate ops (gpu-bridge-style container->GPU placement). Zero training. Emits energy certificates."""
import numpy as np, json, time, datetime

rng = np.random.default_rng(42)
N, C, D = 32, 8, 3          # 32 containers, 8 GPUs, target degree

# conflict graph: D random 'must-not-colocate' peers per container (anti-affinity rules)
edges = set()
for i in range(N):
    for j in rng.choice([x for x in range(N) if x != i], size=D, replace=False):
        edges.add((min(i, int(j)), max(i, int(j))))
edges = sorted(edges)
E_LIST = [[] for _ in range(N)]
for a, b in edges:
    E_LIST[a].append(b); E_LIST[b].append(a)

# hand-set pairwise energy: cost 1 iff same GPU on a conflict edge (1 unique weight, paper-style)
W = np.zeros((N, N, C, C), dtype=np.float32)
for a in range(N):
    for k, b in enumerate(E_LIST[a]):
        for v in range(C):
            W[a, k, v, v] = 1.0

def energy(x):
    e = 0.0
    for a in range(N):
        for k, b in enumerate(E_LIST[a]):
            e += W[a, k, x[a], x[b]]
    return e / 2.0

def solve(chains=16, sweeps=200, seed=0):
    r = np.random.default_rng(seed)
    best_x, best_e = None, np.inf
    for ch in range(chains):
        x = r.integers(0, C, size=N)
        for sw in range(sweeps):
            for i in r.permutation(N):
                costs = np.array([sum(W[i, k, v, x[E_LIST[i][k]]] for k in range(len(E_LIST[i]))) for v in range(C)])
                x[i] = int(np.argmin(costs + (r.random(C) < 0.05) * 0.5))
            e = energy(x)
            if e < best_e:
                best_e, best_x = e, x.copy()
            if best_e == 0.0:
                return best_x, best_e, (ch, sw)
    return best_x, best_e, (chains - 1, sweeps - 1)

t0 = time.time()
x, e, (ch, sw) = solve(seed=7)
el = time.time() - t0
cert = {'answer': x.tolist(), 'E_final': float(e), 'threshold': 0.0, 'margin': float(0.0 - e),
        'sweeps': int(sw), 'chain': int(ch), 'valid': bool(e == 0.0)}
out = {
  'stage': 'r5_designed_energy',
  'problem': {'containers': N, 'gpus': C, 'conflict_edges': len(edges), 'degree': D},
  'energy': {'unique_weights': 1, 'trained_params': 0, 'form': 'hand-set pairwise: cost=1 iff same GPU on anti-affinity edge'},
  'certificate': cert,
  'search_sec': round(el, 3),
  't': datetime.datetime.now(datetime.timezone.utc).isoformat()
}
json.dump(out, open(r'C:\SZL\cram-study\r5_receipt.json', 'w'), indent=2)
print('R5_DONE E=' + str(e) + ' sec=' + str(round(el, 2)))
