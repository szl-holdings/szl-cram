import numpy as np, json, time, random, datetime, os
OUT = r'C:\SZL\cram-study'
os.makedirs(OUT, exist_ok=True)

def peers_of(n, box):
    # n x n grid, box b x b cells (box*b=n)
    P = []
    for r in range(n):
        for c in range(n):
            s = set()
            for k in range(n):
                s.add(r*n+k); s.add(k*n+c)
            br, bc = (r//box)*box, (c//box)*box
            for rr in range(br, br+box):
                for cc in range(bc, bc+box):
                    s.add(rr*n+cc)
            s.discard(r*n+c)
            P.append(sorted(s))
    return P

FULL = 0
def count_solutions(grid, peers, C, cap=2, rng=None):
    # bitmask constraint-propagation backtracking, count up to cap
    N = len(grid)
    allmask = (1 << C) - 1
    cell_vals = []
    for i in range(N):
        v = int(grid[i])
        cell_vals.append(1 << v if v >= 0 else 0)
    def propagate(cv):
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if cv[i] and bin(cv[i]).count('1') == 1:
                    m = cv[i]
                    for j in peers[i]:
                        if cv[j] & m:
                            cv[j] &= ~m
                            changed = True
                            if cv[j] == 0:
                                return False
        return True
    # convert bitmask state into search with MRV
    def search(cv, cnt):
        if cnt[0] >= cap:
            return
        best, bi = None, -1
        for i in range(N):
            if cv[i] == 0:
                # derive candidates from peers (lazy masks)
                m = allmask
                for j in peers[i]:
                    if bin(cv[j]).count('1') == 1:
                        m &= ~cv[j]
                cv[i] = m
            nb = bin(cv[i]).count('1')
            if nb > 1 and (best is None or nb < best):
                best, bi = nb, i
        if bi == -1:
            cnt[0] += 1
            return
        m = cv[bi]
        bits = [b for b in range(C) if (m >> b) & 1]
        if rng is not None:
            rng.shuffle(bits)
        for b in bits:
            cv2 = list(cv)
            cv2[bi] = 1 << b
            if propagate(cv2):
                search(cv2, cnt)
                if cnt[0] >= cap:
                    return
    cnt = [0]
    if not propagate(cell_vals):
        return 0
    search(cell_vals, cnt)
    return cnt[0]

def gen_solution(n, peers, rng):
    C = n
    grid = [-1]* (n*n)
    def bt(grid):
        best, bi = None, -1
        for i in range(n*n):
            if grid[i] == -1:
                used = set(grid[j] for j in peers[i] if grid[j] >= 0)
                cands = [v for v in range(C) if v not in used]
                if not cands:
                    return None
                if best is None or len(cands) < best:
                    best, bi = len(cands), (i, cands)
        if bi == -1:
            return list(grid)
        i, cands = bi
        rng.shuffle(cands)
        for v in cands:
            grid[i] = v
            r = bt(grid)
            if r is not None:
                return r
            grid[i] = -1
        return None
    return bt(grid)

def dig(solution, n, peers, rng, min_clues, max_digs=None):
    C = n
    grid = list(solution)
    order = list(range(n*n))
    rng.shuffle(order)
    digs = 0
    for i in order:
        clues = sum(1 for v in grid if v >= 0)
        if clues <= min_clues:
            break
        if max_digs is not None and digs >= max_digs:
            break
        old = grid[i]
        grid[i] = -1
        if count_solutions(grid, peers, C, cap=2, rng=rng) != 1:
            grid[i] = old
        else:
            digs += 1
    return grid

def save_receipt(stage, data):
    data['stage'] = stage
    data['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(data, open(os.path.join(OUT, 'gen_receipt.json'), 'w'), indent=2)

rng = random.Random(26111730)
t0 = time.time()
save_receipt('start', {})

# 9x9
n, box = 9, 3
peers9 = peers_of(n, box)
train_sols = []
for _ in range(600):
    s = gen_solution(n, peers9, rng)
    train_sols.append(s)
save_receipt('train_sols_9x9_done', {'n_train': len(train_sols), 'sec': round(time.time()-t0,1)})
test = []
for k in range(150):
    s = gen_solution(n, peers9, rng)
    g = dig(s, n, peers9, rng, min_clues=24)
    test.append({'puzzle': g, 'solution': s})
    if (k+1) % 25 == 0:
        save_receipt('test_9x9_progress', {'done': k+1, 'sec': round(time.time()-t0,1)})
clues = [sum(1 for v in t['puzzle'] if v >= 0) for t in test]
np.savez_compressed(os.path.join(OUT, 'sudoku9.npz'),
    train=np.array(train_sols, dtype=np.int8),
    test_puz=np.array([t['puzzle'] for t in test], dtype=np.int8),
    test_sol=np.array([t['solution'] for t in test], dtype=np.int8))
save_receipt('9x9_done', {'sec': round(time.time()-t0,1), 'clue_min': min(clues), 'clue_mean': round(float(np.mean(clues)),1), 'clue_max': max(clues)})

# 16x16 scale probe
n2, box2 = 16, 4
peers16 = peers_of(n2, box2)
train_sols16 = []
for _ in range(250):
    train_sols16.append(gen_solution(n2, peers16, rng))
test16 = []
for k in range(30):
    s = gen_solution(n2, peers16, rng)
    g = dig(s, n2, peers16, rng, min_clues=120, max_digs=140)
    test16.append({'puzzle': g, 'solution': s})
    save_receipt('test_16x16_progress', {'done': k+1, 'sec': round(time.time()-t0,1)})
clues16 = [sum(1 for v in t['puzzle'] if v >= 0) for t in test16]
np.savez_compressed(os.path.join(OUT, 'sudoku16.npz'),
    train=np.array(train_sols16, dtype=np.int8),
    test_puz=np.array([t['puzzle'] for t in test16], dtype=np.int8),
    test_sol=np.array([t['solution'] for t in test16], dtype=np.int8))
save_receipt('done', {'sec': round(time.time()-t0,1), 'clues16_min': min(clues16), 'clues16_mean': round(float(np.mean(clues16)),1), 'clues16_max': max(clues16)})
print('done')