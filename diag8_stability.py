import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
def run():
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    tp, ts = d['test_puz'][:10], d['test_sol'][:10]
    mo = np.load('model100.npz')
    m = CramSudoku(9, 3, seed=1)
    m.W[...] = mo['W']; m.B[...] = mo['B']
    thr, _, _ = m.fit_stopping_threshold(d['train'])
    def state_E(x):
        return float(m.energy(np.eye(m.C, dtype=np.float32)[x]))
    rng = np.random.default_rng(99)
    res = {'thr': thr, 'puzzles': []}
    tot_ret, tot_flip = 0, 0
    for q in range(10):
        sol = ts[q].copy()
        solE = state_E(sol)
        free = np.where(tp[q] < 0)[0]
        ret = 0; flips = 0; ends = []
        for f in rng.choice(free, size=8, replace=False):
            wrong = [v for v in range(9) if v != sol[f]]
            x = sol.copy(); x[f] = int(rng.choice(wrong))
            a, _, _ = m.search(tp[q:q+1], chains=1, init_X=x[None, :], beta_start=6.0, beta_end=6.0, max_sweeps=80, mix_p=0.0, threshold=None, noise=0.0, seed=int(rng.integers(1,10**9)), quorum=False)
            x2 = a[0]
            if bool((x2 == sol).all()): ret += 1
            ends.append(round(state_E(x2), 2))
            flips += 1
        res['puzzles'].append({'q': q, 'solE': round(solE, 2), 'flips': flips, 'returned': ret, 'end_E_mean': round(float(np.mean(ends)), 2), 'end_E_min': float(np.min(ends))})
        tot_ret += ret; tot_flip += flips
    res['summary'] = {'flip_returns': tot_ret, 'total_flips': tot_flip, 'return_rate': round(tot_ret / tot_flip, 3)}
    return res
if not os.path.exists('diag8.json') or os.path.exists('diag8_run.txt'):
    try:
        res = run()
        res['ok'] = True
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-2000:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag8.json', 'w'), indent=2)
print('diag8 done')
