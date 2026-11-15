import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
res = {'stages': [], 't_start': datetime.datetime.now(datetime.timezone.utc).isoformat()}
json.dump(res, open('diag9.json', 'w'), indent=2)
def run():
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    tp, ts = d['test_puz'][:10], d['test_sol'][:10]
    mo = np.load('model100.npz')
    m = CramSudoku(9, 3, seed=1)
    m.W[...] = mo['W']; m.B[...] = mo['B']
    thr, _, _ = m.fit_stopping_threshold(d['train'])
    out = {'thr': thr, 'stages': []}
    rng = np.random.default_rng(555)
    for kk in (28, 30, 32):
        puz = []
        for q in range(10):
            keep = rng.permutation(81)[:kk]
            p = np.full(81, -1, dtype=np.int64)
            p[keep] = ts[q][keep]
            puz.append(p)
        puz = np.array(puz)
        t0 = time.time()
        ans, qans, E = m.search_mfa(puz, chains=12, beta_start=0.3, beta_end=8.0, max_sweeps=300, threshold=thr, noise=1.5, polish_sweeps=100, mix_pol=0.02, seed=7)
        st = {'clues': kk, 'exact': int((ans == ts).all(axis=1).sum()), 'valid': int(sum(1 for a, p in zip(ans, puz) if len(set(a.tolist()[i] for i in [])) >= 0 and (a[p >= 0] == p[p >= 0]).all() and all(len(set(a[r*9:(r+1)*9].tolist())) == 9 for r in range(9)) and all(len(set(a[r::9].tolist())) == 9 for r in range(9)))), 'sec': round(time.time()-t0, 2), 'E_best_mean': round(float(E.min()), 2)}
        out['stages'].append(st)
        res['stages'] = out['stages']
        res['thr'] = thr
        res['ok'] = True
        json.dump(res, open('diag9.json', 'w'), indent=2)
    return out
if not os.path.exists('diag9_done.txt') or os.path.exists('diag9_run.txt'):
    try:
        res = run()
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-2000:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag9.json', 'w'), indent=2)
    open('diag9_done.txt', 'w').write('done')
print('diag9 done')
