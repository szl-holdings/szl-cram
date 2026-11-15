"""16x16-only rerun: launched when gen_sudoku.py finishes sudoku16.npz. Saves cram_results16.json."""
import numpy as np, json, time, datetime, os
from cram import CramSudoku
OUT = r'C:\SZL\cram-study'
os.chdir(OUT)
d16 = np.load('sudoku16.npz')
tr16, tp16, ts16 = d16['train'], d16['test_puz'], d16['test_sol']
R = {'data': {'n_train16': int(len(tr16)), 'n_test16': int(len(tp16))}}
def check_valid(ans, puz, n, box):
    ok = True
    for r in range(n):
        row = ans[r*n:(r+1)*n]
        col = ans[r::n]
        if len(set(row.tolist())) != n or len(set(col.tolist())) != n:
            ok = False; break
    if ok:
        for br in range(0, n, box):
            for bc in range(0, n, box):
                blk = ans[(br+np.arange(box))[:, None]*n + (bc+np.arange(box))[None, :]]
                if len(set(blk.ravel().tolist())) != n:
                    ok = False
    m = puz >= 0
    ok2 = bool((ans[m] == puz[m]).all()) if m.any() else True
    return ok and ok2
def save(stage):
    R['stage'] = stage; R['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(R, open(os.path.join(OUT, 'cram_results16.json'), 'w'), indent=2, default=str)
t0 = time.time()
save('start16')
m16 = CramSudoku(16, 4, seed=4)
fit16 = m16.train_pl(tr16, epochs=30, bs=32, lr=0.5)
thr16, _, _ = m16.fit_stopping_threshold(tr16)
R['fit16'] = {'fit_sec': round(fit16, 3), 'n_params': m16.n_params(), 'threshold': thr16, 'n_test': int(len(tp16))}
save('fit16')
t1 = time.time(); ans, qans, E = m16.search(tp16, chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thr16, noise=1.0, seed=7)
R['cfgB_16'] = {'exact': float((ans == ts16).all(axis=1).mean()), 'valid': float(np.mean([check_valid(a, p, 16, 4) for a, p in zip(ans, tp16)])), 'search_sec': round(time.time()-t1, 2)}
save('cfgB16')
mh16 = CramSudoku(16, 4, seed=5)
mh16.set_hand(cost=1.0)
thrh16, _, _ = mh16.fit_stopping_threshold(tr16)
t2 = time.time(); ans2, _, _ = mh16.search(tp16, chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thrh16, noise=1.0, seed=7)
R['cfgB_handset16'] = {'exact': float((ans2 == ts16).all(axis=1).mean()), 'valid': float(np.mean([check_valid(a, p, 16, 4) for a, p in zip(ans2, tp16)])), 'search_sec': round(time.time()-t2, 2), 'threshold': thrh16}
R['total_sec'] = round(time.time()-t0, 1)
save('done16')
print('done16')
