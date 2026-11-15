import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
try:
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    tp, ts = d['test_puz'][:10], d['test_sol'][:10]
    mh = CramSudoku(9, 3, seed=5)
    mh.set_hand(1.0)
    res = {}
    t0 = time.time()
    ans, qans, E = mh.search(tp, chains=4, beta_start=1.0, beta_end=4.0, max_sweeps=150, threshold=0.05, noise=1.0, seed=7)
    res['hand_focused'] = {'exact': int((ans == ts).all(axis=1).sum()), 'n': 10, 'sec': round(time.time()-t0, 2), 'E_min': float(E.min()), 'E_max': float(E.max())}
    # quick learned sanity: 200 boards, 10 epochs
    m = CramSudoku(9, 3, seed=1)
    fit = m.train_pl(d['train'][:200], epochs=10, bs=64, lr=0.5)
    thr, em, es = m.fit_stopping_threshold(d['train'][:200])
    t1 = time.time()
    ans2, qans2, E2 = m.search(tp, chains=4, beta_start=1.0, beta_end=4.0, max_sweeps=150, threshold=thr, noise=1.0, seed=7)
    res['learned_200b'] = {'fit_sec': round(fit, 2), 'thr': thr, 'exact': int((ans2 == ts).all(axis=1).sum()), 'quorum_exact': int((qans2 == ts).all(axis=1).sum()), 'sec': round(time.time()-t1, 2), 'E_min': float(E2.min())}
    res['ok'] = True
except Exception:
    res = {'ok': False, 'err': traceback.format_exc()[-1200:]}
res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(res, open('diag2.json', 'w'), indent=2)
print('done')