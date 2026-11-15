import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
try:
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    tp, ts = d['test_puz'][:10], d['test_sol'][:10]
    res = {}
    mh = CramSudoku(9, 3, seed=5)
    mh.set_hand(1.0)
    t0 = time.time()
    ans, qans, E = mh.search_mfa(tp, chains=8, beta_start=0.5, beta_end=6.0, max_sweeps=250, threshold=0.05, noise=1.0, polish_sweeps=60, seed=7)
    res['hand_mfa'] = {'exact': int((ans == ts).all(axis=1).sum()), 'quorum_exact': int((qans == ts).all(axis=1).sum()), 'n': 10, 'sec': round(time.time()-t0, 2), 'E_max': float(E.max())}
    m = CramSudoku(9, 3, seed=1)
    fit = m.train_pl(d['train'], epochs=40, bs=64, lr=0.5)
    thr, em, es = m.fit_stopping_threshold(d['train'])
    solE_test = m.solution_energy(ts)
    t1 = time.time()
    ans2, qans2, E2 = m.search_mfa(tp, chains=8, beta_start=0.5, beta_end=6.0, max_sweeps=250, threshold=thr, noise=1.0, polish_sweeps=80, seed=7)
    best_p = E2.reshape(10, 8).min(axis=1)
    below = int((best_p < solE_test - 0.5).sum())
    res['learned600_mfa'] = {'fit_sec': round(fit, 2), 'thr': thr, 'solE_test_mean': float(solE_test.mean()), 'exact': int((ans2 == ts).all(axis=1).sum()), 'quorum_exact': int((qans2 == ts).all(axis=1).sum()), 'sec': round(time.time()-t1, 2), 'E_best_mean': float(best_p.mean()), 'below_solutionE': below}
    res['ok'] = True
except Exception:
    res = {'ok': False, 'err': traceback.format_exc()[-1500:]}
res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(res, open('diag4.json', 'w'), indent=2)
print('done')