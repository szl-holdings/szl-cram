import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
if not os.path.exists('diag5b.json'):
    try:
        from cram import CramSudoku
        d = np.load('sudoku9.npz')
        tp, ts = d['test_puz'][:10], d['test_sol'][:10]
        res = {}
        m = CramSudoku(9, 3, seed=1)
        t0 = time.time()
        fit = m.train_pl(d['train'], epochs=100, bs=64, lr=0.5)
        thr, em, es = m.fit_stopping_threshold(d['train'])
        solE_test = m.solution_energy(ts)
        t1 = time.time()
        ans2, qans2, E2 = m.search_mfa(tp, chains=12, beta_start=0.3, beta_end=8.0, max_sweeps=300, threshold=thr, noise=1.5, polish_sweeps=150, mix_pol=0.02, seed=7)
        best_p = E2.reshape(10, 12).min(axis=1)
        below = int((best_p < solE_test - 0.5).sum())
        res['learned100_mfa2'] = {'fit_sec': round(fit, 2), 'thr': thr, 'solE_test_mean': float(solE_test.mean()), 'exact': int((ans2 == ts).all(axis=1).sum()), 'quorum_exact': int((qans2 == ts).all(axis=1).sum()), 'sec': round(time.time()-t1, 2), 'E_best_mean': float(best_p.mean()), 'below_solutionE': below}
        res['ok'] = True
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-1500:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag5b.json', 'w'), indent=2)
# merge both stages into diag5.json for grid_watch
if os.path.exists('diag5a.json') and os.path.exists('diag5b.json') and not os.path.exists('diag5.json'):
    a = json.load(open('diag5a.json')); b = json.load(open('diag5b.json'))
    merged = {'hand_mfa2': a.get('hand_mfa2'), 'learned100_mfa2': b.get('learned100_mfa2'), 'ok': bool(a.get('ok') and b.get('ok')), 't': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    json.dump(merged, open('diag5.json', 'w'), indent=2)
print('diag5b done')
