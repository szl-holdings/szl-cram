import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
if not os.path.exists('diag6.json'):
    try:
        from cram import CramSudoku
        d = np.load('sudoku9.npz')
        tp, ts = d['test_puz'][:10], d['test_sol'][:10]
        res = {}
        m = CramSudoku(9, 3, seed=1)
        t0 = time.time()
        fit = m.train_pl(d['train'], epochs=100, bs=64, lr=0.5)
        thr, em, es = m.fit_stopping_threshold(d['train'])
        solE = m.solution_energy(ts)
        t1 = time.time()
        ans2, qans2, E2 = m.search_mfa(tp, chains=16, beta_start=0.3, beta_end=12.0, max_sweeps=800, threshold=thr, noise=1.5, polish_sweeps=300, mix_pol=0.03, seed=7)
        best_p = E2.reshape(10, 16).min(axis=1)
        res['learned100_mfa3knobs'] = {'fit_sec': round(fit, 2), 'thr': thr, 'solE_test_mean': float(solE.mean()), 'exact': int((ans2 == ts).all(axis=1).sum()), 'quorum_exact': int((qans2 == ts).all(axis=1).sum()), 'sec': round(time.time()-t1, 2), 'E_best_per_puzzle': [round(float(x), 2) for x in best_p], 'solE_per_puzzle': [round(float(x), 2) for x in solE], 'gap_per_puzzle': [round(float(b - s), 2) for b, s in zip(best_p, solE)], 'near_miss_within_1': int((best_p - solE < 1.0).sum()), 'near_miss_within_3': int((best_p - solE < 3.0).sum())}
        res['ok'] = True
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-1500:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag6.json', 'w'), indent=2)
print('diag6 done')
