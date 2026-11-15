import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
if not os.path.exists('diag5a.json'):
    try:
        from cram import CramSudoku
        d = np.load('sudoku9.npz')
        tp, ts = d['test_puz'][:10], d['test_sol'][:10]
        res = {}
        mh = CramSudoku(9, 3, seed=5)
        mh.set_hand(1.0)
        t0 = time.time()
        ans, qans, E = mh.search_mfa(tp, chains=12, beta_start=0.3, beta_end=8.0, max_sweeps=300, threshold=0.05, noise=1.5, polish_sweeps=150, pol_threshold=0.05, mix_pol=0.02, seed=7)
        res['hand_mfa2'] = {'exact': int((ans == ts).all(axis=1).sum()), 'quorum_exact': int((qans == ts).all(axis=1).sum()), 'n': 10, 'sec': round(time.time()-t0, 2), 'E_max': float(E.max())}
        res['ok'] = True
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-1500:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag5a.json', 'w'), indent=2)
print('diag5a done')
