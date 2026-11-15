import numpy as np, os, json, datetime, time, traceback
os.chdir(r'C:\SZL\cram-study')
def run():
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    tp, ts = d['test_puz'][:10], d['test_sol'][:10]
    m = CramSudoku(9, 3, seed=1)
    t0 = time.time()
    fit = m.train_pl(d['train'], epochs=100, bs=64, lr=0.5)
    thr, em, es = m.fit_stopping_threshold(d['train'])
    np.savez('model100.npz', W=m.W, B=m.B)
    solE = m.solution_energy(ts)
    # MFA v3 knobs (validated cost from diag6)
    t1 = time.time()
    ans, qans, E = m.search_mfa(tp, chains=16, beta_start=0.3, beta_end=12.0, max_sweeps=800, threshold=thr, noise=1.5, polish_sweeps=300, mix_pol=0.03, seed=7)
    mfa_sec = time.time() - t1
    # state energy (integers) via one-hot
    def state_E(x):
        beliefs = np.eye(m.C, dtype=np.float32)[x]
        return float(m.energy(beliefs))
    # per-cell conflict cost: sum of W[i,k][x_i, x_peer]
    def cell_costs(x):
        cc = np.zeros(m.N)
        for i in range(m.N):
            for k, b in enumerate(m.peers[i]):
                cc[i] += m.W[i, k, x[i], x[b]]
        return cc
    rng = np.random.default_rng(123)
    best = ans.copy(); bestE = np.array([state_E(x) for x in best])
    traj = []
    ROUNDS, KICK = 40, 6
    t2 = time.time()
    for rd in range(ROUNDS):
        for q in range(len(tp)):
            x = best[q].copy()
            cc = cell_costs(x)
            free = tp[q] < 0
            cc[~free] = -1e9
            kick = rng.choice(m.N, size=KICK, replace=False, p=None) if False else np.argsort(cc)[-KICK:]
            kick = [int(i) for i in kick if free[i]]
            if not kick: continue
            x[kick] = rng.integers(0, m.C, size=len(kick))
            a, _, pe = m.search(tp[q:q+1], chains=1, init_X=x[None, :], beta_start=6.0, beta_end=6.0, max_sweeps=80, mix_p=0.05, threshold=None, noise=0.5, seed=int(rng.integers(1, 10**9)), quorum=False)
            e_new = state_E(a[0])
            if e_new < bestE[q]:
                bestE[q] = e_new; best[q] = a[0].copy()
        traj.append({'round': rd, 'exact': int((best == ts).all(axis=1).sum()), 'gap_mean': round(float((bestE - solE).mean()), 2)})
    ils_sec = time.time() - t2
    return {'fit_sec': round(fit, 2), 'thr': thr, 'solE_per_puzzle': [round(float(x), 2) for x in solE], 'mfa_sec': round(mfa_sec, 2), 'mfa_exact': int((ans == ts).all(axis=1).sum()), 'ils_sec': round(ils_sec, 2), 'rounds': ROUNDS, 'kick': KICK, 'exact': int((best == ts).all(axis=1).sum()), 'E_best_per_puzzle': [round(float(x), 2) for x in bestE], 'gap_per_puzzle': [round(float(b - s), 2) for b, s in zip(bestE, solE)], 'traj': traj, 'ok': True}
if not os.path.exists('diag7.json') or os.path.exists('diag7_run.txt'):
    try:
        res = run()
    except Exception:
        res = {'ok': False, 'err': traceback.format_exc()[-2000:]}
    res['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(res, open('diag7.json', 'w'), indent=2)
print('diag7 done')
