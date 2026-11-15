import numpy as np, json, time, datetime, os
from cram import CramSudoku
OUT = r'C:\SZL\cram-study'

def save(stage, data):
    data['stage'] = stage
    data['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(data, open(os.path.join(OUT, 'cram_results.json'), 'w'), indent=2, default=str)

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
    # clues respected
    m = puz >= 0
    ok2 = bool((ans[m] == puz[m]).all()) if m.any() else True
    return ok and ok2

t0 = time.time()
save('start', {})
d9 = np.load(os.path.join(OUT, 'sudoku9.npz'))
train9, tp9, ts9 = d9['train'], d9['test_puz'], d9['test_sol']
R = {'data': {'n_train9': int(len(train9)), 'n_test9': int(len(tp9))}}

def eval_config(model, tp, ts, name, **kw):
    t1 = time.time()
    ans, qans, E = model.search(tp, **kw)
    el = time.time() - t1
    exact = float((ans == ts).all(axis=1).mean())
    valid = float(np.mean([check_valid(a, p, model.n, model.box) for a, p in zip(ans, tp)]))
    out = {'exact': exact, 'valid': valid, 'search_sec': round(el, 2)}
    if qans is not None:
        qexact = float((qans == ts).all(axis=1).mean())
        qvalid = float(np.mean([check_valid(a, p, model.n, model.box) for a, p in zip(qans, tp)]))
        out.update({'quorum_exact': qexact, 'quorum_valid': qvalid})
    return out

# ---- learned CRAM 9x9, full data ----
m = CramSudoku(9, 3, seed=1)
fit_s = m.train_pl(train9, epochs=40, bs=64, lr=0.5)
thr, emean, estd = m.fit_stopping_threshold(train9)
R['fit9_full'] = {'fit_sec': round(fit_s, 3), 'n_params': m.n_params(), 'threshold': thr, 'train_E_mean': emean, 'train_E_std': estd}
save('fit9_full', R)

R['cfgA_paper9'] = eval_config(m, tp9, ts9, 'A', chains=4, beta_start=1.0, beta_end=1.0, max_sweeps=200, threshold=thr, noise=1.0, seed=7)
save('cfgA9', R)
R['cfgB_anneal_quorum9'] = eval_config(m, tp9, ts9, 'B', chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=200, threshold=thr, noise=1.0, seed=7)
save('cfgB9', R)

# interpretability: fraction of peer pairs where learned W has its max on the same-digit diagonal
diag_frac = 0.0; tot = 0; diag_cost_mean = 0.0
for a in range(81):
    for k, bp in enumerate(m.peers[a]):
        tot += 1
        ij = m.W[a, k]
        if np.unravel_index(ij.argmax(), ij.shape)[0] == np.unravel_index(ij.argmax(), ij.shape)[1]:
            diag_frac += 1
        diag_cost_mean += float(np.mean([ij[v, v] for v in range(9)]))
R['interpretability9'] = {'peer_pairs': tot, 'argmax_on_diagonal_frac': round(diag_frac/tot, 4), 'mean_diag_cost': round(diag_cost_mean/tot, 4)}
save('interp9', R)

# ---- data efficiency probe: 60 boards ----
m60 = CramSudoku(9, 3, seed=2)
fit_s60 = m60.train_pl(train9[:60], epochs=40, bs=64, lr=0.5)
thr60, _, _ = m60.fit_stopping_threshold(train9[:60])
R['fit9_60boards'] = {'fit_sec': round(fit_s60, 3), 'threshold': thr60}
R['cfgB_60boards'] = eval_config(m60, tp9, ts9, 'B60', chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=200, threshold=thr60, noise=1.0, seed=7)
save('dataeff9', R)

# ---- hand-set saturation probe ----
mh = CramSudoku(9, 3, seed=3)
mh.set_hand(cost=1.0)
thrh, _, _ = mh.fit_stopping_threshold(train9)
R['handset9'] = {'n_params': mh.n_params(), 'threshold': thrh}
R['cfgB_handset9'] = eval_config(mh, tp9, ts9, 'H', chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=200, threshold=thrh, noise=1.0, seed=7)
save('handset9', R)

# ---- 16x16 scale probe ----
try:
    d16 = np.load(os.path.join(OUT, 'sudoku16.npz'))
    tr16, tp16, ts16 = d16['train'], d16['test_puz'], d16['test_sol']
    m16 = CramSudoku(16, 4, seed=4)
    fit16 = m16.train_pl(tr16, epochs=30, bs=32, lr=0.5)
    thr16, _, _ = m16.fit_stopping_threshold(tr16)
    R['fit16'] = {'fit_sec': round(fit16, 3), 'n_params': m16.n_params(), 'threshold': thr16, 'n_test': int(len(tp16))}
    save('fit16', R)
    R['cfgB_16'] = eval_config(m16, tp16, ts16, 'B16', chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thr16, noise=1.0, seed=7)
    mh16 = CramSudoku(16, 4, seed=5)
    mh16.set_hand(cost=1.0)
    thrh16, _, _ = mh16.fit_stopping_threshold(tr16)
    R['cfgB_handset16'] = eval_config(mh16, tp16, ts16, 'H16', chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thrh16, noise=1.0, seed=7)
except Exception as e:
    R['probe16_error'] = repr(e)[:300]
R['total_sec'] = round(time.time()-t0, 1)
save('done', R)
print('done')