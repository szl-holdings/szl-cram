import numpy as np, os, json, time, datetime, traceback
os.chdir(r'C:\SZL\cram-study')
RES = 'cram_results2.json'
def load():
    try: return json.load(open(RES))
    except Exception: return {}
def save(R):
    R['t'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(R, open(RES, 'w'), indent=2, default=str)
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
                if len(set(blk.ravel().tolist())) != n: ok = False
    m = puz >= 0
    return ok and (bool((ans[m] == puz[m]).all()) if m.any() else True)
def get_models():
    from cram import CramSudoku
    d = np.load('sudoku9.npz')
    m = CramSudoku(9, 3, seed=1)
    m60 = CramSudoku(9, 3, seed=2)
    mh = CramSudoku(9, 3, seed=3)
    if os.path.exists('grid2_models.npz'):
        mo = np.load('grid2_models.npz')
        m.W[...] = mo['W']; m.B[...] = mo['B']
        m60.W[...] = mo['W60']; m60.B[...] = mo['B60']
        mh.W[...] = mo['Wh']; mh.B[...] = mo['Bh']
        thrs = json.load(open('grid2_thr.json'))
        return m, m60, mh, thrs['thr'], thrs['thr60'], thrs['thrh'], d
    t0 = time.time()
    fit = m.train_pl(d['train'], epochs=40, bs=64, lr=0.5)
    thr, em, es = m.fit_stopping_threshold(d['train'])
    fit60 = m60.train_pl(d['train'][:60], epochs=40, bs=64, lr=0.5)
    thr60, _, _ = m60.fit_stopping_threshold(d['train'][:60])
    mh.set_hand(cost=1.0)
    thrh, _, _ = mh.fit_stopping_threshold(d['train'])
    np.savez('grid2_models.npz', W=m.W, B=m.B, W60=m60.W, B60=m60.B, Wh=mh.W, Bh=mh.B)
    json.dump({'thr': thr, 'thr60': thr60, 'thrh': thrh, 'fit_sec': fit, 'fit60_sec': fit60}, open('grid2_thr.json', 'w'))
    return m, m60, mh, thr, thr60, thrh, d
MFA = dict(chains=12, beta_start=0.3, beta_end=8.0, max_sweeps=300, noise=1.5, polish_sweeps=150, mix_pol=0.02, quorum=True)
def eval_chunked(m, tp, ts, thr, key, fn='search_mfa', extra=None, chunk=15):
    R = load()
    ch_done = R.get(key, {}).get('chunks_done', [])
    agg = R.get(key, {}).get('agg', {'exact_n': 0, 'valid_n': 0, 'sec': 0.0, 'n': 0})
    kw = dict(MFA) if fn == 'search_mfa' else {}
    kw['threshold'] = thr
    kw['seed'] = 7
    if extra: kw.update(extra)
    n = len(tp)
    for s in range(0, n, chunk):
        ci = s // chunk
        if ci in ch_done: continue
        p = tp[s:s+chunk]; t = ts[s:s+chunk]
        t1 = time.time()
        ans, qans, E = (getattr(m, fn)(p, **kw) if fn == 'search_mfa' else m.search(p, **kw))
        el = time.time() - t1
        ex = int((ans == t).all(axis=1).sum())
        va = int(sum(1 for a, q in zip(ans, p) if check_valid(a, q, m.n, m.box)))
        agg['exact_n'] += ex; agg['valid_n'] += va; agg['sec'] += el; agg['n'] += len(p)
        ch_done.append(ci)
        R[key] = {'chunks_done': ch_done, 'agg': agg}
        R['stage'] = key + '_chunk' + str(ci)
        save(R)
    exact = agg['exact_n'] / agg['n'] if agg['n'] else 0.0
    valid = agg['valid_n'] / agg['n'] if agg['n'] else 0.0
    R[key + '_final'] = {'exact': exact, 'valid': valid, 'exact_n': agg['exact_n'], 'valid_n': agg['valid_n'], 'n': agg['n'], 'search_sec': round(agg['sec'], 2)}
    R['stage'] = key + '_done'
    save(R)
    return R[key + '_final']
def stage_fit(R, d):
    if 'fit9' in R: return
    m, m60, mh, thr, thr60, thrh, _ = get_models()
    th = json.load(open('grid2_thr.json'))
    R['fit9'] = {'fit_sec': round(th['fit_sec'], 3), 'n_params': m.n_params(), 'threshold': thr, 'train_thr_note': '40-epoch fit per train_eval protocol', 'threshold60': thr60, 'thresholdh': thrh}
    R['data'] = {'n_train9': int(len(d['train'])), 'n_test9': int(len(d['test_puz']))}
    R['stage'] = 'fit9'
    save(R)
def stage_cfgA(R, d):
    if 'cfgA9_final' in R: return
    old = json.load(open('cram_results.json'))
    R['cfgA9_final'] = old['cfgA_paper9']
    R['stage'] = 'cfgA9_copied'
    save(R)
def stage_interp(R, d):
    if 'interp9' in R: return
    m, m60, mh, thr, _, _, _ = get_models()
    diag_frac = 0.0; tot = 0; diag_cost_mean = 0.0
    for a in range(81):
        for k, bp in enumerate(m.peers[a]):
            tot += 1
            ij = m.W[a, k]
            if np.unravel_index(ij.argmax(), ij.shape)[0] == np.unravel_index(ij.argmax(), ij.shape)[1]:
                diag_frac += 1
            diag_cost_mean += float(np.mean([ij[v, v] for v in range(9)]))
    offd = []
    for a in range(81):
        for k in range(len(m.peers[a])):
            ij = m.W[a, k]
            offv = ij.copy(); np.fill_diagonal(offv, np.nan)
            offd.append(float(np.nanmin(offv)))
    R['interp9'] = {'peer_pairs': tot, 'argmax_on_diagonal_frac': round(diag_frac/tot, 4), 'mean_diag_cost': round(diag_cost_mean/tot, 4), 'min_offdiag_cost': round(float(np.min(offd)), 4), 'note': 'min_offdiag_cost < 0 implies off-diagonal reward entries (decoy-minima substrate)'}
    R['stage'] = 'interp9'
    save(R)
def stage_60(R, d):
    if 'cfgB60_final' in R: return
    m, m60, mh, thr, thr60, thrh, _ = get_models()
    eval_chunked(m60, d['test_puz'], d['test_sol'], thr60, 'cfgB60')
def stage_hand(R, d):
    if 'handset9_gibbs_final' in R: return
    m, m60, mh, thr, thr60, thrh, _ = get_models()
    eval_chunked(mh, d['test_puz'], d['test_sol'], 0.05, 'handset9_mfa', extra={'pol_threshold': 0.05})
    eval_chunked(mh, d['test_puz'], d['test_sol'], thrh, 'handset9_gibbs', fn='search', extra=dict(chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=200, noise=1.0))
def stage_16(R, d):
    if 'p16' in R and R['p16'].get('fit_done'): return
    try:
        d16 = np.load('sudoku16.npz')
        tr16, tp16, ts16 = d16['train'], d16['test_puz'], d16['test_sol']
        from cram import CramSudoku
        m16 = CramSudoku(16, 4, seed=4)
        t0 = time.time()
        fit16 = m16.train_pl(tr16, epochs=30, bs=32, lr=0.5)
        thr16, _, _ = m16.fit_stopping_threshold(tr16)
        R['p16'] = {'fit_sec': round(fit16, 3), 'n_params': m16.n_params(), 'threshold': thr16, 'n_test': int(len(tp16)), 'fit_done': True}
        R['stage'] = 'p16_fit'
        save(R)
        ans, qans, E = m16.search(tp16, chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thr16, noise=1.0, seed=7)
        R['p16_learned'] = {'exact': float((ans == ts16).all(axis=1).mean()), 'valid': float(np.mean([check_valid(a, p, 16, 4) for a, p in zip(ans, tp16)]))}
        R['stage'] = 'p16_learned'
        save(R)
        mh16 = CramSudoku(16, 4, seed=5)
        mh16.set_hand(cost=1.0)
        thrh16, _, _ = mh16.fit_stopping_threshold(tr16)
        ans2, _, _ = mh16.search(tp16, chains=4, beta_start=0.7, beta_end=4.0, max_sweeps=400, threshold=thrh16, noise=1.0, seed=7)
        R['p16_handset'] = {'exact': float((ans2 == ts16).all(axis=1).mean()), 'valid': float(np.mean([check_valid(a, p, 16, 4) for a, p in zip(ans2, tp16)])), 'threshold': thrh16}
        R['stage'] = 'p16_done'
    except Exception as e:
        R['p16'] = {'error': repr(e)[:300], 'fit_done': False}
    save(R)
if __name__ == '__main__':
    R = load()
    d = np.load('sudoku9.npz')
    m, m60, mh, thr, thr60, thrh, dd = get_models()
    R['data'] = {'n_train9': int(len(dd['train'])), 'n_test9': int(len(dd['test_puz']))}
    save(R)
    for nm, f in [('fit9', lambda: stage_fit(R, dd)), ('cfgA9', lambda: stage_cfgA(R, dd)), ('cfgB9', lambda: eval_chunked(m, dd['test_puz'], dd['test_sol'], thr, 'cfgB9')), ('interp9', lambda: stage_interp(R, dd)), ('cfgB60', lambda: stage_60(R, dd)), ('handset', lambda: stage_hand(R, dd)), ('p16', lambda: stage_16(R, dd))]:
        if nm == 'cfgA9' and 'cfgA9_final' in R: continue
        try:
            f()
            R = load()
        except Exception:
            R = load()
            R['err_' + nm] = traceback.format_exc()[-800:]
            R['stage'] = 'err_' + nm
            save(R)
    R = load()
    R['stage'] = 'done'
    R['total_note'] = 'grid2 resume-aware; deaths only lose the current chunk'
    save(R)
    print('grid2 done')
