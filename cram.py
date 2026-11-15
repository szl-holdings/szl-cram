import numpy as np

def peers_of(n, box):
    P = []
    for r in range(n):
        for c in range(n):
            s = set()
            for k in range(n):
                s.add(r*n+k); s.add(k*n+c)
            br, bc = (r//box)*box, (c//box)*box
            for rr in range(br, br+box):
                for cc in range(bc, bc+box):
                    s.add(rr*n+cc)
            s.discard(r*n+c)
            P.append(sorted(s))
    return P

class CramSudoku:
    'CRAM energy (arXiv:2611.17309 Eq.1) for n x n Sudoku. W[i,k] is the C x C cost matrix between cell i and its k-th peer (peers-only storage, matches the paper 263K param count for 9x9).'
    def __init__(self, n, box, seed=0):
        self.n = n; self.box = box; self.C = n; self.N = n*n
        self.rng = np.random.default_rng(seed)
        self.peers = peers_of(n, box)
        self.K = len(self.peers[0])
        self.PI = np.array(self.peers)
        self.peer_pos = [{p: k for k, p in enumerate(pl)} for pl in self.peers]
        self.W = np.zeros((self.N, self.K, self.C, self.C), dtype=np.float32)
        self.B = np.zeros((self.N, self.C), dtype=np.float32)
        self.beta = 1.0
        self._an = np.arange(self.N)[None, :, None]
        self._ak = np.arange(self.K)[None, None, :]

    def energy(self, P):
        'Beliefs (..., N, C) -> energy per board (Eq.1).'
        P = np.asarray(P, dtype=np.float32)
        ent = np.sum(P * np.log(np.clip(P, 1e-9, 1)), axis=-1)
        E_ent = ent.sum(axis=-1) / self.beta
        pj = P[..., self.PI, :]
        pij = np.einsum('...nc,nkcb,...nkb->...n', P, self.W, pj)
        E_int = 0.5 * pij.sum(axis=-1)
        E_bias = np.einsum('...nc,nc->...', P, self.B)
        return E_ent + E_int + E_bias

    def solution_energy(self, sols):
        'One-hot solutions (n_boards, N) -> energy.'
        sols = np.asarray(sols)
        E = np.zeros(len(sols), dtype=np.float32)
        eye = np.eye(self.C, dtype=np.float32)
        for a in range(self.N):
            for k, b in enumerate(self.peers[a]):
                if b > a:
                    k2 = self.peer_pos[b][a]
                    M = 0.5 * (self.W[a, k] + self.W[b, k2].T)
                    E += np.einsum('bp,pq,bq->b', eye[sols[:, a]], M, eye[sols[:, b]])
        for i in range(self.N):
            E += self.B[i, sols[:, i]]
        return E

    def pl_loss_grad(self, boards, aug=True):
        'Pseudolikelihood: for each cell, cost of each value given labeled peers. No solver, no BPTT.'
        Bn = len(boards)
        X = np.asarray(boards, dtype=np.int64)
        if aug:
            perm = self.rng.permutation(self.C)
            X = perm[X]
        xj = X[:, self.PI]                                   # (B, N, K) peer values
        term = self.W[self._an, self._ak, :, xj]           # (B, N, K, C) gather W[n,k,a,x_j]
        U = term.sum(axis=2) + self.B[None]                  # (B, N, C)
        Pm = -U
        Pm = Pm - Pm.max(axis=-1, keepdims=True)
        p = np.exp(Pm); p /= p.sum(axis=-1, keepdims=True)
        loss = -(Pm[np.arange(Bn)[:, None], np.arange(self.N)[None, :], X]).sum() / Bn
        dU = (p - np.eye(self.C, dtype=np.float32)[X]) / Bn  # (B, N, C)
        xj_onehot = (xj[..., None] == np.arange(self.C)[None, None, None, :]).astype(np.float32)
        gW = np.einsum('bna,bnkc->nkac', dU, xj_onehot)       # (N, K, C, C)
        gB = dU.sum(axis=0)
        return loss, gW, gB

    def train_pl(self, boards, epochs=40, bs=64, lr=0.5, mom=0.9, verbose=False):
        import time
        vX = np.zeros_like(self.W); vB = np.zeros_like(self.B)
        steps = max(1, (len(boards)//bs) * epochs)
        step = 0
        t0 = time.time()
        idx = np.arange(len(boards))
        for ep in range(epochs):
            self.rng.shuffle(idx)
            for s in range(0, len(boards), bs):
                batch = boards[idx[s:s+bs]]
                lr_t = 0.5*lr*(1+np.cos(np.pi*step/steps)) + 1e-3
                loss, gW, gB = self.pl_loss_grad(batch)
                vX = mom*vX + lr_t*gW; self.W += vX
                vB = mom*vB + lr_t*gB; self.B += vB
                step += 1
            if verbose and ep % 10 == 0:
                print(ep, round(float(loss), 4))
        return time.time() - t0

    def fit_stopping_threshold(self, train_boards, tol=0.05):
        E = self.solution_energy(np.asarray(train_boards))
        return float(E.max() + tol), float(E.mean()), float(E.std())

    def search(self, puzzles, chains=4, beta_start=1.0, beta_end=4.0, max_sweeps=200, mix_p=0.0, mix_until=0.7,
               threshold=None, noise=1.0, seed=0, quorum=True, init_X=None):
        'Focused Gibbs relaxation (arXiv:2611.17309): clues clamped; a cell is updated only when its CURRENT value is conflicted (local cost > 0); candidate drawn from softmax(-beta*U). Chains in parallel; stop on energy threshold or global fixed point.'
        P = np.asarray(puzzles)
        Q = len(P); N = self.N; C = self.C
        rng = np.random.default_rng(seed)
        Btot = Q * chains
        clue_mask = (P >= 0)
        if init_X is None:
            init_X = rng.integers(0, C, size=P.shape)
        X = np.where(clue_mask, P, init_X)
        Xb = np.repeat(X, chains, axis=0).astype(np.int64)
        clb = np.repeat(clue_mask, chains, axis=0)
        beliefs = np.eye(C, dtype=np.float32)[Xb]
        best_E = np.full(Btot, np.inf, dtype=np.float32)
        best_X = Xb.copy()
        solved_flag = np.zeros(Btot, dtype=bool)
        beta_sched = np.linspace(beta_start, beta_end, max_sweeps)
        ar = np.arange(Btot)
        for sweep in range(max_sweeps):
            beta = beta_sched[sweep]
            noise_t = noise
            order = rng.permutation(N)
            updated = np.zeros(Btot, dtype=bool)
            for i in order:
                if clb[:, i].all():
                    continue
                pj = beliefs[:, self.PI[i], :]
                U = np.einsum('bkc,kac->ba', pj, self.W[i]) + self.B[i]
                cur_cost = U[ar, Xb[:, i]]
                upd = (~clb[:, i]) & (cur_cost > 1e-6)
                if mix_p > 0.0 and sweep < mix_until * max_sweeps:
                    upd = upd | ((~clb[:, i]) & (rng.random(Btot) < mix_p))
                if not upd.any():
                    continue
                updated[upd] = True
                logits = -beta * U + noise_t * rng.gumbel(size=U.shape)
                logits = logits - logits.max(axis=-1, keepdims=True)
                pn = np.exp(logits); pn /= pn.sum(axis=-1, keepdims=True)
                newv = (pn.cumsum(axis=-1) < rng.random(size=(Btot, 1))).sum(axis=-1)
                newv = np.clip(newv, 0, C-1)
                beliefs[upd, i, :] = 0.0
                beliefs[upd, i, newv[upd]] = 1.0
                Xb[upd, i] = newv[upd]
            E = self.energy(beliefs)
            better = E < best_E
            best_E = np.where(better, E, best_E)
            best_X = np.where(better[:, None], Xb, best_X)
            if threshold is not None:
                solved_flag |= (E <= threshold)
                if solved_flag.all():
                    break
            # restart chains stuck at non-solution fixed points (failed search; counted in compute)
            thr_eff = threshold if threshold is not None else -np.inf
            stuck = (~updated) & (E > thr_eff)
            if stuck.any():
                rs = np.where(stuck)[0]
                free = ~clb[rs]
                newv2 = rng.integers(0, C, size=free.shape)
                Xb[rs] = np.where(free, newv2, Xb[rs])
                beliefs[rs] = np.eye(C, dtype=np.float32)[Xb[rs]]
        if not np.isfinite(best_E).all():
            E = self.energy(beliefs)
            better = E < best_E
            best_E = np.where(better, E, best_E)
            best_X = np.where(better[:, None], Xb, best_X)
        best_E_q = best_E.reshape(Q, chains)
        best_X_q = best_X.reshape(Q, chains, N)
        answers = best_X_q[np.arange(Q), best_E_q.argmin(axis=1)]
        quorum_answers = None
        if quorum:
            k = max(1, chains // 2)
            low = best_E_q.argsort(axis=1)[:, :k]
            cand = best_X_q[np.arange(Q)[:, None], low]
            votes = np.zeros((Q, N, C), dtype=np.int32)
            for ki in range(k):
                np.add.at(votes, (np.arange(Q)[:, None], np.arange(N)[None, :], cand[:, ki, :]), 1)
            quorum_answers = votes.argmax(axis=-1)
        return answers, quorum_answers, best_E

    def set_hand(self, cost=1.0):
        'Hand-set energy from Sudoku rules: same digit in a peer pair costs 1. One unique weight value.'
        for i in range(self.N):
            for k in range(self.K):
                for a in range(self.C):
                    self.W[i, k, a, a] = cost
        self.B[:] = 0.0

    def search_mfa(self, puzzles, chains=8, beta_start=0.5, beta_end=6.0, max_sweeps=200,
               threshold=None, noise=1.0, seed=0, polish_sweeps=60, quorum=True, pol_threshold=None, mix_pol=0.0):
        'Mean-field annealing (Hopfield-Tank descent, arXiv:2611.17309): soft beliefs from uniform init, clues clamped, annealed beta+noise, then focused-Gibbs polish.'
        P = np.asarray(puzzles)
        Q = len(P); N = self.N; C = self.C
        rng = np.random.default_rng(seed)
        Btot = Q * chains
        clue_mask = (P >= 0)
        clb = np.repeat(clue_mask, chains, axis=0)
        Pq = np.repeat(P, chains, axis=0)
        onehot = np.full((Btot, N, C), 1.0 / C, dtype=np.float32)
        cm = clb
        onehot[cm, Pq[cm]] = 1.0
        Psoft = onehot
        best_E = np.full(Btot, np.inf, dtype=np.float32)
        best_X = np.zeros((Btot, N), dtype=np.int64)
        beta_sched = np.linspace(beta_start, beta_end, max_sweeps)
        for sweep in range(max_sweeps):
            beta = beta_sched[sweep]
            nz = noise / max(beta, 0.1)
            for i in range(N):
                free = ~clb[:, i]
                if not free.any():
                    continue
                U = np.einsum('bkc,kac->ba', Psoft[:, self.PI[i], :], self.W[i]) + self.B[i]
                lp = -beta * U + nz * rng.gumbel(size=U.shape)
                lp = lp - lp.max(axis=-1, keepdims=True)
                pn = np.exp(lp); pn /= pn.sum(axis=-1, keepdims=True)
                Psoft[free, i, :] = pn[free]
            if sweep % 10 == 9 or sweep == max_sweeps - 1:
                X = Psoft.argmax(axis=-1)
                Xh = np.where(clb, Pq, X)
                beliefs = np.eye(C, dtype=np.float32)[Xh]
                E = self.energy(beliefs)
                better = E < best_E
                best_E = np.where(better, E, best_E)
                best_X = np.where(better[:, None], Xh, best_X)
                if threshold is not None and sweep >= 0.7 * max_sweeps and (best_E <= threshold).all():
                    break
        # focused-Gibbs polish per chain from its MFA state
        pol_E = np.full(Btot, np.inf, dtype=np.float32)
        pol_X = best_X.copy()
        for c in range(Btot):
            a, _, pe = self.search(P[:1], chains=1, init_X=best_X[c][None, :],
                                   beta_start=beta_end, beta_end=beta_end,
                                   max_sweeps=polish_sweeps, mix_p=mix_pol,
                                   threshold=(threshold if pol_threshold is None else pol_threshold), noise=noise*0.3,
                                   seed=int(rng.integers(1, 10**9)), quorum=False)
            pol_X[c] = a[0]
            pol_E[c] = pe[0]
        fin_E = np.minimum(best_E, pol_E)
        fin_X = np.where((pol_E < best_E)[:, None], pol_X, best_X)
        best_E_q = fin_E.reshape(Q, chains)
        best_X_q = fin_X.reshape(Q, chains, N)
        answers = best_X_q[np.arange(Q), best_E_q.argmin(axis=1)]
        quorum_answers = None
        if quorum:
            k = max(1, chains // 2)
            low = best_E_q.argsort(axis=1)[:, :k]
            cand = best_X_q[np.arange(Q)[:, None], low]
            votes = np.zeros((Q, N, C), dtype=np.int32)
            for ki in range(k):
                np.add.at(votes, (np.arange(Q)[:, None], np.arange(N)[None, :], cand[:, ki, :]), 1)
            quorum_answers = votes.argmax(axis=-1)
        return answers, quorum_answers, fin_E

    def n_params(self):
        return int((self.W != 0).sum() + (self.B != 0).sum())