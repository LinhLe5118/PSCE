"""Optional FP64 array backends. No oracle or learner-data dependencies."""
import numpy as np


def resolve_backend(name, states, *, threshold=256):
    # Local RTX heuristic; remeasure on the deployment GPU before tuning.
    return ('cuda' if states >= threshold else 'numpy') if name == 'hybrid' else name


def array_backend(name):
    if name == 'numpy':
        return np
    if name != 'cuda':
        raise ValueError('Backend must be numpy or cuda')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('CUDA requested but no GPU is visible')
    return cp


def host(value, xp):
    return value if xp is np else xp.asnumpy(value)


def q_sequence(p, r, beta, supported, gamma, k_max, *, backend):
    """Same restricted Bellman sequence; keep every depth on device until done."""
    xp = array_backend(resolve_backend(backend,beta.shape[0]))
    mask = np.zeros(beta.shape, dtype=bool)
    for state, actions in enumerate(supported):
        mask[state, list(actions)] = True
    mask = xp.asarray(mask)
    p, r, beta = (xp.asarray(a, dtype=xp.float64) for a in (p, r, beta))
    states = beta.shape[0]
    p_beta = xp.einsum('sa,san->sn', beta, p)
    r_beta = xp.einsum('sa,sa->s', beta, r)
    try:
        v = xp.linalg.solve(xp.eye(states) - gamma*p_beta, r_beta)
    except xp.linalg.LinAlgError:
        return None
    q = xp.where(mask, r + gamma*xp.einsum('san,n->sa', p, v), xp.nan)
    sequence = xp.empty((k_max+1, *beta.shape), dtype=xp.float64)
    sequence[0] = q
    active = mask.any(axis=1)
    for depth in range(1, k_max+1):
        maxima = xp.where(active, xp.max(xp.where(mask, q, -xp.inf), axis=1), 0)
        q = xp.where(mask, r + gamma*xp.einsum('san,n->sa', p, maxima), xp.nan)
        sequence[depth] = q
    # One result transfer, rather than synchronizing on each state/depth.
    values = host(sequence, xp)
    if not np.isfinite(host(v, xp)).all() or not np.isfinite(values[:, host(mask, xp)]).all():
        return None
    return list(values)


def cfqi_models(models, gamma, iterations=100, tolerance=1e-10, *, backend='numpy'):
    """Batch equal-shape empirical models, freeze each job at its own stop.

    Caller limits the batch size before stacking dense transition models.
    Return host Qs, iteration counts and residuals; policy extraction is shared
    with the reference learner. No reduced precision or changed update budget.
    """
    if not models or iterations <= 0 or not 0 <= gamma < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid CFQI batch or convergence parameters')
    xp = array_backend(resolve_backend(backend, models[0][1].shape[0], threshold=512))
    p, r, counts, terminal = (xp.asarray(np.stack(parts)) for parts in zip(*models))
    support = counts > 0
    active = support.any(axis=2) & ~terminal
    q = xp.zeros_like(r, dtype=xp.float64)
    running = xp.ones(len(models), dtype=bool)
    used = xp.zeros(len(models), dtype=xp.int64)
    residual = xp.zeros(len(models), dtype=xp.float64)
    for step in range(iterations):
        maxima = xp.where(active, xp.max(xp.where(support, q, -xp.inf), axis=2), 0)
        updated = xp.where(support, r + gamma*xp.einsum('bsan,bn->bsa', p, maxima), 0)
        delta = xp.max(xp.abs(updated-q), axis=(1,2))
        q = xp.where(running[:,None,None], updated, q)
        used = xp.where(running, step+1, used)
        residual = xp.where(running, delta, residual)
        running &= delta > tolerance
        # No host synchronization per iteration. Frozen jobs stay unchanged.
    return host(q,xp), host(used,xp), host(residual,xp)


def iql_models(models, gamma, iterations=100, tolerance=1e-10, *,
               expectile=.7, inverse_temperature=3., backend='numpy'):
    """Exact finite-support expectiles across a batch, with independent stops."""
    if not models or iterations <= 0 or not 0 <= gamma < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid IQL batch or convergence parameters')
    if not 0 < expectile < 1 or not np.isfinite(inverse_temperature) or inverse_temperature <= 0:
        raise ValueError('Invalid IQL extraction parameters')
    xp=array_backend(resolve_backend(backend, models[0][1].shape[0], threshold=512))
    p,r,counts,terminal=(xp.asarray(np.stack(parts)) for parts in zip(*models))
    support=counts>0
    totals=counts.sum(axis=2,keepdims=True)
    behavior=counts/xp.where(totals>0,totals,1)
    active=support.any(axis=2)&~terminal
    values=xp.zeros_like(r[:,:,0],dtype=xp.float64)
    running=xp.ones(len(models),dtype=bool)
    used=xp.zeros(len(models),dtype=xp.int64)
    residual=xp.zeros(len(models),dtype=xp.float64)
    actions=counts.shape[2]
    indices=xp.arange(actions)
    all_solved=xp.ones_like(active, dtype=bool)
    for step in range(iterations):
        q=r+gamma*xp.einsum('bsan,bn->bsa',p,values)
        order=xp.argsort(q,axis=2)
        sorted_q=xp.take_along_axis(q,order,axis=2)
        weights=xp.take_along_axis(behavior,order,axis=2)
        updated=xp.zeros_like(values)
        found=~active
        for split in range(actions+1):
            effective=weights*xp.where(indices<split,1-expectile,expectile)
            denominator=effective.sum(axis=2)
            candidate=(effective*sorted_q).sum(axis=2)/xp.where(denominator>0,denominator,1)
            lower=sorted_q[:,:,split-1] if split else -xp.inf
            upper=sorted_q[:,:,split] if split<actions else xp.inf
            valid=active & ~found & (lower-1e-15<=candidate) & (candidate<=upper+1e-15)
            updated=xp.where(valid,candidate,updated)
            found |= valid
        # Validation once on return avoids synchronizing each GPU split/update.
        all_solved &= found | ~running[:, None]
        delta=xp.max(xp.abs(updated-values),axis=1)
        values=xp.where(running[:,None],updated,values)
        used=xp.where(running,step+1,used)
        residual=xp.where(running,delta,residual)
        running &= delta>tolerance
    if not host(all_solved,xp).all():
        raise RuntimeError('Failed to solve batch finite-support expectile')
    q=r+gamma*xp.einsum('bsan,bn->bsa',p,values)
    logits=inverse_temperature*(q-values[:,:,None])
    maximum=xp.max(xp.where(support,logits,-xp.inf),axis=2,keepdims=True)
    maximum=xp.where(active[:,:,None],maximum,0)
    # Terminal/unseen rows use uniform; avoid overflow in ignored logits.
    weighted=behavior*xp.exp(xp.where(support & active[:,:,None],logits-maximum,0))
    normalizer=weighted.sum(axis=2,keepdims=True)
    policy=xp.where(active[:,:,None],weighted/xp.where(normalizer>0,normalizer,1),1/actions)
    return host(policy,xp),host(used,xp),host(residual,xp)
