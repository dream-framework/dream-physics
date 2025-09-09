# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, jsonify
import numpy as np

weave = Blueprint("weave", __name__, template_folder="templates", static_folder="static")

# ---------- helpers ----------
def _rng(seed): return np.random.default_rng(int(seed))

def _orthonormal_projection(d_in=10, d_out=4, rng=None):
    A = rng.normal(size=(d_in, d_out))
    Q, _ = np.linalg.qr(A)
    return Q

def _pca3(Y):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    return Yc @ Vt[:3].T

def retention(lambda_val, lambda_q, D_eff):
    return float(np.exp(- (lambda_val / lambda_q) ** D_eff))

def generate_weave(n_total, rng, n_fil=6, n_fac=3, n_hubs=4):
    parts, labels, focus = [], [], []
    Nf = max(1, int(0.45 * n_total))
    Naf = max(1, int(0.30 * n_total))
    Nh = max(1, int(0.15 * n_total))
    Nd = max(1, n_total - (Nf + Naf + Nh))

    per_fil = max(20, Nf // n_fil)
    for _ in range(n_fil):
        c = rng.normal(scale=0.4, size=10)
        v = rng.normal(size=10); v /= np.linalg.norm(v)
        t = rng.normal(size=(per_fil, 1)) * 0.8
        E = rng.normal(size=(per_fil, 10))
        E -= (E @ v[:, None]) * v[None, :]
        pts = c + t * v + 0.03 * E
        parts.append(pts); labels += ["filament"] * len(pts); focus += [0.25] * len(pts)

    per_fac = max(20, Naf // n_fac)
    for _ in range(n_fac):
        c = rng.normal(scale=0.6, size=10)
        U = rng.normal(size=(10, 2)); U, _ = np.linalg.qr(U)
        s = rng.normal(size=(per_fac, 1)) * 0.6
        t = rng.normal(size=(per_fac, 1)) * 0.35
        pts = c + s * U[:, [0]].T + t * U[:, [1]].T + rng.normal(scale=0.015, size=(per_fac, 10))
        parts.append(pts); labels += ["facet"] * len(pts); focus += [0.45] * len(pts)

    per_hub = max(10, Nh // n_hubs)
    for _ in range(n_hubs):
        c = rng.normal(scale=0.3, size=10)
        pts = c + rng.normal(scale=0.04, size=(per_hub, 10))
        parts.append(pts); labels += ["hub"] * len(pts); focus += [0.35] * len(pts)

    pts = rng.normal(scale=0.9, size=(Nd, 10))
    parts.append(pts); labels += ["dust"] * len(pts); focus += [1.0] * len(pts)

    X10 = np.vstack(parts)
    return X10, np.array(labels), np.array(focus, float)

def project_to_4d(X10, labels, focus, lam, lam_q, D_eff, rng, P=None):
    P = _orthonormal_projection(10, 4, rng) if P is None else P
    Y = X10 @ P
    R = retention(lam, lam_q, D_eff)
    sigma0 = 0.06
    sigmas = sigma0 * (1.0 - R) * focus
    noise = rng.normal(size=Y.shape) * sigmas[:, None]
    return Y + noise, R

# ---------- routes (canonical names) ----------
@weave.get("/weave", endpoint="page")
def page():
    return render_template("weave.html")

@weave.get("/weave/data", endpoint="data")
def data():
    lam  = float(request.args.get("lam", 0.30))
    lamq = float(request.args.get("lam_q", 1.0))
    deff = float(request.args.get("deff", 2.2))
    n    = int(request.args.get("n", 3500))
    seed = int(request.args.get("seed", 42))

    rng = _rng(seed)
    X10, labels, focus = generate_weave(n, rng)
    P = _orthonormal_projection(10, 4, rng)
    Y4, R = project_to_4d(X10, labels, focus, lam, lamq, deff, rng, P)
    PCs = _pca3(Y4)

    out = {}
    for lbl in ["filament","facet","hub","dust"]:
        m = (labels == lbl)
        out[lbl] = {"x":PCs[m,0].tolist(),"y":PCs[m,1].tolist(),"z":PCs[m,2].tolist(),"count":int(m.sum())}

    return jsonify({"r":R, "lambda":lam, "lambda_q":lamq, "deff":deff, "traces":out})