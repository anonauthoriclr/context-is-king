"""Plot the position sweep from the multi-scramble npz. Auto-pick the scramble with the cleanest
sentence-end imposed ring (fewest self-crossings) and the strongest token->end contrast, then show
(A) its natural->imposed RSA crossover, (B) the day-token geometry, (C) the sentence-end geometry ---
each with BOTH orders traced (natural dashed, imposed solid). Prints the chosen scramble's numbers."""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
tag = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-31B-it"
z = np.load(os.path.join(DATA, "geometry", f"possweep_{tag}_all.npz"), allow_pickle=True)
fr = z["fracs"]; days = list(z["tok"]); C0 = z["cents0"]; C1 = z["cents1"]
ORD = z["orders"]; IPOS = z["ipos_all"]; RIMP = z["rimp_all"]; RNAT = z["rnat_all"]
N = len(days); nscr = len(C0)

def mds2(C):
    X = C - C.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); D = 1 - X @ X.T
    J = np.eye(N) - np.ones((N, N)) / N; B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1]
    return V[:, i[:2]] * np.sqrt(np.clip(w[i[:2]], 0, None))
def crossings(P, seq):
    s = list(seq) + [seq[0]]; E = [(s[i], s[i + 1]) for i in range(N)]
    ccw = lambda a, b, d: (d[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(d[0]-a[0])
    c = 0
    for a in range(N):
        for b in range(a + 1, N):
            if set(E[a]) & set(E[b]): continue
            p1, p2, p3, p4 = P[E[a][0]], P[E[a][1]], P[E[b][0]], P[E[b][1]]
            if ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4): c += 1
    return c
def impseq(si): return [int(np.where(IPOS[si] == k)[0][0]) for k in range(N)]

# pick: fewest sentence-end imposed-ring crossings, then strongest token->end contrast (imp@1 - imp@0)
cand = []
for si in range(nscr):
    crE = crossings(mds2(C1[si]), impseq(si))
    cand.append((crE, -(RIMP[si][-1] - RIMP[si][0]), si))
cand.sort(); sbest = cand[0][2]
print(f"chosen scr{sbest}: imp@0={RIMP[sbest][0]:+.2f} imp@1={RIMP[sbest][-1]:+.2f} "
      f"nat@0={RNAT[sbest][0]:+.2f} nat@1={RNAT[sbest][-1]:+.2f} end-crossings={cand[0][0]} order={list(ORD[sbest])}")

imp = impseq(sbest)
fig = plt.figure(figsize=(13, 4.3)); fig.patch.set_facecolor("white")
ax = fig.add_subplot(1, 3, 1)
ax.plot(fr, RNAT[sbest], "-o", color="#2166ac", lw=2.4, label="natural-order RSA")
ax.plot(fr, RIMP[sbest], "-s", color="#b2182b", lw=2.4, label="imposed-order RSA")
ax.axhline(0, color="k", lw=.5); ax.set_xlabel("position: day token (0) $\\rightarrow$ sentence end (1)")
ax.set_ylabel("cyclic-distance RSA"); ax.set_title("Integration over the sentence\n(natural $\\rightarrow$ imposed)", fontsize=10.5)
ax.legend(fontsize=8.5, loc="best"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
def ringpanel(axx, C, title, legend=False):
    P = mds2(C); nat = list(range(N)) + [0]; im = imp + [imp[0]]
    axx.plot(P[nat, 0], P[nat, 1], "--", color="#2166ac", lw=1.6, alpha=.85, zorder=1, label="natural order")
    axx.plot(P[im, 0], P[im, 1], "-", color="#b2182b", lw=1.9, alpha=.9, zorder=2, label="imposed order")
    axx.scatter(P[:, 0], P[:, 1], c=range(N), cmap="twilight", s=150, zorder=3, edgecolor="k", lw=.5)
    for i in range(N): axx.annotate(days[i][:3], (P[i, 0], P[i, 1]), fontsize=7.5, ha="center", va="center")
    axx.set_title(title, fontsize=10.5); axx.set_xticks([]); axx.set_yticks([]); axx.set_aspect("equal", "datalim")
    if legend: axx.legend(fontsize=7.5, loc="upper right", framealpha=.85)
ringpanel(fig.add_subplot(1, 3, 2), C0[sbest], "At the day token", legend=True)
ringpanel(fig.add_subplot(1, 3, 3), C1[sbest], "At the sentence end")
plt.tight_layout()
_OUT = os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(_OUT, exist_ok=True)
for ext in ("pdf", "png"): plt.savefig(os.path.join(_OUT, f"fig_possweep.{ext}"), dpi=200, bbox_inches="tight")
print(f"WROTE {_OUT}/fig_possweep.{{pdf,png}}")
