# generate_hamiltonian.py
import os, argparse
import numpy as np

def make_hamiltonian_file(name: str, D: int, out_dir: str = "data/321",
                          N: int = 321, dt: float = 0.02, seed: int = 0,
                          order: str = "qp"):
    """
    Create data/321/<name>.txt with columns: t, x1, x2, ..., xD
    Dynamics: q' = p, p' = -2 q  (omega = sqrt(2))
    D must be even. If order='qp', columns are [q1..qn, p1..pn];
    if order='interleave', columns are [q1,p1,q2,p2,...].
    """
    assert D % 2 == 0, "D must be even (D=2n)"
    n = D // 2
    rng = np.random.default_rng(seed)

    # time grid: 321 points -> 0..(N-1)*dt (default dt=0.02 -> 0..6.40)
    t = np.arange(N, dtype=float) * dt
    t = t.reshape(-1, 1)  # (N,1)

    # closed-form solution for SHO with omega = sqrt(2)
    omega = np.sqrt(2.0)
    coswt = np.cos(omega * t)          # (N,1)
    sinwt = np.sin(omega * t)          # (N,1)

    # random initial conditions (standard normal)
    q0 = rng.standard_normal(n)        # (n,)
    p0 = rng.standard_normal(n)        # (n,)

    # Broadcast to (N,n)
    q = q0[None, :] * coswt + (p0[None, :] / omega) * sinwt
    p = p0[None, :] * coswt - (omega * q0[None, :]) * sinwt

    if order == "qp":
        X = np.concatenate([q, p], axis=1)   # (N, D)
    elif order == "interleave":
        X = np.empty((N, D))
        X[:, 0::2] = q
        X[:, 1::2] = p
    else:
        raise ValueError("order must be 'qp' or 'interleave'")

    arr = np.column_stack([t, X])  # first col is time
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.txt")
    np.savetxt(out_path, arr, fmt="%.6f")
    print(f"Wrote {out_path} with shape {arr.shape} (N={N}, D={D})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="base filename (without .txt)")
    ap.add_argument("--dim", type=int, required=True, help="total component dimension D (even)")
    ap.add_argument("--out", default="data/321", help="output directory (default: data/321)")
    ap.add_argument("--steps", type=int, default=321, help="number of time steps (default: 321)")
    ap.add_argument("--dt", type=float, default=0.02, help="time step size (default: 0.02)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for q0,p0")
    ap.add_argument("--order", choices=["qp", "interleave"], default="qp",
                    help="column order: qp=[q1..qn,p1..pn], interleave=[q1,p1,q2,p2,...]")
    args = ap.parse_args()
    make_hamiltonian_file(args.name, args.dim, args.out, args.steps, args.dt, args.seed, args.order)
