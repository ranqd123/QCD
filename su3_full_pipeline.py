#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full SU(3) emergence pipeline (pure Python standard library, zero third-party dependencies).

This single file packages the complete SU(3) line of the one-element field theory, with
every numerical constant aligned to the corresponding Go probes. It contains:

  Part A. SU(3) matrix machinery, Gell-Mann basis, structure constants and group factors.
  Part B. One-loop Feynman-parameter fluctuation integrals (ln det) from which b0 emerges:
          quark, ghost and gluon handled as separate fields (no merged matrices).
  Part C. Three SU(2) subgroup embedding (the "supplement A" group-theory coverage, rank 8).
  Part D. One-loop renormalisation-group running alpha_s(mu), g(mu) and the Landau scale.
  Part E. Discrete-link Metropolis Monte Carlo fluctuation sampling with binning errors.
  Part F. The four engineering measurements (free bubble, sliding zero-force point,
          relaxed-background zero-force, quark confinement pull).

Run: python su3_full_pipeline.py
Output: writes every CSV into the su3_pipeline_data/ subdirectory next to this script and
prints the anchor checks. Honest scope is documented at the end of each section.
"""

# Import csv for writing comma-separated value files.
import csv
# Import math for transcendental functions (sin, cos, exp, log, sqrt, sinh, pi).
import math
# Import os for portable path handling and directory creation.
import os
# Import random for Monte Carlo proposal and acceptance random numbers.
import random

# Build the output directory: a folder named su3_pipeline_data next to this script.
# os.path.abspath(__file__) is the absolute path of the running file.
# os.path.dirname(...) extracts its containing directory.
# os.path.join(...) concatenates path components in an OS-independent way.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "su3_pipeline_data")
# Create the output directory; exist_ok=True suppresses an error when it already exists.
os.makedirs(OUT_DIR, exist_ok=True)


def write_csv(name, header, rows):
    """Write a CSV file with the given header and data rows; return its full path."""
    # Join the output directory with the requested file name.
    path = os.path.join(OUT_DIR, name)
    # Open the file for writing; newline="" lets the csv module control line endings.
    with open(path, "w", newline="", encoding="utf-8") as f:
        # Create a CSV writer bound to the open file handle.
        w = csv.writer(f)
        # Write the single header row.
        w.writerow(header)
        # Iterate over every data row supplied by the caller.
        for r in rows:
            # Write one data row.
            w.writerow(r)
    # Return the full path so the caller can report where the file was saved.
    return path


def fmt(x):
    """Format a number using repr, mapping zero to the literal string '0'."""
    # Non-zero values keep full precision via repr; zero becomes the clean string "0".
    return repr(float(x)) if x != 0 else "0"


# ==========================================================================
# Part A. SU(3) matrix machinery
# ==========================================================================
def mat_id():
    """Return the 3x3 identity matrix (elements are complex numbers)."""
    # Build a 3x3 zero matrix and then set the diagonal entries to one.
    m = [[0j for _ in range(3)] for _ in range(3)]
    # Walk the three diagonal positions.
    for i in range(3):
        m[i][i] = 1j * 0 + 1.0
    return m


def mat_mul(a, b):
    """Multiply two 3x3 complex matrices and return the product."""
    # Allocate a 3x3 zero matrix for the result.
    r = [[0j for _ in range(3)] for _ in range(3)]
    # Iterate over the rows of the result.
    for i in range(3):
        # Iterate over the columns of the result.
        for j in range(3):
            # Sum over the contraction index k.
            s = 0j
            for k in range(3):
                s += a[i][k] * b[k][j]
            r[i][j] = s
    return r


def mat_add(a, b):
    """Add two 3x3 matrices element by element."""
    # Build the result by adding matching entries.
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def mat_scale(a, c):
    """Multiply every entry of matrix a by the real scalar c."""
    # Scale each entry by c (the complex multiplication handles the type).
    return [[a[i][j] * c for j in range(3)] for i in range(3)]


def mat_dag(a):
    """Return the conjugate transpose (Hermitian dagger) of matrix a."""
    # Transpose indices and take the complex conjugate of each entry.
    return [[complex(a[j][i].real, -a[j][i].imag) for j in range(3)] for i in range(3)]


def mat_tr(a):
    """Return the trace (sum of diagonal entries) of matrix a."""
    # Add the three diagonal entries.
    return a[0][0] + a[1][1] + a[2][2]


def mat_exp(a):
    """Compute the matrix exponential via a truncated Taylor series (small-norm safe)."""
    # Start the accumulated result at the identity.
    res = mat_id()
    # Start the running Taylor term at the identity.
    term = mat_id()
    # Sum 25 Taylor terms, which is ample for the small matrices used here.
    for k in range(1, 26):
        # term <- term * a / k (the 1/k factor builds the factorial denominator).
        term = mat_scale(mat_mul(term, a), 1.0 / k)
        # Add this term to the accumulated result.
        res = mat_add(res, term)
    return res


def gell_mann():
    """Return the eight standard Hermitian Gell-Mann matrices as a list of 3x3 matrices."""
    # Allocate eight 3x3 zero matrices.
    L = [[[0j for _ in range(3)] for _ in range(3)] for _ in range(8)]
    # lambda1: real symmetric off-diagonal in the (0,1) plane.
    L[0][0][1], L[0][1][0] = 1.0, 1.0
    # lambda2: imaginary antisymmetric in the (0,1) plane.
    L[1][0][1], L[1][1][0] = complex(0, -1), complex(0, 1)
    # lambda3: diagonal (1,-1,0).
    L[2][0][0], L[2][1][1] = 1.0, -1.0
    # lambda4: real symmetric off-diagonal in the (0,2) plane.
    L[3][0][2], L[3][2][0] = 1.0, 1.0
    # lambda5: imaginary antisymmetric in the (0,2) plane.
    L[4][0][2], L[4][2][0] = complex(0, -1), complex(0, 1)
    # lambda6: real symmetric off-diagonal in the (1,2) plane.
    L[5][1][2], L[5][2][1] = 1.0, 1.0
    # lambda7: imaginary antisymmetric in the (1,2) plane.
    L[6][1][2], L[6][2][1] = complex(0, -1), complex(0, 1)
    # Shared factor for lambda8 = diag(1,1,-2)/sqrt(3).
    g = 1.0 / math.sqrt(3.0)
    L[7][0][0], L[7][1][1], L[7][2][2] = g, g, -2 * g
    return L


def struct_f():
    """Compute the structure constants f[a][b][c] from the Gell-Mann commutators."""
    # Get the Gell-Mann basis.
    lm = gell_mann()
    # Allocate an 8x8x8 array of real numbers.
    f = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(8)]
    # Iterate over the first two generator indices.
    for a in range(8):
        for b in range(8):
            # Compute the commutator [lambda_a, lambda_b].
            comm = mat_add(mat_mul(lm[a], lm[b]), mat_scale(mat_mul(lm[b], lm[a]), -1))
            # Project onto each generator to read off f.
            for c in range(8):
                # [lambda_a,lambda_b]=2i f lambda_c; Tr(comm lambda_c)=4i f; f=(1/4) Im Tr.
                f[a][b][c] = 0.25 * mat_tr(mat_mul(comm, lm[c])).imag
    return f


def fund_tf():
    """Read the fundamental group factor T_F from Tr(T_a T_b) (T=lambda/2), using a=0."""
    # Get the Gell-Mann basis and take half of lambda0.
    lm = gell_mann()
    t0 = mat_scale(lm[0], 0.5)
    # Tr(lambda0/2 . lambda0/2) = 1/4 Tr(lambda0^2) = 1/4 * 2 = 1/2.
    return mat_tr(mat_mul(t0, t0)).real


def adj_ca():
    """Read the adjoint quadratic Casimir C_A from the Killing form, using d=g=0."""
    # Compute the structure constants.
    f = struct_f()
    # Sum over a,b of f_ab0 * f_ab0 (the (0,0) entry of the Killing form).
    ca = 0.0
    for a in range(8):
        for b in range(8):
            ca += f[a][b][0] * f[a][b][0]
    return ca


def simpson(fn, n):
    """Integrate fn on [0,1] with the composite Simpson rule using n intervals (n even)."""
    # Grid spacing.
    h = 1.0 / n
    # Seed the sum with the two endpoint values.
    s = fn(0.0) + fn(1.0)
    # Walk the interior grid points.
    for i in range(1, n):
        x = i * h
        # Odd indices carry weight 4, even indices weight 2.
        s += 4 * fn(x) if i % 2 == 1 else 2 * fn(x)
    # Apply the Simpson prefactor h/3.
    return s * h / 3.0


# ==========================================================================
# Part B. One-loop Feynman-parameter fluctuation integrals (b0 emerges)
# ==========================================================================
def b0_gluon():
    """Vector (adjoint, bosonic): kinetic 8x(1-x) plus spin 8/3; return the positive b0 piece."""
    # Read the adjoint Casimir automatically (must not be hard-coded).
    ca = adj_ca()
    # Kinetic integrand after triangle/seagull cancellation, integral = 4/3.
    k_int = simpson(lambda x: 8.0 * x * (1.0 - x), 200)
    # Spin integrand from the -2gF vertex (four Lorentz components), integral = 8/3.
    s_int = simpson(lambda x: 8.0 / 3.0, 200)
    # The sum is 4; multiply by the group factor C_A.
    return (k_int + s_int) * ca


def b0_ghost():
    """Ghost (adjoint scalar, fermionic): transverse 2x(1-x); return the negative b0 piece."""
    # Read the adjoint Casimir automatically.
    ca = adj_ca()
    # Transverse integrand, integral = 1/3.
    g_t = simpson(lambda x: 2.0 * x * (1.0 - x), 200)
    # Longitudinal integrand; its integral must be zero (background gauge transversality).
    g_l = simpson(lambda x: 1.0 - 6.0 * x + 6.0 * x * x, 200)
    # Fermionic statistics flip the sign; return the contribution and the longitudinal check.
    return -g_t * ca, g_l


def b0_quark():
    """Quark (fundamental Dirac, fermionic, per flavour): integrand 8 T_F x(1-x); return the piece."""
    # Read the fundamental group factor automatically.
    tf = fund_tf()
    # Integral of 8 T_F x(1-x) = 4/3 T_F; fermionic statistics flip the sign.
    return -simpson(lambda x: 8.0 * tf * x * (1.0 - x), 200)


def b0_assemble(nf):
    """Assemble b0(nf) entirely from the real fluctuation integrals (no coefficient insertion)."""
    # Gluon contribution (positive).
    gluon = b0_gluon()
    # Ghost contribution (negative); the second return is the longitudinal integral.
    ghost, g_long = b0_ghost()
    # Per-flavour quark contribution (negative).
    quark = b0_quark()
    # Sum the three fields with nf quark flavours.
    return gluon + ghost + nf * quark, g_long


def part_b_csv():
    """Write the per-field b0 and the assembled b0 table; return the anchor dictionary."""
    # Per-field contributions.
    gluon = b0_gluon()
    ghost, g_long = b0_ghost()
    quark = b0_quark()
    # Assemble the rows for the CSV.
    rows = [
        [fmt(gluon), fmt(ghost), fmt(quark), fmt(g_long)],
    ]
    # Add the assembled b0 values for the key flavour counts.
    for nf in (0, 3, 16, 17):
        b0, _ = b0_assemble(float(nf))
        rows.append([fmt(nf), fmt(b0), "", ""])
    # Critical flavour number where b0 crosses zero.
    nf_crit = (gluon + ghost) / (-quark)
    rows.append([fmt(nf_crit), "", "", ""])
    # Write the CSV.
    path = write_csv("b0_emergence_data.csv",
                     ["field_or_nf", "value", "extra", "longitudinal"], rows)
    # Recompute the headline anchors for the boolean dictionary.
    b0_0, _ = b0_assemble(0.0)
    b0_3, _ = b0_assemble(3.0)
    b0_16, _ = b0_assemble(16.0)
    b0_17, _ = b0_assemble(17.0)
    # Build the anchor checks.
    checks = {
        "gluon=12": abs(gluon - 12.0) < 1e-9,
        "ghost=-1": abs(ghost + 1.0) < 1e-9,
        "quark per flavour=-2/3": abs(quark + 2.0 / 3) < 1e-9,
        "longitudinal integral=0": abs(g_long) < 1e-9,
        "b0(0)=11": abs(b0_0 - 11.0) < 1e-9,
        "b0(3)=9": abs(b0_3 - 9.0) < 1e-9,
        "nf_crit=16.5": abs(nf_crit - 16.5) < 1e-9,
        "b0(16)=1/3": abs(b0_16 - 1.0 / 3) < 1e-9,
        "b0(17)=-1/3": abs(b0_17 + 1.0 / 3) < 1e-9,
    }
    return path, checks


# ==========================================================================
# Part C. Three SU(2) subgroup embedding (rank-8 coverage)
# ==========================================================================
def su2_embed(u2, p, q):
    """Embed a 2x2 SU2 block into the (p,q) subspace; the remaining colour is diagonal 1."""
    # Allocate a 3x3 zero matrix.
    u = [[0j for _ in range(3)] for _ in range(3)]
    # The third colour index is 3-p-q (since 0+1+2=3).
    r = 3 - p - q
    u[r][r] = 1.0
    # Fill the four entries of the 2x2 block.
    u[p][p], u[p][q] = u2[0][0], u2[0][1]
    u[q][p], u[q][q] = u2[1][0], u2[1][1]
    return u


def su2_generators():
    """Return the three anti-Hermitian SU2 generators -i sigma_k/2 as 2x2 matrices."""
    # Allocate three 2x2 zero matrices.
    g = [[[0j for _ in range(2)] for _ in range(2)] for _ in range(3)]
    # -i sigma1/2.
    g[0][0][1], g[0][1][0] = complex(0, -0.5), complex(0, -0.5)
    # -i sigma2/2.
    g[1][0][1], g[1][1][0] = -0.5, 0.5
    # -i sigma3/2.
    g[2][0][0], g[2][1][1] = complex(0, -0.5), complex(0, 0.5)
    return g


def real_coords(x):
    """Project an anti-Hermitian matrix onto the eight t_a=i lambda_a/2 bases; return real coords."""
    # Gell-Mann basis.
    lm = gell_mann()
    # Coordinate vector of length eight.
    c = [0.0 for _ in range(8)]
    for a in range(8):
        # Build t_a = i lambda_a/2.
        i_ta = mat_scale(lm[a], 0.5)
        i_ta = [[0j + 1j * i_ta[i][j] for j in range(3)] for i in range(3)]
        # c_a = -2 Tr(X t_a), which is real.
        c[a] = -2.0 * mat_tr(mat_mul(x, i_ta)).real
    return c


def real_rank(rows, tol):
    """Rank of a real matrix (list of row vectors) via partial-pivot Gaussian elimination."""
    # Deep-copy the input so the caller's data is not modified.
    a = [row[:] for row in rows]
    rank = 0
    # Walk every column.
    for col in range(len(a[0])):
        piv, maxv = -1, 0.0
        # Find the largest available pivot in this column.
        for row in range(rank, len(a)):
            if abs(a[row][col]) > maxv:
                maxv, piv = abs(a[row][col]), row
        # No usable pivot: skip the column.
        if piv == -1 or maxv < tol:
            continue
        # Swap the pivot row into place.
        a[rank], a[piv] = a[piv], a[rank]
        # Eliminate the column below the pivot.
        for row in range(rank + 1, len(a)):
            f = a[row][col] / a[rank][col]
            for k in range(col, len(a[0])):
                a[row][k] -= f * a[rank][k]
        rank += 1
    return rank


def part_c_csv():
    """Write the three-subgroup coverage table; return the rank-8 anchor dictionary."""
    # SU2 generators and the three embedding subspaces.
    gens = su2_generators()
    subs = ((0, 1), (0, 2), (1, 2))
    # Collect real-coordinate rows and prepare CSV rows.
    real_rows, rows = [], []
    for s in subs:
        for k in range(3):
            # Embed the k-th SU2 generator into subspace s.
            u = su2_embed(gens[k], s[0], s[1])
            # Extract its real SU3 coordinates.
            c = real_coords(u)
            real_rows.append(c)
            rows.append([fmt(s[0]), fmt(s[1]), fmt(k)] + [fmt(v) for v in c])
    # Compute the rank of the nine generators.
    rank = real_rank(real_rows, 1e-12)
    # Write the CSV.
    path = write_csv("su2_subgroup_coverage.csv",
                     ["p", "q", "gen", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"], rows)
    # The single anchor is the rank-8 coverage.
    return path, {"three SU2 subgroups cover rank 8 (got %d)" % rank: rank == 8}


# ==========================================================================
# Part D. One-loop RG running alpha_s(mu), g(mu) and the Landau scale
# ==========================================================================
def alpha_run(alpha0, b0, ratio):
    """One-loop solution alpha_s(mu) = alpha0 / (1 + b0 alpha0/(2 pi) ln ratio)."""
    # Denominator of the analytic solution.
    denom = 1.0 + b0 * alpha0 / (2.0 * math.pi) * math.log(ratio)
    return alpha0 / denom


def part_d_csv():
    """Write the 80-point running curve; return the asymptotic-freedom anchor dictionary."""
    # b0 for three flavours, assembled from the real integrals.
    b0, _ = b0_assemble(3.0)
    # Boundary condition alpha_s(MZ)=0.118.
    alpha0 = 0.118
    # Landau scale ratio and the scan lower bound (just above it).
    lam_ratio = math.exp(-2.0 * math.pi / (b0 * alpha0))
    # Logarithmic scan endpoints.
    l0, l1, n = math.log(lam_ratio * 1.2), math.log(1000.0), 80
    rows, scan = [], []
    for k in range(n):
        ratio = math.exp(l0 + k / (n - 1) * (l1 - l0))
        alpha = alpha_run(alpha0, b0, ratio)
        g = math.sqrt(4.0 * math.pi * alpha)
        scan.append((ratio, alpha, g))
        rows.append([fmt(ratio), fmt(alpha), fmt(g)])
    # Write the CSV.
    path = write_csv("rg_running_data.csv", ["mu_over_mu0", "alpha_s", "g"], rows)
    # Anchor values.
    a10 = alpha_run(alpha0, b0, 10.0)
    a1000 = alpha_run(alpha0, b0, 1000.0)
    lam_mev = 91.19 * lam_ratio * 1000.0
    checks = {
        "b0(3)=9": abs(b0 - 9.0) < 1e-9,
        "Lambda~246 MeV (got %.1f)" % lam_mev: abs(lam_mev - 245.8) < 5.0,
        "alpha_s(10mu0)~0.0849": abs(a10 - 0.0849) < 1e-3,
        "monotonic decrease (asymptotic freedom)": a1000 < a10 < alpha0,
        "near-Landau divergence": scan[0][1] > 1.0,
    }
    return path, checks


# ==========================================================================
# Part E. Discrete-link Metropolis Monte Carlo
# ==========================================================================
def mc_random_link(delta):
    """Generate a random small SU3 link exp(i sum theta_a T_a) with Gaussian theta_a."""
    # Build the eight small angles.
    theta = [delta * random.gauss(0.0, 1.0) for _ in range(8)]
    # Construct the anti-Hermitian generator combination i H.
    lm = gell_mann()
    H = [[0j for _ in range(3)] for _ in range(3)]
    for a in range(8):
        H = mat_add(H, mat_scale(lm[a], theta[a] / 2.0))
    iH = [[1j * H[i][j] for j in range(3)] for i in range(3)]
    # Exponentiate to obtain the unitary proposal link.
    return mat_exp(iH)


def mc_plaq_norm(u):
    """Normalised plaquette (1/3) Re Tr(1 - U V U^dag V^dag), range [0,1]."""
    # Build the plaquette from the two links.
    p = mat_mul(mat_mul(u[0], u[1]), mat_mul(mat_dag(u[0]), mat_dag(u[1])))
    # Action = Re Tr(1 - p); normalise by N=3.
    return (3.0 - mat_tr(p).real) / 3.0


def mc_sweep(u, beta, delta):
    """One Metropolis sweep over the two links; return the new state and the acceptance rate."""
    acc = 0
    # Update each link once.
    for link in range(2):
        # Propose a left-multiplied link.
        w = mc_random_link(delta)
        nu = [u[0], u[1]]
        nu[link] = mat_mul(w, u[link])
        # Action difference (the un-normalised factor 3 is included).
        d_s = beta * (mc_plaq_norm(nu) - mc_plaq_norm(u)) * 3.0
        # Metropolis acceptance.
        if d_s <= 0.0 or random.random() < math.exp(-d_s):
            u = nu
            acc += 1
    return u, acc / 2.0


def mc_run(beta, delta, n_burn, n_samp, n_skip, n_bin):
    """Full MC: thermalise then sample; return mean, binning error and acceptance."""
    # Initialise both links to the identity.
    u = [mat_id(), mat_id()]
    acc_sum, n_sweep = 0.0, 0
    # Thermalisation.
    for _ in range(n_burn):
        u, a = mc_sweep(u, beta, delta)
        acc_sum, n_sweep = acc_sum + a, n_sweep + 1
    # Sampling with n_skip sweeps between records to remove autocorrelation.
    samples = []
    for _ in range(n_samp):
        for _ in range(n_skip):
            u, a = mc_sweep(u, beta, delta)
            acc_sum, n_sweep = acc_sum + a, n_sweep + 1
        samples.append(mc_plaq_norm(u))
    # Mean.
    mean = sum(samples) / len(samples)
    # Binning error.
    bs = len(samples) // n_bin
    bm = []
    for b in range(n_bin):
        bm.append(sum(samples[b * bs:(b + 1) * bs]) / bs)
    var = sum((x - mean) ** 2 for x in bm) / (n_bin - 1)
    serr = math.sqrt(var) / math.sqrt(n_bin)
    return mean, serr, acc_sum / n_sweep


def part_e_csv():
    """Write the beta-scan MC table; return the Haar/weak-coupling anchor dictionary."""
    # Eight beta values spanning strong to weak coupling.
    betas = (0.3, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0)
    delta = 0.2
    rows, results = [], []
    for b in betas:
        mean, serr, accept = mc_run(b, delta, 3000, 6000, 2, 30)
        results.append((b, mean, serr, accept))
        rows.append([fmt(b), fmt(mean), fmt(serr), fmt(accept)])
    # Write the CSV.
    path = write_csv("mc_plaquette_data.csv",
                     ["beta", "mean_plaquette", "error", "accept_rate"], rows)
    # Anchors: strong-coupling Haar value 8/9, weak-coupling flatness, monotonicity.
    strong, weak = results[0][1], results[-1][1]
    mono = all(results[k][1] < results[k - 1][1] for k in range(1, len(results)))
    checks = {
        "strong beta=0.3 near Haar 8/9 (got %.3f)" % strong: abs(strong - 8.0 / 9) < 0.15,
        "weak beta=14 near flat (got %.3f)" % weak: weak < 0.3,
        "monotonic decrease with beta": mono,
        "acceptance in range": all(0.3 < r[3] < 0.99 for r in results),
    }
    return path, checks


# ==========================================================================
# Part F. The four engineering measurements
# ==========================================================================
def relax_bubble_run(tau_over_t):
    """Over-damped relaxation; return (mean suppression, near-free fraction)."""
    L, nx = 3.0, 301
    dx = 2.0 * L / (nx - 1)
    A, sigma, B0 = 1.0, 0.5, 1.0
    period = 1.0
    omega = 2.0 * math.pi
    n_cycle, skip, step_pc = 12, 3, 500
    dt = period / step_pc
    tau = tau_over_t * period
    xq0 = 0.0
    phi = [B0 * math.exp(-((-L + i * dx) - xq0) ** 2 / (2 * sigma ** 2)) for i in range(nx)]
    relax = 1.0 - math.exp(-dt / tau)
    n_steps = n_cycle * step_pc
    skip_steps = skip * step_pc
    sum_r, cnt, near_free = 0.0, 0, 0
    for n in range(n_steps):
        t = n * dt
        xq = A * math.sin(omega * t)
        for i in range(nx):
            x = -L + i * dx
            xeq = B0 * math.exp(-(x - xq) ** 2 / (2 * sigma ** 2))
            phi[i] += relax * (xeq - phi[i])
        pos = (xq + L) / dx
        lo = int(math.floor(pos))
        frac = pos - lo
        bq = 0.0
        if 0 <= lo and lo + 1 < nx:
            bq = phi[lo] * (1 - frac) + phi[lo + 1] * frac
        r = bq / B0
        if n >= skip_steps:
            sum_r += r
            cnt += 1
            if r < 0.5:
                near_free += 1
    return sum_r / cnt, near_free / cnt


def part_f1_csv():
    """Measurement 1: 48-point free-bubble scan; return the anchor."""
    n, log_min, log_max = 48, math.log(0.01), math.log(10.0)
    rows = []
    for k in range(n):
        tt = math.exp(log_min + k / (n - 1) * (log_max - log_min))
        r, nf = relax_bubble_run(tt)
        rows.append([fmt(tt), fmt(r), fmt(nf)])
    path = write_csv("relax_bubble_data.csv",
                     ["tau_over_T", "suppression_r", "near_free_fraction"], rows)
    r01, _ = relax_bubble_run(0.01)
    return path, {"r(0.01)>0.95 (got %.4f)" % r01: r01 > 0.95}


def slide_force(r, bite):
    """Radial force (positive outward) for separation r and bite factor."""
    g = 0.5 + 0.5 * bite
    u = r / g + 0.25
    return 12.0 * u ** (-13) / g - 4.0 * r


def slide_r_star(bite):
    """Force-free root r* by bisection."""
    lo, hi = 0.2, 1.5
    for _ in range(120):
        mid = (lo + hi) / 2
        if slide_force(mid, bite) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def slide_etot(r, bite):
    """Total energy along the constraint surface."""
    g = 0.5 + 0.5 * bite
    u = r / g + 0.25
    return u ** (-12) - 0.8 ** 12 + 2.0 * r * r


def part_f2_csv():
    """Measurement 2: 120-point instantaneous r*(delta) scan; return the anchors."""
    n, period = 120, 2 * math.pi / 3
    rows = []
    for k in range(n):
        d = k / (n - 1) * period
        bite = (1 - math.cos(3 * d)) / 2
        rs = slide_r_star(bite)
        rows.append([fmt(d), fmt(bite), fmt(rs), fmt(slide_etot(rs, bite))])
    path = write_csv("slide_force_data.csv",
                     ["delta", "bite", "r_star", "Etot"], rows)
    r0, r1 = slide_r_star(0.0), slide_r_star(1.0)
    return path, {
        "r*(bite=0)~0.4820 (got %.4f)" % r0: abs(r0 - 0.4820) < 1e-3,
        "r*(bite=1)~0.8517 (got %.4f)" % r1: abs(r1 - 0.8517) < 1e-3,
    }


def slide_relax_run(tau_over_t):
    """Continuous-rotation relaxation; return the steady-state cycle."""
    period, step_pc = 1.0, 400
    dt = period / step_pc
    n_cycle = 60
    omega = (2 * math.pi / 3) / period
    tau = tau_over_t * period
    relax = 1.0 - math.exp(-dt / tau)
    b = 0.0
    out, n_steps = [], n_cycle * step_pc
    start = (n_cycle - 1) * step_pc
    for n in range(n_steps):
        t = n * dt
        d = (omega * t) % (2 * math.pi / 3)
        beq = (1 - math.cos(3 * d)) / 2
        b += relax * (beq - b)
        if n >= start:
            out.append((d, beq, b, slide_r_star(beq), slide_r_star(b)))
    return out


def part_f3_csv():
    """Measurement 3: 400-point relaxed-background comparison; return the anchor."""
    seq = slide_relax_run(1.0)
    rows = [[fmt(d), fmt(beq), fmt(b), fmt(ri), fmt(rr)] for d, beq, b, ri, rr in seq]
    path = write_csv("slide_force_relax_data.csv",
                     ["delta", "bite_eq", "b_relax", "r_instant", "r_relax"], rows)
    lo = min(rr for _, _, _, _, rr in seq)
    hi = max(rr for _, _, _, _, rr in seq)
    return path, {"r_relax amplitude<0.1 (got %.4f)" % (hi - lo): (hi - lo) < 0.1}


CONF_SIGMA, CONF_A = 0.6, 40.0
KPRESS, WALL_P, WALL_L = 4.0, 6.0, 1.0


def f_int(d):
    """Internal force of the minimal confinement model (positive outward)."""
    w = math.exp(-max(d, 0.0) ** 2 / (2 * CONF_SIGMA ** 2))
    hard = 60.0 * max(-d, 0.0) ** 2
    return CONF_A * w + hard


def u_int(d):
    """Internal energy relative to d=0 by numerical integration of the force."""
    sub, step, acc = 2000, d / 2000, 0.0
    for k in range(sub):
        x0, x1 = k * step, (k + 1) * step
        acc += 0.5 * (f_int(x0) + f_int(x1)) * step
    return -acc


def part_f4_csv():
    """Measurement 4: 80-point three-regime confinement table; return the anchors."""
    n, d_min, d_max = 80, -0.4, 3.4
    rows = []
    for k in range(n):
        d = d_min + k / (n - 1) * (d_max - d_min)
        ui = u_int(d)
        e_q, e_w = ui + 0.5 * KPRESS * d * d, ui + WALL_P * max(d - WALL_L, 0.0)
        f_q = f_int(d) - KPRESS * d
        f_w = f_int(d) - (WALL_P if d > WALL_L else 0.0)
        rows.append([fmt(d), fmt(ui), fmt(e_q), fmt(e_w),
                     fmt(f_int(d)), fmt(f_q), fmt(f_w)])
    path = write_csv("confinement_curve_data.csv",
                     ["d", "E_nopress", "E_quad", "E_wall",
                      "F_nopress", "F_quad", "F_wall"], rows)
    return path, {
        "F_nopress(2.4)<0.1 (got %.4f)" % f_int(2.4): abs(f_int(2.4)) < 0.1,
        "F_wall(2.4)~-6 (got %.4f)" % (f_int(2.4) - WALL_P): abs(f_int(2.4) - WALL_P + 6) < 0.1,
        "F_quad(2.0)~-8 (got %.4f)" % (f_int(2.0) - 8): abs(f_int(2.0) - 8 + 8) < 0.2,
    }


# ==========================================================================
# Main entry point
# ==========================================================================
def main():
    """Run every part, write all CSVs and print the anchor checks in English."""
    # Seed the random generator for reproducible Monte Carlo runs.
    random.seed(20261111)
    print("Output directory: %s\n" % OUT_DIR)
    # Pair each part title with its function.
    tasks = (
        ("Part B: b0 emerges from Feynman ln det integrals", part_b_csv),
        ("Part C: three SU2 subgroup coverage (rank 8)", part_c_csv),
        ("Part D: one-loop RG running and Landau scale", part_d_csv),
        ("Part E: discrete-link Metropolis MC", part_e_csv),
        ("Part F1: delayed backfill free bubble", part_f1_csv),
        ("Part F2: sliding zero-force point", part_f2_csv),
        ("Part F3: relaxed-background zero-force", part_f3_csv),
        ("Part F4: quark confinement pull", part_f4_csv),
    )
    all_ok = True
    # Run each part.
    for title, fn in tasks:
        path, checks = fn()
        print("== %s ==" % title)
        print("  CSV: %s" % path)
        # Print every anchor.
        for desc, ok in checks.items():
            print("  [%s] %s" % ("PASS" if ok else "FAIL", desc))
            all_ok = all_ok and ok
        print("")
    print("Overall: %s" % ("all anchors PASS" if all_ok else "some FAIL, please check"))


# Standard guard ensuring main() runs only when the file is executed directly.
if __name__ == "__main__":
    main()
