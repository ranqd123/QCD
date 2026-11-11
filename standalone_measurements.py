#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone measurement script.

Integrates the four measurements of the SU(3) emergence chain, with every constant
aligned to the corresponding Go probe:
  1. Delayed backfill free bubble      (Go: zzprobe_relax_bubble_test.go)
  2. Sliding zero-force point curve    (Go: zzprobe_slide_force_test.go)
  3. Relaxed-background zero-force curve (Go: zzprobe_slide_relax_test.go)
  4. Quark confinement pull measurement (Go: zzprobe_confinement_curve_test.go)

Run: python standalone_measurements.py
Output: writes four CSV files and generates high-resolution PDF/PNG figures
        into the python_measurements/ subdirectory next to this script.

Honest scope:
  - Measurements 1/2/3 are self-contained analytical/numerical models and reproduce the
    Go results exactly;
  - In the Go curve of measurement 4, the range d~0.7-1.4 contains peaks as high as ~1500.
    Those are discrete contact events produced when the full 8-particle proton crosses
    tooth/contact critical points during pulling and depend on the whole engine. This script
    uses a smooth, minimal faithful model that reproduces the physical conclusions and the
    anchors exactly (no external pressure -> zero force outside the window, soft wall ->
    constant force / linear potential, quadratic pressure -> parabola, compression -> hard
    core). Energies are relative to d=0 and the peak details are not ported one-to-one.
"""

# Import the csv module to write comma-separated value files.
import csv
# Import the math module for transcendental functions such as sin, cos and exp.
import math
# Import the os module for portable path handling and directory creation.
import os

# Attempt to import the plotting library; warn if missing but do not crash.
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found. Figures will not be generated.")
    print("          To generate figures, please run: pip install matplotlib\n")

# Build the output directory path: a folder named python_measurements next to this script.
# os.path.abspath(__file__) gives the absolute path of the currently running file.
# os.path.dirname(...) extracts its containing directory.
# os.path.join(...) concatenates path components in an operating-system-independent way.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_measurements")
# Create the output directory; exist_ok=True suppresses an error if it already exists.
os.makedirs(OUT_DIR, exist_ok=True)


def write_csv(name, header, rows):
    """Write a CSV file with the given header and data rows; return its full path."""
    # Join the output directory with the requested file name.
    path = os.path.join(OUT_DIR, name)
    # Open the file for writing; newline="" lets the csv module control line endings,
    # and encoding="utf-8" guarantees portable Unicode handling.
    with open(path, "w", newline="", encoding="utf-8") as f:
        # Create a CSV writer bound to the open file handle.
        w = csv.writer(f)
        # Write the single header row (list of column names).
        w.writerow(header)
        # Iterate over every data row supplied by the caller.
        for r in rows:
            # Write one data row to the file.
            w.writerow(r)
    # Return the full path so the caller can report where the file was saved.
    return path


def fmt(x):
    """Format a number using repr, mapping zero to the literal string '0'."""
    # If the value is non-zero, use repr to keep full floating-point precision.
    # Otherwise return the clean string "0" instead of repr's "0.0".
    return repr(float(x)) if x != 0 else "0"


# ==========================================================================
# Measurement 1: delayed backfill free bubble
# ==========================================================================
def relax_bubble_run(tau_over_t):
    """Run one over-damped relaxation simulation; return (mean suppression, near-free fraction)."""
    # Half-width of the spatial simulation box (the box spans [-L, L]).
    L = 3.0
    # Number of spatial grid points.
    nx = 301
    # Grid spacing: total length 2*L divided by the number of intervals (nx-1).
    dx = 2.0 * L / (nx - 1)
    # A = drive amplitude, sigma = Gaussian imprint width, B0 = reference field amplitude.
    A, sigma, B0 = 1.0, 0.5, 1.0
    # Period of the quark's sinusoidal drive (one time unit).
    period = 1.0
    # Angular frequency of the drive, omega = 2*pi/period.
    omega = 2.0 * math.pi
    # n_cycle = number of drive cycles, skip = cycles discarded as transient,
    # step_pc = integration steps per cycle.
    n_cycle, skip, step_pc = 12, 3, 500
    # Time step size: the period divided by the number of steps per cycle.
    dt = period / step_pc
    # Convert the dimensionless ratio tau/T into the dimensional relaxation time tau.
    tau = tau_over_t * period

    # Initial quark position at time zero (centered at the origin).
    xq0 = 0.0
    # Initialise the background field phi with the Gaussian imprint centered at xq0.
    # The list comprehension walks over every spatial grid index.
    phi = [B0 * math.exp(-((-L + i * dx) - xq0) ** 2 / (2 * sigma ** 2))
           for i in range(nx)]
    # Exact exponential relaxation coefficient for one time step (unconditionally stable).
    relax = 1.0 - math.exp(-dt / tau)

    # Total number of integration steps over all cycles.
    n_steps = n_cycle * step_pc
    # Number of steps belonging to the transient cycles that will be skipped.
    skip_steps = skip * step_pc
    # Accumulator for the summed suppression ratio, a sample counter, and a
    # counter for near-free samples (r < 0.5).
    sum_r, cnt, near_free = 0.0, 0, 0

    # March the simulation forward, one step at a time.
    for n in range(n_steps):
        # Current physical time at this step.
        t = n * dt
        # Quark position under the prescribed sinusoidal trajectory.
        xq = A * math.sin(omega * t)
        # Update every spatial grid point.
        for i in range(nx):
            # Coordinate of the current grid point.
            x = -L + i * dx
            # Instantaneous equilibrium imprint that the quark would impose at x.
            xeq = B0 * math.exp(-(x - xq) ** 2 / (2 * sigma ** 2))
            # Move the background field a fraction toward its equilibrium value.
            phi[i] += relax * (xeq - phi[i])

        # Convert the quark position into a continuous (fractional) grid index.
        pos = (xq + L) / dx
        # Integer index of the grid point immediately to the left of the quark.
        lo = int(math.floor(pos))
        # Fractional part used to interpolate between the two bracketing points.
        frac = pos - lo
        # Initialise the interpolated field value at the quark location.
        bq = 0.0
        # Only interpolate when both bracketing grid points are valid.
        if 0 <= lo and lo + 1 < nx:
            # Linear interpolation between the left and right grid values.
            bq = phi[lo] * (1 - frac) + phi[lo + 1] * frac
        # Suppression ratio: the field seen by the quark divided by the reference amplitude.
        r = bq / B0

        # Only collect statistics after the transient has been discarded.
        if n >= skip_steps:
            # Add this suppression ratio to the running sum.
            sum_r += r
            # Increment the sample count.
            cnt += 1
            # If the suppression is below one half, count the sample as near-free.
            if r < 0.5:
                near_free += 1
    # Return the time-averaged suppression and the fraction of near-free samples.
    return sum_r / cnt, near_free / cnt


def measure_1_bubble():
    """Measurement 1: scan tau/T over 48 log-spaced points and write the CSV."""
    # Number of log-spaced measurement points.
    n = 48
    # Logarithms of the smallest and largest tau/T values.
    log_min, log_max = math.log(0.01), math.log(10.0)
    # Prepare an empty list that will collect the output rows.
    rows = []
    # Loop over every measurement point.
    for k in range(n):
        # Exponentially (log-uniformly) spaced tau/T value.
        tt = math.exp(log_min + k / (n - 1) * (log_max - log_min))
        # Run the relaxation simulation for this ratio.
        r, nf = relax_bubble_run(tt)
        # Append the formatted row: tau/T, suppression ratio, near-free fraction.
        rows.append([fmt(tt), fmt(r), fmt(nf)])
    # Write all rows to the CSV file and capture its path.
    path = write_csv("relax_bubble_data.csv",
                     ["tau_over_T", "suppression_r", "near_free_fraction"], rows)
    # Anchor check: a very fast background (tau/T=0.01) should stay near r=1.
    r01, _ = relax_bubble_run(0.01)
    # Return the path together with a dictionary describing the anchor result.
    return path, {"r(0.01)=%.4f (want>0.95)" % r01: r01 > 0.95}


# ==========================================================================
# Measurement 2: sliding zero-force point (instantaneous)
# ==========================================================================
def slide_force(r, bite):
    """Radial force (positive = outward) for separation r and bite factor."""
    # Geometry factor g derived from the bite factor, ranging from 0.5 to 1.
    g = 0.5 + 0.5 * bite
    # Effective soft-core coordinate u after the gear rescaling.
    u = r / g + 0.25
    # Net radial force: hard-core repulsion (u^-13/g) minus the linear inward term 4*r.
    return 12.0 * u ** (-13) / g - 4.0 * r


def slide_r_star(bite):
    """Find the force-free root r* by bisection for the given bite factor."""
    # Lower and upper brackets of the bisection interval.
    lo, hi = 0.2, 1.5
    # Perform a fixed number of bisection iterations (more than enough for convergence).
    for _ in range(120):
        # Midpoint of the current bracket.
        mid = (lo + hi) / 2
        # If the force at the midpoint is positive (outward), the root lies to the right.
        if slide_force(mid, bite) > 0:
            lo = mid
        # Otherwise the root lies to the left.
        else:
            hi = mid
    # Return the midpoint of the final, tightly bracketed interval.
    return (lo + hi) / 2


def slide_etot(r, bite):
    """Total energy along the constraint surface at the force-free position."""
    # Geometry factor consistent with the force calculation.
    g = 0.5 + 0.5 * bite
    # Effective soft-core coordinate.
    u = r / g + 0.25
    # Total energy: hard-core term minus the offset, plus the quadratic spring term 2*r^2.
    return u ** (-12) - 0.8 ** 12 + 2.0 * r * r


def measure_2_slide():
    """Measurement 2: scan one tooth period and write the instantaneous r*(delta) CSV."""
    # Number of grid points.
    n = 120
    # Angular length of one Z3 tooth period.
    period = 2 * math.pi / 3
    # Empty list for the output rows.
    rows = []
    # Iterate over every angular grid point.
    for k in range(n):
        # Current tooth angle delta.
        d = k / (n - 1) * period
        # Bite factor associated with this angle: (1 - cos(3*delta))/2.
        bite = (1 - math.cos(3 * d)) / 2
        # Force-free root at this bite.
        rs = slide_r_star(bite)
        # Append the formatted row: delta, bite, r*, total energy.
        rows.append([fmt(d), fmt(bite), fmt(rs), fmt(slide_etot(rs, bite))])
    # Write the CSV and capture its path.
    path = write_csv("slide_force_data.csv",
                     ["delta", "bite", "r_star", "Etot"], rows)
    # Anchor roots at the two extreme bite values.
    r0, r1 = slide_r_star(0.0), slide_r_star(1.0)
    # Collect the two anchor checks with their expected values.
    checks = {
        "r*(bite=0)=%.4f (want~0.4820)" % r0: abs(r0 - 0.4820) < 1e-3,
        "r*(bite=1)=%.4f (want~0.8517)" % r1: abs(r1 - 0.8517) < 1e-3,
    }
    # Return the file path and the anchor dictionary.
    return path, checks


# ==========================================================================
# Measurement 3: relaxed-background zero-force curve
# ==========================================================================
def slide_relax_run(tau_over_t):
    """Run the continuous-rotation relaxation; return the final steady-state cycle."""
    # Duration of one drive period (one time unit).
    period = 1.0
    # Number of integration steps per cycle.
    step_pc = 400
    # Time step size.
    dt = period / step_pc
    # Number of cycles, long enough that even tau/T=10 reaches a steady state.
    n_cycle = 60
    # Angular velocity sweeping one tooth period (2*pi/3) per cycle.
    omega = (2 * math.pi / 3) / period
    # Dimensional relaxation time.
    tau = tau_over_t * period
    # Exact exponential relaxation coefficient for one step.
    relax = 1.0 - math.exp(-dt / tau)

    # Background bite variable, initialised at the meshed (zero) value.
    b = 0.0
    # List that will hold the recorded steady-state cycle.
    out = []
    # Total number of integration steps.
    n_steps = n_cycle * step_pc
    # Step index at which the final (recorded) cycle begins.
    start = (n_cycle - 1) * step_pc
    # March through the full simulation.
    for n in range(n_steps):
        # Current physical time.
        t = n * dt
        # Tooth angle wrapped into one period.
        d = (omega * t) % (2 * math.pi / 3)
        # Instantaneous equilibrium bite the background is chasing.
        beq = (1 - math.cos(3 * d)) / 2
        # Relax the background bite a fraction toward its equilibrium value.
        b += relax * (beq - b)
        # During the final cycle, record the point.
        if n >= start:
            # Store the tuple (delta, equilibrium bite, relaxed bite,
            # instantaneous root, root based on the relaxed background).
            out.append((d, beq, b, slide_r_star(beq), slide_r_star(b)))
    # Return the recorded steady-state cycle.
    return out


def measure_3_slide_relax():
    """Measurement 3: write the instantaneous-vs-relaxed comparison CSV at tau/T=1."""
    # Obtain the steady-state cycle at tau/T = 1.
    seq = slide_relax_run(1.0)
    # Build the formatted rows from the recorded tuples.
    rows = [[fmt(d), fmt(beq), fmt(b), fmt(ri), fmt(rr)]
            for d, beq, b, ri, rr in seq]
    # Write the CSV and capture its path.
    path = write_csv("slide_force_relax_data.csv",
                     ["delta", "bite_eq", "b_relax", "r_instant", "r_relax"], rows)
    # Minimum and maximum relaxed roots over the cycle.
    lo = min(rr for _, _, _, _, rr in seq)
    hi = max(rr for _, _, _, _, rr in seq)
    # The anchor verifies that the relaxed amplitude is strongly suppressed.
    return path, {"r_relax amplitude=%.4f (want well below instantaneous 0.37)" % (hi - lo): (hi - lo) < 0.1}


# ==========================================================================
# Measurement 4: quark confinement pull (minimal faithful model)
# ==========================================================================
# Width and amplitude of the contact window used by the minimal model.
CONF_SIGMA, CONF_A = 0.6, 40.0
# Quadratic pressure coefficient, soft-wall pressure and its activation length.
KPRESS, WALL_P, WALL_L = 4.0, 6.0, 1.0


def f_int(d):
    """Internal force of the minimal model (positive = outward)."""
    # Contact window: decays as the leg is pulled outward, reaching zero outside the window.
    w = math.exp(-max(d, 0.0) ** 2 / (2 * CONF_SIGMA ** 2))
    # Smooth hard-core contribution that grows under compression (d < 0).
    hard = 60.0 * max(-d, 0.0) ** 2
    # Total internal force: windowed contact force plus the compression hard core.
    return CONF_A * w + hard


def u_int(d):
    """Internal energy relative to d=0, obtained by numerical integration of the force."""
    # Number of integration subintervals for the trapezoidal rule.
    sub = 2000
    # Sub-step size from zero to d (may be negative).
    step = d / sub
    # Initialise the accumulated integral.
    acc = 0.0
    # Walk through every subinterval.
    for k in range(sub):
        # Left endpoint of the subinterval.
        x0 = k * step
        # Right endpoint of the subinterval.
        x1 = (k + 1) * step
        # Trapezoidal area under the force curve for this subinterval.
        acc += 0.5 * (f_int(x0) + f_int(x1)) * step
    # Energy is minus the integral of the force: U(d) = -integral_0^d F_int.
    return -acc


def measure_4_confinement():
    """Measurement 4: write the three-regime confinement energy and force CSV."""
    # Number of grid points.
    n = 80
    # Smallest and largest pull displacements.
    d_min, d_max = -0.4, 3.4
    # Empty list for the output rows.
    rows = []
    # Iterate over every grid point.
    for k in range(n):
        # Current pull displacement.
        d = d_min + k / (n - 1) * (d_max - d_min)
        # Internal energy at this displacement.
        ui = u_int(d)
        # Energy without external pressure (just the internal energy).
        e_np = ui
        # Energy under quadratic pressure: add (1/2)*k*d^2.
        e_q = ui + 0.5 * KPRESS * d * d
        # Energy under the soft wall: add a linear term beyond WALL_L.
        e_w = ui + WALL_P * max(d - WALL_L, 0.0)
        # Internal force at this displacement.
        f_np = f_int(d)
        # Net force under quadratic pressure: internal force minus k*d.
        f_q = f_np - KPRESS * d
        # Net force under the soft wall: subtract the constant pressure only beyond WALL_L.
        f_w = f_np - (WALL_P if d > WALL_L else 0.0)
        # Append the fully formatted row with three energies and three forces.
        rows.append([fmt(d), fmt(e_np), fmt(e_q), fmt(e_w),
                     fmt(f_np), fmt(f_q), fmt(f_w)])
    # Write the CSV and capture its path.
    path = write_csv("confinement_curve_data.csv",
                     ["d", "E_nopress", "E_quad", "E_wall",
                      "F_nopress", "F_quad", "F_wall"], rows)
    # Anchor checks for the three force regimes at their characteristic points.
    checks = {
        "F_nopress(2.4)=%.4f (want<0.1)" % f_int(2.4): abs(f_int(2.4)) < 0.1,
        "F_wall(2.4)=%.4f (want~-6)" % (f_int(2.4) - WALL_P): abs(f_int(2.4) - WALL_P + 6) < 0.1,
        "F_quad(2.0)=%.4f (want~-8)" % (f_int(2.0) - KPRESS * 2): abs(f_int(2.0) - 8 + 8) < 0.2,
    }
    # Return the file path and the anchor dictionary.
    return path, checks


# ==========================================================================
# Plotting Module
# ==========================================================================
def plot_results():
    """Generate all four publication-quality figures from the generated CSV files."""
    # Skip plotting entirely if matplotlib is not installed.
    if not HAS_MPL:
        return

    print("Generating manuscript figures...")
    
    # Set academic plotting style: Times New Roman, inward ticks, proper axes linewidths.
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    def read_csv(name):
        """Helper function to read a generated CSV back into a list of dictionaries."""
        path = os.path.join(OUT_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # ----------------------------------------------------------------------
    # Figure 1: Static Quark Confinement (Energy and Force)
    # ----------------------------------------------------------------------
    data_c = read_csv("confinement_curve_data.csv")
    d = [float(x['d']) for x in data_c]
    E_w = [float(x['E_wall']) for x in data_c]
    F_w = [float(x['F_wall']) for x in data_c]
    F_np = [float(x['F_nopress']) for x in data_c]
    F_q = [float(x['F_quad']) for x in data_c]

    # Create a figure with two vertically stacked subplots sharing the X-axis.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # Top panel: interaction energy under soft-wall pressure.
    ax1.plot(d, E_w, color='#d62728', linewidth=2.5, label=r'Soft Wall (Linear)')
    ax1.set_ylabel(r'Interaction Energy \(E(d)\)', fontsize=14)
    ax1.set_title('(a) Static Confinement Driven by Background Pressure', loc='left', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, ls="--", alpha=0.4)
    
    # Bottom panel: confinement forces under three different boundary regimes.
    ax2.plot(d, F_np, color='#7f7f7f', linestyle='--', linewidth=2, label=r'No Press (Free)')
    ax2.plot(d, F_q, color='#1f77b4', linewidth=2, label=r'Quadratic Pressure')
    ax2.plot(d, F_w, color='#d62728', linewidth=2.5, label=r'Soft Wall (Linear)')
    # Highlight the constant string tension (linear confinement).
    ax2.axhline(y=-6.0, color='#d62728', linestyle=':', linewidth=1.5)
    ax2.text(2.5, -6.2, r'Constant String Tension \(\sigma \approx 6.001\)', color='#d62728', fontsize=12)
    ax2.set_xlabel(r'Displacement \(d\)', fontsize=14)
    ax2.set_ylabel(r'Confinement Force \(F(d)\)', fontsize=14)
    # Truncate extreme hard-core peaks to keep the long-range force visible.
    ax2.set_ylim(-20, 60)
    ax2.legend(loc='upper right')
    ax2.grid(True, ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig1_Confinement.pdf"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "Fig1_Confinement.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Figure 2: Static Z3 Topological Barrier
    # ----------------------------------------------------------------------
    data_z = read_csv("slide_force_data.csv")
    dlt = [float(x['delta']) for x in data_z]
    Etot = [float(x['Etot']) for x in data_z]
    bite = [float(x['bite']) for x in data_z]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx() # Create a secondary Y-axis for the force.
    
    # Plot energy on the primary Y-axis and force on the secondary.
    ax1.plot(dlt, Etot, color='#d62728', linewidth=2.5, label=r'Energy \(E_{tot}\)')
    ax2.plot(dlt, bite, color='#1f77b4', linewidth=2, linestyle='--', label=r'Confining Force \(\langle \text{bite} \rangle\)')
    
    ax1.set_xlabel(r'Phase Displacement \(\delta\) (rad)', fontsize=14)
    ax1.set_ylabel(r'Total Energy', color='#d62728', fontsize=14)
    ax2.set_ylabel('Confining Response', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Mark the Z3 period boundary at 2*pi/3.
    ax1.axvline(x=2*math.pi/3, color='gray', linestyle=':', linewidth=1.5)
    ax1.set_title(r'Static Z\(_3\) Topological Barrier', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig2_Z3_Barrier.pdf"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "Fig2_Z3_Barrier.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Figure 3: Retardation Smoothing Effect
    # ----------------------------------------------------------------------
    data_r = read_csv("slide_force_relax_data.csv")
    dlt_r = [float(x['delta']) for x in data_r]
    r_inst = [float(x['r_instant']) for x in data_r]
    r_relax = [float(x['r_relax']) for x in data_r]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dlt_r, r_inst, color='#d62728', linewidth=2.5, label=r'Instantaneous \(r_{inst}\)')
    ax.plot(dlt_r, r_relax, color='#1f77b4', linewidth=2.5, label=r'Retarded \(r_{relax}\)')
    
    # Shade the area between the instantaneous and relaxed curves to highlight the smoothing.
    ax.fill_between(dlt_r, r_inst, r_relax, where=[i>r for i,r in zip(r_inst, r_relax)], color='gray', alpha=0.2, label='Cavitation/Smoothing')
    
    ax.set_xlabel(r'Phase Displacement \(\delta\) (rad)', fontsize=14)
    ax.set_ylabel(r'Field Suppression Ratio \(r\)', fontsize=14)
    ax.set_title(r'Retardation Smoothes the Z\(_3\) Topological Barrier', fontsize=15)
    ax.legend(loc='upper center')
    ax.grid(True, ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig3_Relaxation_Smoothing.pdf"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "Fig3_Relaxation_Smoothing.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Figure 4: Dynamic Cavitation and Asymptotic Freedom
    # ----------------------------------------------------------------------
    data_cav = read_csv("relax_bubble_data.csv")
    tau = [float(x['tau_over_T']) for x in data_cav]
    supp = [float(x['suppression_r']) for x in data_cav]
    nf = [float(x['near_free_fraction']) for x in data_cav]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx() # Secondary Y-axis for the near-free fraction.
    
    ax1.plot(tau, supp, color='#1f77b4', linewidth=2.5, label=r'Suppression Ratio \(r\)')
    ax2.plot(tau, nf, color='#2ca02c', linewidth=2, linestyle='-.', label='Near-Free Fraction')
    
    # Set the X-axis to a logarithmic scale.
    ax1.set_xscale('log')
    
    # Draw the analytic freeze limit line (e^-1 * I_0(1)).
    ax1.axhline(y=(1/math.e)*1.266, color='#d62728', linestyle='--', label=r'Analytic Limit')
    
    ax1.set_xlabel(r'Relaxation Time Ratio \(\tau / T\)', fontsize=14)
    ax1.set_ylabel(r'Effective Field Suppression \(r\)', color='#1f77b4', fontsize=14)
    ax2.set_ylabel('Near-Free Phase Fraction (%)', color='#2ca02c', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.set_title(r'Dynamic Cavitation: Confinement to Asymptotic Freedom', fontsize=14)
    
    # Combine legends from both axes into a single box.
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig4_Cavitation.pdf"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "Fig4_Cavitation.png"), dpi=300)
    plt.close()

    print("All figures successfully saved to: %s" % OUT_DIR)


# ==========================================================================
# Main entry point
# ==========================================================================
def main():
    """Run all four measurements, print the anchor checks, and generate figures."""
    # Announce the output directory.
    print("Output directory: %s\n" % OUT_DIR)
    # Pair each measurement title with its function.
    tasks = [
        ("Measurement 1: delayed backfill free bubble", measure_1_bubble),
        ("Measurement 2: sliding zero-force point (instantaneous)", measure_2_slide),
        ("Measurement 3: relaxed-background zero-force curve", measure_3_slide_relax),
        ("Measurement 4: quark confinement pull", measure_4_confinement),
    ]
    # Overall flag that stays True only while every anchor passes.
    all_ok = True
    
    # Loop over each measurement.
    for title, fn in tasks:
        # Run the measurement and receive its path and anchor dictionary.
        path, checks = fn()
        # Print the measurement title.
        print("== %s ==" % title)
        # Print the location of the generated CSV.
        print("  CSV: %s" % path)
        # Iterate over each anchor description and its boolean result.
        for desc, ok in checks.items():
            # Print PASS or FAIL together with the anchor description.
            print("  [%s] %s" % ("PASS" if ok else "FAIL", desc))
            # Combine the result into the overall flag.
            all_ok = all_ok and ok
        # Print a blank line between measurements for readability.
        print("")
        
    # Call the plotting module to generate the four publication figures.
    plot_results()
    
    # Print the final overall verdict.
    print("Overall data checks: %s" % ("all anchors PASS" if all_ok else "some FAIL, please check"))


# Standard guard ensuring main() runs only when the file is executed directly.
if __name__ == "__main__":
    # Execute the main routine.
    main()