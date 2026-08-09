# Notes on reproduction

## Determinism

All computations are deterministic. The only stochastic element anywhere in
the project was a random perturbation used in exploratory work that is not
part of the paper and is not included here.

## What does and does not converge

Three quantities behave differently under refinement and should be read
accordingly.

**The steady convergence length** converges geometrically and saturates at
the Newton tolerance, near 6e-13, from N = 24. This is the clean case.

**The rightmost eigenvalue of the semidiscrete operator** does not converge
as a number: it moves between -0.25 and -0.34 for 8 <= N <= 48 because the
identity of the rightmost mode changes. Its sign does converge, and that is
all the paper claims. The weakly damped complex pair -0.31038 + 0.63253i is
reproduced at N = 24 and N = 32 and is the quantity that governs the
frequency response.

**The degeneracy time** converges in the resolution to 10.766826 at N = 64
and 96. Its stability to nine digits under reduction of the event threshold
is a different statement, made at fixed N = 32, and should not be read as
accuracy.

## Known limitations of the drivers

The event that detects the loss of transversality cannot use a threshold
below about 1e-6, nor a relative tolerance as loose as 1e-8, because the
right-hand side raises a FloatingPointError on a tentative integration stage
before the step is accepted, so the event is never evaluated. A cleaner
implementation would return a penalised value instead of raising. This
affects only how close to the event the solution can be sampled, not the
reported value.

The second degeneracy, in which the minimum of m over the domain tends to
zero at high Strouhal number, is not verified: the event time varies by
three per cent between N = 24 and N = 64 without a trend, and the state at
the event differs materially between resolutions. It is reported in the
paper as an open question and no data file is deposited for it.

## Symbolic check of the compatibility condition

The compatibility condition that removes the singularity on the critical
surface, Eq. (13) of the paper, is equivalent to Eq. (35) of

  J. I. Ramos, Appl. Math. Modelling 20, 440-458 (1996)

under the dictionary: unit volumetric flow rate; Froude number as in the
present paper, as that paper's own Torricelli law u = (1 + 2 z / F)^(1/2)
requires; Weber number twice ours; pressure difference of opposite sign and
normalised by inertia rather than by capillarity; slope s = tan(theta). The
double sign in that equation distinguishes downward from upward jets. The
equivalence is checked symbolically by docs/check_compatibility.py.

## Environment sensitivity

Two of the numbers in the tables are not fully converged in the resolution
and are therefore sensitive to the exact arithmetic of the environment: the
mean convergence length at N = 16 and N = 24 in table 4. Between two
environments differing only in library versions they were observed to move
by 2e-4 and 2e-6 respectively, while the N = 32 and N = 48 values agreed to
all eight digits printed. This is the expected behaviour of an unconverged
quantity and not an error; it is the reason the paper quotes agreement
between N = 32 and N = 48 rather than across the whole column.

A run of the finite-difference cross-check at M = 640 returned "not
detected" in one environment and t* = 10.766778 in another, on the same
input. The tasks now distinguish three outcomes -- event found, integration
completed without the event, integrator or right-hand-side failure -- so
that a failed job can no longer be mistaken for the statement that a mesh
does not resolve the degeneracy. If a task reports FAILED, its result must
not be used.

## The guard on the right-hand side

`UnsteadySpectral.rhs` used to raise on a non-physical state. Because
`solve_ivp` does not catch exceptions from the right-hand side, a single
failed tentative stage aborted the whole integration, which made the
degeneracy event unreachable below a threshold of about 1e-6 and, in the
more strongly forced cases, unreachable at all.

The default is now `guard='clamp'`: the offending quantity is floored, the
occurrence is counted in `n_clamped`, and the integrator rejects the step in
the ordinary way. Clamping is a numerical device and not a model. A run is
trustworthy only up to the first accepted step at which it occurred, so
every task that integrates in time reports `n_clamped`, and a non-zero value
must be treated as a warning rather than ignored. The event fires at a
positive threshold, before that point, which is why the device is safe here.

Even with the guard the step size still collapses below a threshold of
about 1e-6 in the strongly forced cases, so the supporting studies use 1e-4,
which costs about 1e-5 in the event time.

## A worked example of version drift

The finite-difference cross-check at M = 640 reported "not detected" on the
cluster and t* = 10.766778 on the development machine. The cause was not
the physics and not the arithmetic. The cluster carries numpy 1.24, which
has `trapz` but not `trapezoid`; the file had therefore been patched by hand
there, and in the process the copy fell behind the one that later acquired
the tip-closure fix. The run aborted on the transversality guard after 0.1 s
and the old code reported that abort with the same label it used for a mesh
too coarse to resolve the event.

Three changes followed. The single use of the renamed function now binds
whichever name exists, so the same file runs unmodified under both numpy 1
and numpy 2 and there is no reason to patch it locally. Every task prints
the digests of the modules it imported and the numpy and scipy versions into
its log. And `SHA256SUMS` allows two copies to be compared before their
results are.

With the copies synchronised, the cluster reproduces t* = 10.766778, under a
different numpy and a differently written volume integral. That agreement is
worth more than the original number.

## What a reader running this code will obtain

Every task is deterministic. Running the same task twice in the same
environment gives bit-identical output. Across environments the situation
is not uniform, and the distinction matters:

*Converged quantities* agree to every digit printed. The steady convergence
lengths, the finite-thickness values, the frequency response and the
degeneracy time at the resolutions where it has converged were reproduced
on two machines carrying numpy 2.4 and numpy 1.24 with no difference at the
printed precision.

*Unconverged quantities* do not, and cannot. The mean convergence length at
N = 16 and N = 24 in table 4 moved by 2e-4 and 2e-6 between those two
environments, while the N = 32 and N = 48 entries agreed to all eight digits
printed. Those two rows are in the table to show convergence, not as
results, and the paper's claim is the agreement between N = 32 and N = 48.

Nothing else in the deposited data is environment-dependent at the precision
quoted.

## Why several numbers changed after the first draft

None of them changed because the same computation gave a different answer.
The causes were, in every case, one of three:

*A quantity that had never been computed.* The degeneracy time appeared to
have converged because N = 64 and N = 96 agreed to seven decimals. Computing
N = 128 and N = 160 showed that agreement to be a plateau, and moved the
value by 1.9e-5. The old numbers are still exactly what the old resolutions
give; what was wrong was the inference drawn from two of them.

*An interpretation not tested against a diagnostic.* The failure at high
Strouhal number was read as a collapse of the mass per unit length driven by
a steepening velocity gradient. Recording the gradient alongside the mass
showed it never exceeds 0.67, far too small for that mechanism, and that at
the amplitude used in the frequency sweep the mass does not collapse at all.
The reading was wrong; the numbers were not.

*Version drift between two copies of the code.* One entry of the
cross-check table was produced by a file that had been patched by hand on
the cluster and had fallen behind. This is the only case in which the same
nominal computation gave two different answers, and it is the reason for
SHA256SUMS, for the digests printed into every job log, and for the
numpy-version shim that removed the need to patch anything locally.

The practical lesson is in `check_paper_against_data.py`, which verifies
mechanically that every deposited number appears in the manuscript. A
transcription that goes stale is the failure mode that no amount of care
prevents.
