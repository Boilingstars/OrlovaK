# Verification report — packaging, NOT quantum chemistry

Date: 2026-11-04.

Checks run on the assembled K package:

| Check | Outcome |
|---|---|
| `python tests/selftest.py` | PASS: 11 `.vecfle`, Hessian parser on synthetic modes, MDAtom fractional parser, K geometry |
| `python tests/test_k_parameters.py` | PASS: K charge/electron parity, n=4–14, all SHA256 inputs, published spherical packing minimum angles, K vendor radii, GWH `zyz`, xTB +1/uhf0 flags |
| `python tests/test_k_review.py` | PASS: **synthetic only**, K–O geometry metrics, indexing/ID join, global-min candidate, output records and human REVIEW status |
| `python scripts/k_pipeline.py check --skip-xtb` | PASS: input vectors, hashes, source packing directions and norm |
| `python scripts/k_driver.py --dry-run` | PASS: assembled expected commands through 11 stochiometries; no energies calculated |
| `python scripts/k_driver.py --status` | Pending n=4–14; no K calculations in supplied ZIP |
| xTB **actual** K optimization / Hessian | **NOT RUN** in this environment; must be performed on user's workstation/server |
| Real GWH and CoCoS/MDAtom end-to-end integration | **NOT VERIFIED** without xTB outputs; Cursor must inspect pilot logs |

*Scientific caveats:* 
1. K crystal radius 1.52 Å for CN=6, K effective radius 1.38 Å — different Shannon conventions. Starter 3.04 Å uses the **crystal** convention by analogy with the original Cs package plus an explicitly empirical water contribution 1.52 Å; **not an optimized parameter**.
2. `radius[A]` MDAtom is a weighted mean to water reference points; not a cutoff.
3. CoCoS CritFreq=125 cm⁻¹ and `n−CN≤0.15` are project protocol diagnostics, not validated physical universal thresholds.
4. The synthetic unit tests do not validate the electronic structure method or prove the existence of any stable K hydrate minimum.
