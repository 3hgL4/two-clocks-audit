"""SAC arm guard: the directory name is the ONLY arm identity; metrics fields are not.

Why this file exists (measured 2026-10-30, not guessed)
-------------------------------------------------------
`runs/sac_s0` during burn-in is *field-for-field identical* to `runs/mu16_s0`:

    mu=16   b_responses=256   br_responses=4096   br stays 4096

and metrics.jsonl carries NO tag/arm field -- the only thing separating a SAC arm
from a fixed arm is the directory name. Any analysis script that walks ``runs/``
and keys cells on (mu, b) -- the design pattern of ``bmu_2x2.load_cells`` -- will
fold the SAC arm into the fixed mu16_b256 cell (or the reverse). This is the same
failure mode as the 2026-10-27 noclip/mu16 collision, where the numbers moved by
only ~0.x pp, so nothing looked wrong.

Rules enforced here
-------------------
1. Whitelist. A run directory name must match ``^sac[SR]?_s\\d+$`` or
   ``^mu\\d+(_b\\d+)?_s\\d+$``. Anything else -> SystemExit. We refuse rather
   than guess, because guessing is what silently merges arms.
2. Arm identity comes from the directory name only. Never from mu/b/br.
3. Cycle grouping uses ``rollout_idx``, never ``len(rows) // n_cycles``.
   (2026-10-27: ``half = len(rows)//2`` treated rows as rollouts and halved
   nothing; it also silently accepted the 19-rollout files.)
4. ``assert_no_collision`` makes the (mu, b) key clash explicit, for use as a
   layout check before archiving or before any cross-arm comparison.

Usage
-----
    from sac_arms import load_run, by_cycle, assert_no_collision
    rows, arm = load_run("runs_sac/sac_s0")     # arm == "sac"
    cycles = by_cycle(rows)                     # {rollout_idx: [rows...]}

    assert_no_collision(["runs/mu16_s0", "runs_sac/sac_s0"])   # raises

    python analysis/sac_arms.py --self-test     # runs the guards on real+shim data
"""
import json
import os
import re
import sys

# --- arm identity (rule 1 + 2) -------------------------------------------------

SAC_RE = re.compile(r"^sac([SR])?_s(\d+)$")
FIXED_RE = re.compile(r"^mu\d+(?:_b\d+)?_s\d+$")

#: arm label -> what it means for comparison purposes
#    sac   SAC-mu controller arm (dynamic mu_plan, U=1.0738, slow loop on)
#    sacS  ablation S: fast loop only, slope fixed (no slow-loop ladder)
#    sacR  ablation R: random trigger, hazard-matched to the frozen table
#    fixed non-adaptive arm (the historical campaign)
ARM_KIND = {"sac": "sac", "sacS": "sacS", "sacR": "sacR", "fixed": "fixed"}


def classify(name):
    """Directory basename -> arm label. Raises SystemExit on unknown names."""
    base = os.path.basename(name.rstrip("/\\"))
    m = SAC_RE.match(base)
    if m:
        return "sac" + (m.group(1) or "")
    if FIXED_RE.match(base):
        return "fixed"
    raise SystemExit(
        "sac_arms: refusing unknown run directory %r.\n"
        "  Allowed: ^sac[SR]?_s\\d+$  or  ^mu\\d+(_b\\d+)?_s\\d+$\n"
        "  If this is a new arm, add it to SAC_RE/FIXED_RE deliberately -- do not\n"
        "  let it through by guessing, that is how SAC and fixed arms get merged."
        % base
    )


def is_sac(arm):
    return arm.startswith("sac")


# --- loading (rule 2: identity from the name, never from the fields) -----------

def load_run(path):
    """Load one run directory. Returns (rows, arm). arm is derived from dirname."""
    arm = classify(path)
    mpath = os.path.join(path, "metrics.jsonl")
    if not os.path.exists(mpath):
        raise SystemExit("sac_arms: no metrics.jsonl under %r" % path)
    rows = []
    with open(mpath, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit("sac_arms: bad json in %s line %d: %s" % (mpath, i, e))
            r["_dir"] = os.path.basename(os.path.normpath(path))
            r["_arm"] = arm
            rows.append(r)
    if not rows:
        raise SystemExit("sac_arms: empty metrics: %s" % mpath)
    return rows, arm


# --- cycle grouping (rule 3) ---------------------------------------------------

def by_cycle(rows, strict=True):
    """Group rows by rollout_idx (= cycle index, NOT a rollout). Preserves order.

    strict=True additionally asserts that each cycle's mu_step values are exactly
    0..n-1. A gap means the file was spliced or a step failed silently -- both are
    reasons to stop, not to average.
    """
    cycles = {}
    for r in rows:
        if "rollout_idx" not in r:
            raise SystemExit(
                "sac_arms: row without rollout_idx (cannot group cycles); "
                "refusing to fall back to row order"
            )
        cycles.setdefault(int(r["rollout_idx"]), []).append(r)
    if strict:
        for idx in sorted(cycles):
            steps = sorted(int(r.get("mu_step", -1)) for r in cycles[idx])
            if steps != list(range(len(steps))):
                raise SystemExit(
                    "sac_arms: cycle %d has mu_step %s, expected 0..%d -- "
                    "spliced or partially failed run, not analysable"
                    % (idx, steps, len(steps) - 1)
                )
    return {k: cycles[k] for k in sorted(cycles)}


def cycle_keys(rows):
    """[(cycle_idx, mu, b_responses, n_steps)] -- mu is per-cycle (SAC drifts)."""
    out = []
    for idx, cr in by_cycle(rows).items():
        mu = int(cr[0].get("mu", -1))
        b = int(cr[0].get("b_responses", -1))
        out.append((idx, mu, b, len(cr)))
    return out


def cycle_wall_s(rows):
    """Per-cycle wall seconds, three components, matching the frozen P4 gate.

    Summing per row is correct ONLY because of how train.py records these fields:
      - wall_ms_rollout / wall_ms_old_logp are written on the mu_step==0 row only
        (all later rows in the cycle carry 0.0)
      - wall_ms_update is written on EVERY row with the same per-step value
        (mu32: 15155.293 ms on all 32 rows; mu64: 7579.954 ms on all 64 rows)
    So sum-over-rows == cycle total for all three. If that recording convention
    ever becomes uniform (e.g. update written once per cycle), this function will
    silently under-count -- see P4 audit note 2026-10-30 item (3).
    """
    out = []
    for idx, cr in by_cycle(rows).items():
        ms = sum(
            float(r.get("wall_ms_rollout", 0))
            + float(r.get("wall_ms_old_logp", 0))
            + float(r.get("wall_ms_update", 0))
            for r in cr
        )
        out.append(ms / 1000.0)
    return out


# --- layout guard (rule 4) -----------------------------------------------------

def assert_no_collision(paths, verbose=False):
    """Raise if two DIFFERENT arms share the same (mu, b, br) key in any cycle.

    This is the check that would have caught noclip/mu16 on 2026-10-27 and that
    catches sac_s0 vs mu16_s0 today. Call it before any cross-arm join, and before
    archiving SAC runs next to the historical campaign.
    """
    seen = {}   # (mu, b) -> {(arm, dir)}
    for p in paths:
        rows, arm = load_run(p)
        name = os.path.basename(os.path.normpath(p))
        for _, mu, b, _n in cycle_keys(rows):
            seen.setdefault((mu, b), set()).add((arm, name))

    clashes = []
    for key, owners in sorted(seen.items()):
        arms = {a for a, _ in owners}
        # sac vs sacR/sacS also matter: ablations are distinct arms
        if len(arms) > 1:
            clashes.append((key, sorted(owners)))
    if verbose:
        for key, owners in sorted(seen.items()):
            print("  key (mu=%d, b=%d): %s" % (key[0], key[1], owners))
    if clashes:
        msg = ["sac_arms: KEY COLLISION -- distinct arms share (mu, b):"]
        for (mu, b), owners in clashes:
            msg.append("  mu=%-3d b=%-4d  %s" % (mu, b, ", ".join("%s:%s" % o for o in owners)))
        msg.append(
            "  -> these runs MUST NOT be joined by (mu, b). Key on the directory\n"
            "     name / arm label instead, and keep SAC runs out of runs/."
        )
        raise SystemExit("\n".join(msg))
    return True


# --- self-test -----------------------------------------------------------------

def _self_test(repo_root):
    ok = True
    def check(label, fn, expect_fail):
        nonlocal ok
        try:
            fn()
            got = "passed"
        except SystemExit as e:
            got = "refused"
            if not expect_fail:
                print("  FAIL %s should have passed: %s" % (label, e))
                ok = False
        if (got == "refused") != expect_fail:
            if got == "passed" and expect_fail:
                print("  FAIL %s: guard did not fire" % label)
                ok = False
        print("  %-46s %s" % (label, got))

    print("sac_arms self-test @ %s" % repo_root)
    print("[1] name whitelist")
    check("classify('mu16_s0')  -> fixed",
          lambda: _expect(classify("mu16_s0"), "fixed"), False)
    check("classify('sac_s0')   -> sac",
          lambda: _expect(classify("sac_s0"), "sac"), False)
    check("classify('sacR_s1')  -> sacR",
          lambda: _expect(classify("sacR_s1"), "sacR"), False)
    check("classify('sacS_s2')  -> sacS",
          lambda: _expect(classify("sacS_s2"), "sacS"), False)
    check("classify('sac_dryrun')   -> refused", lambda: classify("sac_dryrun"), True)
    check("classify('p4_mu32')      -> refused", lambda: classify("p4_mu32"), True)
    check("classify('runs_quarantine') -> refused", lambda: classify("runs_quarantine"), True)

    print("[2] cycle grouping on real data")
    real = os.path.join(repo_root, "runs", "mu16_s0")
    if os.path.isdir(real):
        rows, arm = load_run(real)
        cyc = by_cycle(rows)
        print("  mu16_s0: arm=%s rows=%d cycles=%d rows/cycle=%s"
              % (arm, len(rows), len(cyc), [len(v) for v in cyc.values()]))
        walls = cycle_wall_s(rows)
        print("  mu16_s0 cycle walls (3-component): %s" % [round(w, 1) for w in walls])
        if not all(660 <= w <= 1120 for w in walls):
            print("  WARN: cycle walls outside historical ~890s neighborhood")
        print("  %-46s %s" % ("mu16_s0 loads as fixed, groups by rollout_idx", "passed"))
    else:
        print("  SKIP: %s not present" % real)

    print("[3] collision detector (shim: sac_s0 cloned from mu16_s0)")
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="sac_arms_selftest_")
    try:
        shim = os.path.join(tmp, "sac_s0")
        os.makedirs(shim)
        shutil.copy(os.path.join(real, "metrics.jsonl"), os.path.join(shim, "metrics.jsonl"))
        check("same run under two arms -> collision detected",
              lambda: assert_no_collision([real, shim]), True)
        alone = tempfile.mkdtemp(prefix="sac_arms_alone_")
        try:
            s2 = os.path.join(alone, "sac_s0")
            os.makedirs(s2)
            shutil.copy(os.path.join(real, "metrics.jsonl"), os.path.join(s2, "metrics.jsonl"))
            check("sac_s0 kept apart from runs/ -> no collision",
                  lambda: assert_no_collision([s2]), False)
        finally:
            shutil.rmtree(alone, ignore_errors=True)
        print("  note: the collision above IS the real risk -- sac_s0 burn-in and")
        print("        mu16_s0 are field-identical, so only the path separates them.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[4] strict cycle check refuses spliced files")
    def spliced():
        rows = [{"rollout_idx": 0, "mu_step": 0}, {"rollout_idx": 0, "mu_step": 2}]
        by_cycle(rows)
    check("mu_step gap 0,2 -> refused", spliced, True)

    print("RESULT:", "ok" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def _expect(got, want):
    if got != want:
        raise SystemExit("expected %r got %r" % (want, got))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sys.exit(_self_test(root))
    print(__doc__)
