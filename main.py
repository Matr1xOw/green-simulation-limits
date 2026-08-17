"""Print the three results.

    python main.py
    python main.py --plot
"""

import argparse

from gns import ESTIMATORS, convergence, error_by_tail, slope, study

DIMENSIONS = [1, 2, 4, 8]
BUDGETS = [1000, 2000, 4000, 8000, 16000]
OUTER, INNER, TRIALS = 200, 20, 40


def table(header, rows, first="  d"):
    width = len(first)
    print(first + "".join(name.rjust(13) for name in header))
    print(" " * (width - 2) + "-" * (13 * len(header) + 2))
    for label, values in rows:
        print(str(label).rjust(width) + "".join(values))


def cell(value, ess=False):
    return f"{value:13.1f}" if ess else f"{value:13.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="also write imse.png")
    args = parser.parse_args()

    columns = list(ESTIMATORS) + ["kernel ESS"]

    print(f"\n1. IMSE by conditioning dimension "
          f"({OUTER} scenarios x {INNER} paths, {TRIALS} trials)\n")
    results = {d: study(d, OUTER, INNER, TRIALS) for d in DIMENSIONS}
    table(
        columns,
        [(d, [cell(results[d][n], n == "kernel ESS") for n in columns])
         for d in DIMENSIONS],
    )
    print(f"\n   ESS is how many of the {OUTER * INNER} pooled draws the kernel really")
    print(f"   averages. Standard nested simulation gives each scenario {INNER}, so ESS")
    print("   falling that far means the reuse has bought nothing.")

    print("\n\n2. IMSE by total budget, d = 1 (theory says slope -1)\n")
    conv = convergence(BUDGETS, trials=TRIALS // 2)
    table(
        list(ESTIMATORS),
        [(b, [cell(conv[b][n]) for n in ESTIMATORS]) for b in BUDGETS],
        first="budget",
    )
    slopes = {n: slope(BUDGETS, [conv[b][n] for b in BUDGETS]) for n in ESTIMATORS}
    print(" slope" + "".join(f"{slopes[n]:13.2f}" for n in ESTIMATORS))

    print("\n\n3. IMSE by distance from the mixture centre, d = 1\n")
    tails = error_by_tail(trials=TRIALS)
    labels = ["central", "inner", "outer", "tail"]
    table(
        list(ESTIMATORS),
        [(labels[b], [cell(tails[n][b]) for n in ESTIMATORS])
         for b in range(len(labels))],
        first="  |kappa|",
    )
    print("\n   The paper's own caveat, reproduced: reuse degrades in the tails,")
    print("   which is exactly where VaR and capital requirements are read.\n")

    if args.plot:
        from plot import draw
        print(f"wrote {draw(results, DIMENSIONS)}\n")


if __name__ == "__main__":
    main()
