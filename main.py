"""Print the table, and optionally draw the chart.

    python main.py
    python main.py --plot
"""

import argparse

from gns import ESTIMATORS, study

DIMENSIONS = [1, 2, 4, 8]
OUTER, INNER, TRIALS = 200, 20, 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="also write imse.png")
    args = parser.parse_args()

    print("\nGreen nested simulation - known vs estimated density")
    print(f"{OUTER} outer scenarios, {INNER} inner paths each, "
          f"budget {OUTER * INNER}, {TRIALS} trials\n")

    header = "  d" + "".join(name.rjust(13) for name in ESTIMATORS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for d in DIMENSIONS:
        results[d] = study(d, OUTER, INNER, TRIALS)
        row = f"  {d}" + "".join(f"{results[d][n]:13.4f}" for n in ESTIMATORS)
        print(row)

    print("\nIMSE against the closed form; lower is better. `standard` and")
    print("`GNS known` never read the conditioning features, so their columns")
    print("do not move with d. `GNS kernel` climbing toward `standard` is the")
    print("method losing everything it was supposed to buy.\n")

    if args.plot:
        from plot import draw
        path = draw(results, DIMENSIONS)
        print(f"wrote {path}\n")


if __name__ == "__main__":
    main()
