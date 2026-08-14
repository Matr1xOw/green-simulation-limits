import {
  gnsKernel,
  gnsKnown,
  imse,
  mean,
  pooled,
  simulate,
  standard,
} from "./gns.js";

const OUTER = 200;
const INNER = 20;
const TRIALS = 40;
const DIMENSIONS = [1, 2, 4, 8];

function main() {
  console.log("\nGreen nested simulation — known vs estimated density");
  console.log(
    `${OUTER} outer scenarios, ${INNER} inner paths each, budget ${OUTER * INNER}, ${TRIALS} trials\n`,
  );
  console.log("  d   standard   GNS known   GNS kernel   pooled mean");
  console.log("  " + "─".repeat(56));

  for (const d of DIMENSIONS) {
    const rows = { standard: [] as number[], known: [] as number[], kernel: [] as number[], flat: [] as number[] };
    for (let trial = 0; trial < TRIALS; trial++) {
      const e = simulate(1_000 + trial * 7, d, OUTER, INNER);
      rows.standard.push(imse(standard(e), e.kappa));
      rows.known.push(imse(gnsKnown(e), e.kappa));
      rows.kernel.push(imse(gnsKernel(e, d), e.kappa));
      rows.flat.push(imse(pooled(e), e.kappa));
    }
    const cell = (v: number[]) => mean(v).toFixed(4).padStart(11);
    console.log(
      `  ${d}` + cell(rows.standard) + cell(rows.known) + cell(rows.kernel) + cell(rows.flat),
    );
  }

  console.log(
    "\nIMSE against the closed form; lower is better. `standard` and `GNS known`\n" +
      "never touch the conditioning features, so their columns do not move with d.\n" +
      "`GNS kernel` climbing toward `standard` is the method losing everything it\n" +
      "was supposed to buy.\n",
  );
}

main();
