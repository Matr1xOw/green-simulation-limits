import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  cdf,
  gnsKernel,
  gnsKnown,
  imse,
  mean,
  normals,
  pooled,
  rng,
  simulate,
  standard,
  truth,
  weightedMean,
} from "./gns.js";

describe("closed form", () => {
  it("matches a plain Monte Carlo estimate", () => {
    // If the analytic truth were wrong every IMSE below would be meaningless,
    // so it is checked against brute force before anything is built on it.
    const next = normals(rng(42));
    const kappa = 0.3;
    const draws: number[] = [];
    for (let i = 0; i < 400_000; i++) {
      draws.push(Math.max(kappa + 0.02 + next() - 0.5, 0));
    }
    assert.ok(Math.abs(mean(draws) - truth(kappa)) < 0.01);
  });

  it("has a sane normal CDF", () => {
    assert.ok(Math.abs(cdf(0) - 0.5) < 1e-9);
    assert.ok(Math.abs(cdf(1.96) - 0.975) < 1e-3);
    assert.ok(Math.abs(cdf(-1.96) - 0.025) < 1e-3);
  });
});

describe("weightedMean", () => {
  it("self-normalises, so weight scale does not matter", () => {
    const values = [1, 2, 3];
    assert.equal(weightedMean(values, [1, 1, 1]), weightedMean(values, [7, 7, 7]));
  });

  it("is NaN rather than zero when every weight vanishes", () => {
    assert.ok(Number.isNaN(weightedMean([1, 2], [0, 0])));
  });
});

describe("estimators", () => {
  const experiment = simulate(1_234, 1, 200, 20);

  it("all produce one estimate per outer scenario", () => {
    for (const e of [standard, gnsKnown, pooled]) {
      assert.equal(e(experiment).length, experiment.kappa.length);
    }
    assert.equal(gnsKernel(experiment, 1).length, experiment.kappa.length);
  });

  it("reproduces the paper: reuse beats discarding paths", () => {
    assert.ok(
      imse(gnsKnown(experiment), experiment.kappa) <
        imse(standard(experiment), experiment.kappa) / 10,
      "GNS with a known density should be an order of magnitude better",
    );
  });

  it("beats the unconditional mean, which is the floor", () => {
    assert.ok(
      imse(gnsKnown(experiment), experiment.kappa) <
        imse(pooled(experiment), experiment.kappa),
    );
  });

  it("loses its advantage as irrelevant dimensions are added", () => {
    // The result the repo exists to report.
    const low = imse(gnsKernel(simulate(99, 1, 200, 20), 1), simulate(99, 1, 200, 20).kappa);
    const high = imse(gnsKernel(simulate(99, 8, 200, 20), 8), simulate(99, 8, 200, 20).kappa);
    assert.ok(high > low * 5, `d=8 (${high}) should be far worse than d=1 (${low})`);
  });

  it("is deterministic", () => {
    assert.deepEqual(simulate(7, 2, 50, 5), simulate(7, 2, 50, 5));
  });
});
