/**
 * Green nested simulation, and the same method with an estimated density.
 *
 * Feng, Li & Zhou (2022), "Green nested simulation via likelihood ratio:
 * Applications to longevity risk management", Insurance: Mathematics and
 * Economics 106, 285–301.
 *
 * The problem is estimating V(κ) = E[H | κ] across many outer scenarios. The
 * naive approach simulates N inner paths per scenario and throws them away.
 * GNS pools all of them and reweights by a likelihood ratio f(x|κ)/g(x), so
 * every path informs every scenario.
 *
 * It needs f. The paper has one because its risk factor is a random walk with
 * drift. This module also implements the version you are forced into when
 * there is no generative model — kernel similarity over the conditioning
 * state — so the two can be measured against each other.
 */

export const MU = 0.02;
export const SIGMA = 1;
export const STRIKE = 0.5;

/** Standard normal density. */
export const pdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Standard normal CDF — Abramowitz & Stegun 7.1.26. */
export function cdf(z: number): number {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

/** Call-style payoff, as in the paper's K-call case study. */
export const payoff = (x: number) => Math.max(x - STRIKE, 0);

/**
 * The exact answer.
 *
 * X ~ N(κ + μ, σ²), so E[max(X − K, 0)] is Bachelier's formula. Having this in
 * closed form is what makes the comparison below a measurement rather than
 * another estimate.
 */
export function truth(kappa: number): number {
  const m = kappa + MU;
  const z = (m - STRIKE) / SIGMA;
  return (m - STRIKE) * cdf(z) + SIGMA * pdf(z);
}

export const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;

/** Self-normalised weighted average — the paper's recommended variant. */
export function weightedMean(values: number[], weights: number[]): number {
  let total = 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    total += weights[i];
    sum += weights[i] * values[i];
  }
  return total > 0 ? sum / total : NaN;
}

/** Seeded LCG, so every number in the README is reproducible. */
export function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1_103_515_245 + 12_345) % 2_147_483_648;
    return s / 2_147_483_648;
  };
}

/** Box-Muller on top of the LCG. */
export function normals(next: () => number) {
  return () => {
    const u = Math.max(next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
}

export type Experiment = {
  /** Outer scenarios: the conditioning value that drives the payoff. */
  kappa: number[];
  /** Additional conditioning features that carry no information. */
  noise: number[][];
  /** Pooled inner draws, their payoffs, and which scenario produced each. */
  draws: number[];
  payoffs: number[];
  source: number[];
};

export function simulate(
  seed: number,
  dimensions: number,
  outer: number,
  inner: number,
): Experiment {
  const next = normals(rng(seed));

  const kappa: number[] = [];
  const noise: number[][] = [];
  for (let i = 0; i < outer; i++) {
    kappa.push(next());
    noise.push(Array.from({ length: dimensions - 1 }, () => next()));
  }

  const draws: number[] = [];
  const payoffs: number[] = [];
  const source: number[] = [];
  for (let i = 0; i < outer; i++) {
    for (let n = 0; n < inner; n++) {
      const x = kappa[i] + MU + SIGMA * next();
      draws.push(x);
      payoffs.push(payoff(x));
      source.push(i);
    }
  }

  return { kappa, noise, draws, payoffs, source };
}

/** Integrated mean squared error against the closed form. */
export const imse = (estimates: number[], kappa: number[]) =>
  mean(estimates.map((e, i) => (e - truth(kappa[i])) ** 2));

/** Standard nested simulation: each scenario keeps only its own inner paths. */
export function standard(e: Experiment): number[] {
  const buckets: number[][] = e.kappa.map(() => []);
  for (let j = 0; j < e.draws.length; j++) buckets[e.source[j]].push(e.payoffs[j]);
  return buckets.map(mean);
}

/**
 * GNS with the density known, as in the paper.
 *
 * The denominator is the equal mixture over all scenarios rather than a
 * pairwise ratio to one reference. That is the choice that keeps the weights
 * finite when two scenarios are far apart.
 */
export function gnsKnown(e: Experiment): number[] {
  const outer = e.kappa.length;
  const mixture = e.draws.map((x) => {
    let total = 0;
    for (const k of e.kappa) total += pdf((x - k - MU) / SIGMA) / SIGMA;
    return total / outer;
  });

  return e.kappa.map((k) => {
    const weights = e.draws.map(
      (x, j) => pdf((x - k - MU) / SIGMA) / SIGMA / mixture[j],
    );
    return weightedMean(e.payoffs, weights);
  });
}

/**
 * GNS with the density estimated.
 *
 * With no generative model the likelihood ratio has to be replaced by
 * something learned from the sample, and kernel similarity over the
 * conditioning state is the natural choice. The bandwidth follows the usual
 * Silverman-style rate — and the exponent is where dimension does its damage.
 */
export function gnsKernel(e: Experiment, dimensions: number): number[] {
  const budget = e.draws.length;
  const h = Math.pow(budget, -1 / (dimensions + 4));
  const state = (i: number) => [e.kappa[i], ...e.noise[i]];

  return e.kappa.map((_, i) => {
    const target = state(i);
    const weights = e.source.map((s) => {
      const from = state(s);
      let squared = 0;
      for (let d = 0; d < dimensions; d++) squared += (target[d] - from[d]) ** 2;
      return Math.exp((-0.5 * squared) / (h * h));
    });
    return weightedMean(e.payoffs, weights);
  });
}

/** The floor: ignore the conditioning entirely. */
export function pooled(e: Experiment): number[] {
  const flat = mean(e.payoffs);
  return e.kappa.map(() => flat);
}
