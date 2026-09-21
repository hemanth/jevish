import { defaultEngine, Engine } from './engine.js';

/**
 * Parse a pattern key into label and optional confidence guard.
 * Examples:
 *   'bug'           => { label: 'bug', op: null, threshold: 0 }
 *   'bug @ >0.8'    => { label: 'bug', op: '>', threshold: 0.8 }
 *   'spam @ >= 0.9' => { label: 'spam', op: '>=', threshold: 0.9 }
 *   '_'             => { label: '_', isFallback: true }
 */
export function parsePatternKey(key) {
  const trimmed = key.trim();
  if (trimmed === '_' || trimmed === 'default') {
    return { label: '_', isFallback: true };
  }

  const guardMatch = trimmed.match(/^([^@]+)@\s*([><]=?)\s*([0-9.]+)\s*$/);
  if (guardMatch) {
    const label = guardMatch[1].trim();
    const op = guardMatch[2];
    const threshold = parseFloat(guardMatch[3]);
    return { label, op, threshold, isFallback: false };
  }

  return { label: trimmed, op: null, threshold: 0, isFallback: false };
}

/**
 * Evaluate if a confidence score satisfies a guard condition
 */
export function checkGuard(score, op, threshold) {
  if (!op) return true;
  switch (op) {
    case '>': return score > threshold;
    case '>=': return score >= threshold;
    case '<': return score < threshold;
    case '<=': return score <= threshold;
    default: return true;
  }
}

/**
 * Execute pattern matching over an object of handlers.
 * @param {string} input - Text to evaluate
 * @param {Record<string, Function|any>} branches - Pattern branches
 * @param {object} [options]
 */
export async function matchObject(input, branches, options = {}) {
  let engine = options.engine || defaultEngine;
  if (typeof engine === 'string') {
    engine = new Engine({ type: engine, ...options });
  }
  const entries = Object.entries(branches);
  const parsed = entries.map(([key, handler]) => ({
    key,
    handler,
    ...parsePatternKey(key),
  }));

  // Extract candidate labels (excluding fallbacks)
  const candidateLabels = [...new Set(
    parsed
      .filter(p => !p.isFallback)
      .map(p => p.label)
  )];

  if (candidateLabels.length === 0) {
    const fallback = parsed.find(p => p.isFallback);
    return typeof fallback?.handler === 'function' ? fallback.handler(input, {}) : fallback?.handler;
  }

  // If only 1 label is provided, contrast with 'other' for calibrated probability
  const queryLabels = candidateLabels.length === 1
    ? [candidateLabels[0], 'other']
    : candidateLabels;

  // Run model classification once for all candidates
  const result = await engine.classify(input, queryLabels);
  const { label: winLabel, score: winScore, probs, engine: engineUsed } = result;
  const meta = { label: winLabel, score: winScore, probs, engine: engineUsed };

  // Evaluate branches in order of definition
  for (const branch of parsed) {
    if (branch.isFallback) continue;
    if (branch.label.toLowerCase() === winLabel.toLowerCase()) {
      if (checkGuard(winScore, branch.op, branch.threshold)) {
        return typeof branch.handler === 'function' ? branch.handler(input, meta) : branch.handler;
      }
    }
  }

  // Fallback to default '_' if no guarded rule passed
  const fallback = parsed.find(p => p.isFallback);
  if (fallback) {
    return typeof fallback.handler === 'function' ? fallback.handler(input, meta) : fallback.handler;
  }

  return undefined;
}
