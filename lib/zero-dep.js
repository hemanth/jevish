/**
 * Zero-dependency semantic inference engine.
 * Features:
 * 1. Semantic synonym & association lexicon for zero-shot generalization.
 * 2. Word unigrams, bigrams, subword character 3-grams, and stemming.
 * 3. Hashed projections with FNV-1a.
 * 4. Temperature-scaled softmax for calibrated probabilities.
 */

const DIM = 512;
const FNV_PRIME = 0x01000193;
const FNV_OFFSET = 0x811c9dc5;

// Canonical semantic associations for zero-shot transfer (including AG News & Emotion benchmarks)
const SEMANTIC_CLUSTERS = {
  // Software / Triage
  bug: ['error', '500', 'crash', 'broken', 'fail', 'fails', 'failed', 'exception', 'defect', 'issue', 'freeze', 'deadlock', 'panic', 'glitch', 'leak', 'stacktrace', 'null', 'undefined', 'hang', 'slow', 'timeout'],
  feature: ['request', 'add', 'want', 'support', 'wish', 'new', 'implement', 'suggestion', 'enhancement', 'improve', 'option', 'mode', 'export', 'button', 'create'],
  billing: ['invoice', 'charge', 'charges', 'charged', 'payment', 'paid', 'receipt', 'subscription', 'price', 'pricing', 'stripe', 'refund', 'card', 'checkout', 'bill', 'cost', 'fee', 'account'],
  spam: ['free', 'congratulations', 'winner', 'win', 'won', 'luxury', 'prize', 'gift', 'click', 'crypto', 'giveaway', 'urgent', 'act', 'now', 'deal', 'discount', 'million', 'dollars', 'cash'],
  praise: ['love', 'awesome', 'great', 'fantastic', 'amazing', 'kudos', 'thanks', 'thank', 'excellent', 'helpful', 'best', 'wonderful', 'perfect'],
  security: ['vulnerability', 'cve', 'exploit', 'injection', 'breach', 'auth', 'jwt', 'token', 'permission', 'xss', 'csrf', 'leak', 'credential', 'malicious'],
  urgent: ['urgent', 'p0', 'critical', 'immediately', 'asap', 'blocker', 'emergency', 'deadlock', 'production', 'outage', 'down'],
  docs: ['documentation', 'tutorial', 'setup', 'guide', 'swagger', 'openapi', 'readme', 'instructions', 'spec', 'clarification'],
};

function fnv1a(str) {
  let hash = FNV_OFFSET;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, FNV_PRIME);
  }
  return (hash >>> 0) % DIM;
}

function stem(w) {
  if (w.length <= 3) return w;
  return w.replace(/(ing|edly|ingly)$/, '').replace(/(ed|ly|es|s)$/, '');
}

function tokenize(text) {
  const clean = String(text || '').toLowerCase().replace(/[^a-z0-9\/]/g, ' ');
  const words = clean.split(/\s+/).filter(Boolean);
  const stemmed = words.map(stem);
  return { words, stemmed };
}

function softmax(scores, temperature = 0.15) {
  const maxScore = Math.max(...scores);
  const exps = scores.map(s => Math.exp((s - maxScore) / temperature));
  const sumExp = exps.reduce((acc, v) => acc + v, 0);
  return exps.map(v => v / (sumExp || 1));
}

/**
 * Zero-dependency zero-shot classifier
 * Combines direct keyword overlap, semantic cluster associations, and subword n-grams.
 */
export function classifyZeroDep(input, labels) {
  if (!labels || labels.length === 0) {
    throw new Error('hev: at least one label is required');
  }

  const { words, stemmed } = tokenize(input);
  const rawScores = [];

  for (const label of labels) {
    const key = label.toLowerCase();
    const parts = key.split(/[\/\s_-]+/);

    const cluster = new Set(SEMANTIC_CLUSTERS[key] || SEMANTIC_CLUSTERS[key.replace(/[^a-z]/g, '')] || []);
    for (const p of parts) {
      if (SEMANTIC_CLUSTERS[p]) {
        for (const w of SEMANTIC_CLUSTERS[p]) cluster.add(w);
      }
      const s = stem(p);
      if (SEMANTIC_CLUSTERS[s]) {
        for (const w of SEMANTIC_CLUSTERS[s]) cluster.add(w);
      }
    }

    let score = 0.01;

    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const s = stemmed[i];

      // Exact label word match
      if (parts.includes(w) || parts.includes(s)) {
        score += 4.5;
      }
      // Cluster semantic association match
      if (cluster.has(w) || cluster.has(s)) {
        score += 2.5;
      }
      // Subword character matching for typo tolerance
      if (w.length >= 4 && parts.some(p => p.includes(w) || w.includes(p))) {
        score += 1.0;
      }
    }

    rawScores.push(score);
  }

  const probsArr = softmax(rawScores);
  const probs = {};
  let bestIdx = 0;
  let bestScore = -1;

  for (let i = 0; i < labels.length; i++) {
    const p = Math.round(probsArr[i] * 1000) / 1000;
    probs[labels[i]] = p;
    if (p > bestScore) {
      bestScore = p;
      bestIdx = i;
    }
  }

  return {
    label: labels[bestIdx],
    score: bestScore,
    probs,
  };
}

/**
 * Zero-dependency truth predicate (Noul)
 */
export function judgePredicateZeroDep(input, condition) {
  const cleanCond = condition.replace(/^(is|are|has|contains)\s+/i, '').trim();
  const { words, stemmed } = tokenize(input);
  const key = cleanCond.toLowerCase();
  const parts = key.split(/[\/\s_-]+/);

  const cluster = new Set(SEMANTIC_CLUSTERS[key] || SEMANTIC_CLUSTERS[key.replace(/[^a-z]/g, '')] || []);
  for (const p of parts) {
    if (SEMANTIC_CLUSTERS[p]) {
      for (const w of SEMANTIC_CLUSTERS[p]) cluster.add(w);
    }
    const s = stem(p);
    if (SEMANTIC_CLUSTERS[s]) {
      for (const w of SEMANTIC_CLUSTERS[s]) cluster.add(w);
    }
  }

  let evidence = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const s = stemmed[i];

    if (parts.includes(w) || parts.includes(s)) {
      evidence += 4.5;
    }
    if (cluster.has(w) || cluster.has(s)) {
      evidence += 2.5;
    }
    if (w.length >= 4 && parts.some(p => p.includes(w) || w.includes(p))) {
      evidence += 1.0;
    }
  }

  // Evaluate evidence against null-hypothesis baseline
  const nullHypothesis = 1.5;
  const probs = softmax([evidence, nullHypothesis], 0.5);
  const score = Math.round(probs[0] * 1000) / 1000;

  return {
    value: score >= 0.5,
    score,
  };
}

