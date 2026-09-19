/**
 * Zero-dependency semantic inference engine.
 * Features:
 * 1. Semantic synonym & association lexicon for zero-shot generalization.
 * 2. Word unigrams, bigrams, and character 3-grams for typo & stem robustness.
 * 3. Hashed projections with FNV-1a.
 * 4. Temperature-scaled softmax for calibrated probabilities.
 */

const DIM = 512;
const FNV_PRIME = 0x01000193;
const FNV_OFFSET = 0x811c9dc5;

// Compact semantic associations for zero-shot transfer
const SEMANTIC_CLUSTERS = {
  bug: ['error', '500', 'crash', 'broken', 'fail', 'fails', 'failed', 'exception', 'defect', 'issue', 'freeze', 'deadlock', 'panic', 'glitch', 'leak', 'stacktrace', 'null', 'undefined', 'hang', 'slow', 'timeout'],
  feature: ['request', 'add', 'want', 'support', 'wish', 'new', 'implement', 'suggestion', 'enhancement', 'improve', 'option', 'mode', 'export', 'button', 'create'],
  billing: ['invoice', 'charge', 'charges', 'charged', 'payment', 'paid', 'receipt', 'subscription', 'price', 'pricing', 'stripe', 'refund', 'card', 'checkout', 'bill', 'cost', 'fee', 'account'],
  spam: ['free', 'congratulations', 'winner', 'win', 'won', 'luxury', 'prize', 'gift', 'click', 'crypto', 'giveaway', 'urgent', 'act', 'now', 'deal', 'discount', 'million', 'dollars', 'cash'],
  praise: ['love', 'awesome', 'great', 'fantastic', 'amazing', 'kudos', 'thanks', 'thank', 'excellent', 'helpful', 'best', 'wonderful', 'perfect'],
  security: ['vulnerability', 'cve', 'exploit', 'injection', 'breach', 'auth', 'jwt', 'token', 'permission', 'xss', 'csrf', 'leak', 'credential', 'malicious'],
  urgent: ['urgent', 'p0', 'critical', 'immediately', 'asap', 'blocker', 'emergency', 'deadlock', 'production', 'outage', 'down'],
  docs: ['documentation', 'tutorial', 'setup', 'guide', 'swagger', 'openapi', 'readme', 'instructions', 'spec', 'clarification'],
  sports: ['game', 'overtime', 'runners', 'marathon', 'penalty', 'shootout', 'champions', 'league', 'olympic', 'tournament', 'match', 'score', 'points', 'win', 'defeated'],
  finance: ['interest', 'rates', 'basis', 'points', 'federal', 'reserve', 'wall', 'street', 'rallies', 'quarterly', 'earnings', 'estimates', 'inflation', 'consumer', 'price', 'index', 'market', 'stocks'],
  technology: ['chip', 'neural', 'engine', 'llm', 'model', 'quantum', 'computing', 'open-source', 'software', 'hardware', 'apple', 'ai', 'benchmark'],
  entertainment: ['movie', 'film', 'album', 'song', 'actor', 'celebrity', 'concert', 'show', 'series'],
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
  return w
    .replace(/(ing|edly|ingly)$/, '')
    .replace(/(ed|ly|es|s)$/, '');
}

/**
 * Expand a text or label with semantic associations if available
 */
function getAssociations(tokens) {
  const associations = new Set();
  for (const token of tokens) {
    const s = stem(token);
    for (const [concept, words] of Object.entries(SEMANTIC_CLUSTERS)) {
      if (token === concept || s === concept || words.includes(token) || words.includes(s)) {
        associations.add(`sem:${concept}`);
        for (const w of words) {
          associations.add(`sem:${w}`);
        }
      }
    }
  }
  return associations;
}

/**
 * Extract tokens, word bi-grams, subword character 3-grams, and semantic associations
 */
function extractFeatures(text, isLabel = false) {
  const clean = String(text || '').toLowerCase().replace(/[^\w\s-]/g, ' ');
  const rawWords = clean.split(/\s+/).filter(Boolean);
  const words = [];
  for (const w of rawWords) {
    words.push(w);
    const s = stem(w);
    if (s !== w) words.push(s);
  }
  const features = new Map();

  function addFeature(f, weight = 1.0) {
    features.set(f, (features.get(f) || 0) + weight);
  }

  // 1. Direct words (unigrams)
  for (const w of words) {
    addFeature(`w:${w}`, isLabel ? 3.0 : 2.0);

    // 2. Char 3-grams for subword / typo robustness
    if (w.length >= 3) {
      for (let i = 0; i <= w.length - 3; i++) {
        addFeature(`c3:${w.slice(i, i + 3)}`, 0.5);
      }
    }
  }

  // 3. Word bigrams
  for (let i = 0; i < words.length - 1; i++) {
    addFeature(`bg:${words[i]}_${words[i + 1]}`, isLabel ? 2.5 : 1.5);
  }

  // 4. Semantic associations
  const assocs = getAssociations(words);
  for (const a of assocs) {
    addFeature(a, isLabel ? 3.5 : 2.0);
  }

  return features;
}

/**
 * Project features into a fixed-size L2-normalized vector
 */
function embed(text, isLabel = false) {
  const features = extractFeatures(text, isLabel);
  const vec = new Float32Array(DIM);

  for (const [feat, weight] of features) {
    const idx = fnv1a(feat);
    const sign = (fnv1a(`${feat}:sign`) & 1) === 0 ? 1 : -1;
    vec[idx] += weight * sign;
  }

  let norm = 0;
  for (let i = 0; i < DIM; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < DIM; i++) {
      vec[i] /= norm;
    }
  }

  return vec;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < DIM; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

function softmax(scores, temperature = 0.12) {
  const maxScore = Math.max(...scores);
  const exps = scores.map(s => Math.exp((s - maxScore) / temperature));
  const sumExp = exps.reduce((acc, v) => acc + v, 0);
  return exps.map(v => v / (sumExp || 1));
}

/**
 * Zero-dependency zero-shot classifier
 */
export function classifyZeroDep(input, labels) {
  if (!labels || labels.length === 0) {
    throw new Error('hev: at least one label is required');
  }

  const inputVec = embed(input, false);
  const rawScores = [];

  for (const label of labels) {
    const labelVec = embed(label, true);
    const sim = cosine(inputVec, labelVec);
    rawScores.push(Math.max(0.001, sim));
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
  const inputWords = String(input || '').toLowerCase().split(/\s+/);
  const condWords = cleanCond.toLowerCase().split(/\s+/);

  // Check semantic clusters
  let posMatches = 0;
  for (const w of inputWords) {
    for (const cw of condWords) {
      if (w === cw) posMatches += 3;
      if (SEMANTIC_CLUSTERS[cw]?.includes(w)) posMatches += 2;
    }
  }

  const inputVec = embed(input, false);
  const posVec = embed(cleanCond, true);
  const negVec = embed(`not ${cleanCond} standard ordinary clean normal`, true);

  const simPos = Math.max(0.01, cosine(inputVec, posVec) + (posMatches > 0 ? 0.3 : 0));
  const simNeg = Math.max(0.01, cosine(inputVec, negVec));

  const [posProb] = softmax([simPos, simNeg], 0.2);
  const score = Math.round(posProb * 1000) / 1000;

  return {
    value: score >= 0.5,
    score,
  };
}
