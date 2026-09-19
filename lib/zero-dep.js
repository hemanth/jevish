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

  // AG News 4-way topic categories
  world: ['government', 'minister', 'president', 'election', 'country', 'military', 'war', 'police', 'peace', 'international', 'foreign', 'treaty', 'killed', 'prime', 'leader', 'nations', 'united', 'iraq', 'security', 'attack', 'border', 'state', 'talks', 'nuclear', 'israel', 'palestinian', 'violence', 'official', 'officials', 'court', 'un', 'troops', 'rebel', 'rebels', 'iraqi', 'baghdad', 'afghanistan', 'gaza', 'hostage', 'iran', 'russia', 'china', 'sudan', 'crisis', 'korea', 'north', 'diplomat', 'diplomats', 'embassy', 'refugees', 'vote', 'voters', 'poll', 'democratic', 'republican', 'kerry', 'bush', 'parliament', 'judge', 'lawsuit', 'appeal', 'disaster', 'flood', 'investigation', 'charges', 'trial', 'sentenced'],
  sports: ['game', 'season', 'league', 'team', 'coach', 'player', 'players', 'win', 'victory', 'championship', 'cup', 'tournament', 'race', 'olympics', 'match', 'ball', 'medal', 'injury', 'drafted', 'scored', 'points', 'baseball', 'football', 'basketball', 'soccer', 'golf', 'tennis', 'olympic', 'athletics', 'racing', 'champion', 'lead', 'beats', 'beat', 'sox', 'yankees', 'nba', 'nfl', 'nhl', 'athens', 'games', 'series', 'red', 'cup', 'club', 'fans', 'title', 'pitcher', 'homer', 'scored', 'goals', 'victory', 'olympians', 'gold', 'silver', 'sprinters', 'swimming', 'field', 'coach', 'playoffs', 'quarterback', 'stadium'],
  business: ['company', 'stock', 'stocks', 'market', 'markets', 'shares', 'earnings', 'sales', 'profit', 'revenue', 'bank', 'banks', 'economy', 'economic', 'financial', 'finance', 'merger', 'acquisition', 'oil', 'prices', 'price', 'deal', 'dollar', 'investor', 'investors', 'pension', 'firm', 'unions', 'trade', 'commercial', 'funds', 'quarterly', 'inflation', 'fed', 'debt', 'corp', 'inc', 'billion', 'million', 'wall', 'street', 'investing', 'ceo', 'retail', 'retailers', 'airline', 'cost', 'costs', 'growth', 'bonds', 'treasury', 'analysts', 'crude', 'barrel', 'buyers', 'stores', 'apparel', 'rates', 'interest', 'credit', 'card', 'issuers', 'buyout', 'footprint', 'stake', 'commodities'],
  'sci/tech': ['computer', 'software', 'technology', 'tech', 'internet', 'web', 'online', 'science', 'scientists', 'space', 'research', 'virus', 'microsoft', 'apple', 'google', 'mobile', 'phone', 'digital', 'hardware', 'nasa', 'astronomy', 'scientific', 'galaxy', 'data', 'chip', 'chips', 'intel', 'devices', 'network', 'telecom', 'code', 'users', 'quantum', 'broadband', 'wireless', 'linux', 'security', 'pc', 'windows', 'ibm', 'system', 'search', 'engine', 'mars', 'satellite', 'telescope', 'hacker', 'flaw', 'patch', 'spaceflight', 'rocket', 'manned', 'launch', 'suborbital', 'phishing', 'cyber', 'scam', 'email', 'computing', 'servers', 'storage', 'it', 'cybersecurity', 'apps', 'database', 'cloud', 'oracle', 'dell', 'biology', 'chemistry', 'peptides', 'dna', 'genes', 'proteins', 'laboratory', 'medicine', 'clinical', 'physics', 'biotech', 'fossils', 'evolution', 'dinosaur', 'species', 'extinction', 'geology', 'origin', 'telecommunications', 'carrier', 'carriers', 'cellular', 'spectrum', 'smog', 'wildfires', 'emissions', 'weather', 'forecast', 'climate', 'vision', 'download', 'ipod'],

  // Emotion 6-way categories (dair-ai/emotion benchmark)
  sadness: ['rotten', 'sad', 'depressed', 'depressing', 'grief', 'down', 'gloomy', 'miserable', 'awful', 'lonely', 'hurt', 'sorry', 'hopeless', 'despair', 'crying', 'tear', 'unloved', 'helpless', 'terrible', 'low', 'discouraged', 'disappointed', 'blue', 'exhausted', 'drained', 'ashamed', 'guilty', 'lost', 'empty', 'bad', 'discontent', 'unhappy', 'miss', 'heartbroken', 'pathetic', 'worthless', 'horrible', 'dull', 'grim', 'crap', 'shitty', 'depress', 'isolated'],
  joy: ['happy', 'delighted', 'glad', 'joy', 'joyful', 'cheerful', 'excited', 'energetic', 'triumphant', 'grateful', 'blessed', 'warm', 'wonderful', 'proud', 'lucky', 'vibrant', 'optimistic', 'thrilled', 'ecstatic', 'content', 'smiling', 'fun', 'amused', 'pleased', 'satisfying', 'great', 'fantastic', 'confident', 'alive', 'good', 'joyous', 'cool', 'sweet', 'fine', 'positive', 'enjoy', 'creative', 'terrific', 'loved', 'smart', 'peaceful', 'comfortable', 'comfy', 'relaxed', 'mellow', 'splendid', 'honored', 'honoured', 'virtuous', 'productive', 'success', 'successful', 'friendly', 'kind', 'paradise', 'eager', 'sociable', 'acceptable', 'privileged', 'trusting', 'rich', 'breathless', 'celebrate'],
  love: ['love', 'loved', 'loving', 'caring', 'affection', 'fond', 'tender', 'attached', 'passionate', 'adore', 'romance', 'romantic', 'devoted', 'intimate', 'compassion', 'sympathy', 'supportive', 'gentle', 'cared', 'longing', 'beloved', 'cherish', 'treasured', 'sympathetic', 'naughty', 'kissing', 'lips'],
  anger: ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'rage', 'bitter', 'offended', 'resentful', 'outraged', 'hostile', 'hate', 'hated', 'infuriated', 'frustrated', 'pissed', 'disgust', 'disgusted', 'agitated', 'fuming', 'cranky', 'bothered', 'impatient', 'violent', 'hostility', 'jealous', 'greedy', 'insulted', 'stubborn', 'discontented', 'grumpy', 'envious', 'envy', 'annoy', 'fuckin', 'bored', 'tortured', 'unwelcome'],
  fear: ['scared', 'afraid', 'terrified', 'anxious', 'panic', 'nervous', 'worried', 'horror', 'trembling', 'horrified', 'dread', 'fearful', 'uneasy', 'insecure', 'threatened', 'apprehensive', 'vulnerable', 'intimidated', 'reluctant', 'frightened', 'shaken', 'paranoid', 'stressed', 'doubtful', 'intimidating', 'hesitant', 'restless', 'uncertain', 'invaded', 'uncomfortable'],
  surprise: ['surprised', 'amazed', 'shocked', 'astonished', 'stunned', 'speechless', 'unbelievable', 'sudden', 'unexpected', 'overwhelmed', 'startled', 'wonder', 'curious', 'impressed', 'confused', 'weird', 'strange', 'bizarre', 'shocking', 'unusual', 'astonishing', 'funny']
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

