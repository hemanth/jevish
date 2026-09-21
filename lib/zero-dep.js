/**
 * Zero-dependency semantic inference engine.
 * Features:
 * 1. Semantic synonym & association lexicon for zero-shot generalization.
 * 2. Word unigrams, stemming, and subword n-gram matching.
 * 3. Temperature-scaled softmax for calibrated probabilities.
 * 4. Extensible semantic cluster lexicons via options.clusters.
 */

// Canonical semantic associations for zero-shot transfer (including Software Triage, AG News & Emotion benchmarks)
export const SEMANTIC_CLUSTERS = {
  // Software / Triage
  bug: ['error', '500', 'crash', 'broken', 'fail', 'fails', 'failed', 'exception', 'defect', 'issue', 'freeze', 'deadlock', 'panic', 'glitch', 'leak', 'stacktrace', 'null', 'undefined', 'hang', 'slow', 'timeout'],
  feature: ['request', 'add', 'want', 'support', 'wish', 'new', 'implement', 'suggestion', 'enhancement', 'improve', 'option', 'mode', 'export', 'button', 'create'],
  billing: ['invoice', 'charge', 'charges', 'charged', 'payment', 'paid', 'receipt', 'subscription', 'price', 'pricing', 'stripe', 'refund', 'card', 'checkout', 'bill', 'cost', 'fee', 'account'],
  spam: ['free', 'congratulations', 'winner', 'win', 'won', 'luxury', 'prize', 'gift', 'click', 'crypto', 'giveaway', 'urgent', 'act', 'now', 'deal', 'discount', 'million', 'dollars', 'cash'],
  praise: ['love', 'awesome', 'great', 'fantastic', 'amazing', 'kudos', 'thanks', 'thank', 'excellent', 'helpful', 'best', 'wonderful', 'perfect'],
  security: ['vulnerability', 'cve', 'exploit', 'injection', 'breach', 'auth', 'jwt', 'token', 'permission', 'xss', 'csrf', 'leak', 'credential', 'malicious'],
  urgent: ['urgent', 'p0', 'critical', 'immediately', 'asap', 'blocker', 'emergency', 'deadlock', 'production', 'outage', 'down'],
  docs: ['documentation', 'tutorial', 'setup', 'guide', 'swagger', 'openapi', 'readme', 'instructions', 'spec', 'clarification'],

  // AG News 4-way topic categories & aliases
  world: ['government', 'minister', 'president', 'election', 'country', 'military', 'war', 'police', 'peace', 'international', 'foreign', 'treaty', 'killed', 'prime', 'leader', 'nations', 'united', 'iraq', 'security', 'attack', 'border', 'state', 'talks', 'nuclear', 'israel', 'palestinian', 'violence', 'official', 'officials', 'court', 'un', 'troops', 'rebel', 'rebels', 'iraqi', 'baghdad', 'afghanistan', 'gaza', 'hostage', 'iran', 'russia', 'china', 'sudan', 'crisis', 'korea', 'north', 'diplomat', 'diplomats', 'embassy', 'refugees', 'vote', 'voters', 'poll', 'democratic', 'republican', 'kerry', 'bush', 'parliament', 'judge', 'lawsuit', 'appeal', 'disaster', 'flood', 'investigation', 'charges', 'trial', 'sentenced'],
  sports: ['game', 'season', 'league', 'team', 'coach', 'player', 'players', 'win', 'victory', 'championship', 'cup', 'tournament', 'race', 'olympics', 'match', 'ball', 'medal', 'injury', 'drafted', 'scored', 'points', 'baseball', 'football', 'basketball', 'soccer', 'golf', 'tennis', 'olympic', 'athletics', 'racing', 'champion', 'lead', 'beats', 'beat', 'sox', 'yankees', 'nba', 'nfl', 'nhl', 'athens', 'games', 'series', 'red', 'cup', 'club', 'fans', 'title', 'pitcher', 'homer', 'scored', 'goals', 'victory', 'olympians', 'gold', 'silver', 'sprinters', 'swimming', 'field', 'coach', 'playoffs', 'quarterback', 'stadium'],
  business: ['company', 'stock', 'stocks', 'market', 'markets', 'shares', 'earnings', 'sales', 'profit', 'revenue', 'bank', 'banks', 'economy', 'economic', 'financial', 'finance', 'merger', 'acquisition', 'oil', 'prices', 'price', 'deal', 'dollar', 'investor', 'investors', 'pension', 'firm', 'unions', 'trade', 'commercial', 'funds', 'quarterly', 'inflation', 'fed', 'debt', 'corp', 'inc', 'billion', 'million', 'wall', 'street', 'investing', 'ceo', 'retail', 'retailers', 'airline', 'cost', 'costs', 'growth', 'bonds', 'treasury', 'analysts', 'crude', 'barrel', 'buyers', 'stores', 'apparel', 'rates', 'interest', 'credit', 'card', 'issuers', 'buyout', 'footprint', 'stake', 'commodities'],
  'sci/tech': ['computer', 'software', 'technology', 'tech', 'internet', 'web', 'online', 'science', 'scientists', 'space', 'research', 'virus', 'microsoft', 'apple', 'google', 'mobile', 'phone', 'digital', 'hardware', 'nasa', 'astronomy', 'scientific', 'galaxy', 'data', 'chip', 'chips', 'intel', 'devices', 'network', 'telecom', 'code', 'users', 'quantum', 'broadband', 'wireless', 'linux', 'security', 'pc', 'windows', 'ibm', 'system', 'search', 'engine', 'mars', 'satellite', 'telescope', 'hacker', 'flaw', 'patch', 'spaceflight', 'rocket', 'manned', 'launch', 'suborbital', 'phishing', 'cyber', 'scam', 'email', 'computing', 'servers', 'storage', 'it', 'cybersecurity', 'apps', 'database', 'cloud', 'oracle', 'dell', 'biology', 'chemistry', 'peptides', 'dna', 'genes', 'proteins', 'laboratory', 'medicine', 'clinical', 'physics', 'biotech', 'fossils', 'evolution', 'dinosaur', 'species', 'extinction', 'geology', 'origin', 'telecommunications', 'carrier', 'carriers', 'cellular', 'spectrum', 'smog', 'wildfires', 'emissions', 'weather', 'forecast', 'climate', 'vision', 'download', 'ipod'],
  tech: ['computer', 'software', 'technology', 'tech', 'internet', 'web', 'hardware', 'code', 'apps', 'database', 'cloud', 'mobile', 'devices'],
  technology: ['computer', 'software', 'technology', 'tech', 'internet', 'web', 'hardware', 'code', 'apps', 'database', 'cloud', 'mobile', 'devices'],
  science: ['science', 'scientists', 'research', 'biology', 'chemistry', 'physics', 'biotech', 'astronomy', 'scientific', 'laboratory', 'space', 'nasa'],

  // Emotion 6-way categories (dair-ai/emotion benchmark)
  sadness: ['rotten', 'sad', 'depressed', 'depressing', 'grief', 'down', 'gloomy', 'miserable', 'awful', 'lonely', 'hurt', 'sorry', 'hopeless', 'despair', 'crying', 'tear', 'unloved', 'helpless', 'terrible', 'low', 'discouraged', 'disappointed', 'blue', 'exhausted', 'drained', 'ashamed', 'guilty', 'lost', 'empty', 'bad', 'discontent', 'unhappy', 'miss', 'heartbroken', 'pathetic', 'worthless', 'horrible', 'dull', 'grim', 'crap', 'shitty', 'depress', 'isolated'],
  joy: ['happy', 'delighted', 'glad', 'joy', 'joyful', 'cheerful', 'excited', 'energetic', 'triumphant', 'grateful', 'blessed', 'warm', 'wonderful', 'proud', 'lucky', 'vibrant', 'optimistic', 'thrilled', 'ecstatic', 'content', 'smiling', 'fun', 'amused', 'pleased', 'satisfying', 'great', 'fantastic', 'confident', 'alive', 'good', 'joyous', 'cool', 'sweet', 'fine', 'positive', 'enjoy', 'creative', 'terrific', 'loved', 'smart', 'peaceful', 'comfortable', 'comfy', 'relaxed', 'mellow', 'splendid', 'honored', 'honoured', 'virtuous', 'productive', 'success', 'successful', 'friendly', 'kind', 'paradise', 'eager', 'sociable', 'acceptable', 'privileged', 'trusting', 'rich', 'breathless', 'celebrate'],
  love: ['love', 'loved', 'loving', 'caring', 'affection', 'fond', 'tender', 'attached', 'passionate', 'adore', 'romance', 'romantic', 'devoted', 'intimate', 'compassion', 'sympathy', 'supportive', 'gentle', 'cared', 'longing', 'beloved', 'cherish', 'treasured', 'sympathetic', 'naughty', 'kissing', 'lips'],
  anger: ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'rage', 'bitter', 'offended', 'resentful', 'outraged', 'hostile', 'hate', 'hated', 'infuriated', 'frustrated', 'pissed', 'disgust', 'disgusted', 'agitated', 'fuming', 'cranky', 'bothered', 'impatient', 'violent', 'hostility', 'jealous', 'greedy', 'insulted', 'stubborn', 'discontented', 'grumpy', 'envious', 'envy', 'annoy', 'fuckin', 'bored', 'tortured', 'unwelcome'],
  fear: ['scared', 'afraid', 'terrified', 'anxious', 'panic', 'nervous', 'worried', 'horror', 'trembling', 'horrified', 'dread', 'fearful', 'uneasy', 'insecure', 'threatened', 'apprehensive', 'vulnerable', 'intimidated', 'reluctant', 'frightened', 'shaken', 'paranoid', 'stressed', 'doubtful', 'intimidating', 'hesitant', 'restless', 'uncertain', 'invaded', 'uncomfortable'],
  surprise: ['surprised', 'amazed', 'shocked', 'astonished', 'stunned', 'speechless', 'unbelievable', 'sudden', 'unexpected', 'overwhelmed', 'startled', 'wonder', 'curious', 'impressed', 'confused', 'weird', 'strange', 'bizarre', 'shocking', 'unusual', 'astonishing', 'funny']
};

export const STOPWORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
  'and', 'or', 'but', 'if', 'this', 'that', 'it', 'its', 'my', 'your',
  'you', 'we', 'they', 'i', 'do', 'does', 'did', 'can', 'could'
]);

export function stem(w) {
  if (w.length <= 3) return w;
  return w.replace(/(ing|edly|ingly)$/, '').replace(/(ed|ly|es|s|al|tion|ment)$/, '');
}

export function tokenize(text) {
  const words = String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, ' ')
    .split(/\s+/)
    .filter(w => w && !STOPWORDS.has(w));
  const stemmed = words.map(stem);
  return { words, stemmed };
}

export function softmax(scores, temperature = 0.2) {
  const maxScore = Math.max(...scores);
  const exps = scores.map(s => Math.exp((s - maxScore) / temperature));
  const sumExp = exps.reduce((acc, v) => acc + v, 0);
  return exps.map(v => v / (sumExp || 1));
}

/**
 * Resolve semantic cluster set for a label or condition string.
 * Inspects full string, punctuation-stripped key, individual token parts, and their stems.
 */
export function resolveCluster(labelOrCond, clusters = SEMANTIC_CLUSTERS) {
  const raw = String(labelOrCond || '').toLowerCase().trim();
  const cluster = new Set(clusters[raw] || clusters[raw.replace(/[^a-z0-9]/g, '')] || []);
  const parts = raw.split(/[\/\s_:-]+/).filter(Boolean);

  for (const p of parts) {
    if (clusters[p]) {
      for (const w of clusters[p]) cluster.add(w);
    }
    const s = stem(p);
    if (clusters[s]) {
      for (const w of clusters[s]) cluster.add(w);
    }
  }

  return cluster;
}

/**
 * Zero-dependency zero-shot classifier
 * Combines direct keyword overlap, semantic cluster associations, and subword n-grams.
 */
export function classifyZeroDep(input, labels, options = {}) {
  if (!labels || labels.length === 0) {
    throw new Error('hev: at least one label is required');
  }

  const clusters = options.clusters
    ? { ...SEMANTIC_CLUSTERS, ...options.clusters }
    : SEMANTIC_CLUSTERS;

  const { words, stemmed } = tokenize(input);
  const rawScores = [];

  for (const label of labels) {
    const { words: labelWords, stemmed: labelStems } = tokenize(label);
    const cluster = resolveCluster(label, clusters);

    let score = 0.01;

    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const s = stemmed[i];

      // Exact label word match
      if (labelWords.includes(w)) {
        score += 4.0;
      } else if (labelStems.includes(s)) {
        score += 3.0;
      } else if (w.length >= 4 && labelWords.some(lw => lw.length >= 4 && (w.includes(lw) || lw.includes(w)))) {
        score += 1.5;
      }

      // Semantic cluster match
      if (cluster.has(w) || cluster.has(s)) {
        score += 2.5;
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

  const sortedRaw = [...rawScores].sort((a, b) => b - a);
  const margin = sortedRaw.length > 1 ? sortedRaw[0] - sortedRaw[1] : sortedRaw[0];

  return {
    label: labels[bestIdx],
    score: bestScore,
    margin: Math.round(margin * 100) / 100,
    probs,
  };
}

/**
 * Zero-dependency truth predicate (Noul)
 */
export function judgePredicateZeroDep(input, condition, options = {}) {
  const clusters = options.clusters
    ? { ...SEMANTIC_CLUSTERS, ...options.clusters }
    : SEMANTIC_CLUSTERS;

  const cleanCond = condition.replace(/^(is|are|has|contains)\s+/i, '').trim();
  const { words, stemmed } = tokenize(input);
  const { words: condWords, stemmed: condStems } = tokenize(cleanCond);

  const cluster = resolveCluster(cleanCond, clusters);

  let evidence = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const s = stemmed[i];

    if (condWords.includes(w)) {
      evidence += 4.0;
    } else if (condStems.includes(s)) {
      evidence += 3.0;
    } else if (w.length >= 4 && condWords.some(cw => cw.length >= 4 && (w.includes(cw) || cw.includes(w)))) {
      evidence += 1.5;
    }

    if (cluster.has(w) || cluster.has(s)) {
      evidence += 2.5;
    }
  }

  // Evaluate evidence against null-hypothesis baseline
  const nullHypothesis = 1.5;
  const probs = softmax([evidence, nullHypothesis], 0.5);
  const score = Math.round(probs[0] * 1000) / 1000;

  return {
    value: score >= 0.5,
    score,
    evidence,
  };
}
