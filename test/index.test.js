import test from 'node:test';
import assert from 'node:assert/strict';
import jevish, { hev, jevish as namedJevish, SEMANTIC_CLUSTERS } from '../index.js';

test('1. Zero-shot classification (array mode)', async () => {
  const result = await hev('The checkout button returns a 500 internal server error', [
    'bug report',
    'feature request',
    'billing inquiry',
  ]);

  assert.equal(result, 'bug report');
});

test('2. Pattern matching (object mode)', async () => {
  let executed = '';

  const action = await hev('Can I get an invoice for last month charges?', {
    'bug report': (t) => { executed = 'jira'; return 'jira'; },
    'billing inquiry': (t) => { executed = 'stripe'; return 'stripe'; },
    'feature request': (t) => { executed = 'linear'; return 'linear'; },
    _: (t) => { executed = 'support'; return 'support'; },
  });

  assert.equal(action, 'stripe');
  assert.equal(executed, 'stripe');
});

test('3. Confidence guards (@ >threshold)', async () => {
  // Input clearly matches bug
  const highConf = await hev('Fatal error: database connection refused on checkout', {
    'bug @ >0.5': (t, meta) => `high-priority-bug:${meta.score}`,
    'bug': () => 'low-priority-bug',
    _: () => 'fallback',
  });

  assert.match(highConf, /^high-priority-bug:/);

  // Fallback when guard threshold fails (>0.999 is impossible for this)
  const guardedFallback = await hev('The app button could maybe look slightly greener', {
    'bug @ >0.999': () => 'impossible-bug',
    _: () => 'fell-through',
  });

  assert.equal(guardedFallback, 'fell-through');
});

test('4. Boolean predicate (Noul mode)', async () => {
  const isSpam1 = await hev('Congratulations! You won a free luxury iPhone, click here immediately!', 'is spam');
  const isSpam2 = await hev('Here is the weekly engineering sync agenda for tomorrow.', 'is spam');

  assert.equal(isSpam1, true);
  assert.equal(isSpam2, false);
});

test('5. Auto-currying in functional pipelines', async () => {
  const triage = hev(['bug report', 'feature request']);
  
  const tickets = [
    'Null pointer exception in auth middleware',
    'Add dark mode support to the settings page',
  ];

  const results = await Promise.all(tickets.map(triage));
  assert.deepEqual(results, ['bug report', 'feature request']);
});

test('6. Curried pattern matcher', async () => {
  const router = hev({
    'bug report': () => 'file-bug',
    'feature request': () => 'file-feature',
  });

  assert.equal(await router('Crash on startup'), 'file-bug');
  assert.equal(await router('Please add export to CSV button'), 'file-feature');
});

test('7. Metadata access via hev.detailed()', async () => {
  const meta = await hev.detailed('Database deadlock on write', ['bug report', 'feature request']);

  assert.equal(meta.label, 'bug report');
  assert.equal(typeof meta.score, 'number');
  assert.ok(meta.score > 0.5);
  assert.ok(typeof meta.probs === 'object');
  assert.ok('bug report' in meta.probs);
  assert.ok(['builtin', 'typesafe', 'gliner'].includes(meta.engine));
});

test('8. Probability score via hev.score()', async () => {
  const score = await hev.score('Urgent security vulnerability in JWT parsing', 'security issue');
  assert.equal(typeof score, 'number');
  assert.ok(score >= 0 && score <= 1);
});

test('9. Handler receives (text, meta)', async () => {
  let passedMeta = null;
  await hev('System memory leak detected', {
    'bug report': (text, meta) => {
      passedMeta = meta;
      return 'handled';
    },
  });

  assert.ok(passedMeta);
  assert.equal(passedMeta.label, 'bug report');
  assert.ok(passedMeta.score > 0);
});

test('10. Hardware device detection & metadata reporting', async () => {
  const currentDevice = hev.device();
  assert.ok(['cuda', 'mps', 'cpu', 'webgpu', 'wasm'].includes(currentDevice));

  const meta = await hev.detailed('Fatal crash in memory manager', ['bug', 'feature']);
  assert.ok(['cpu', 'mps', 'cuda', 'cloud'].includes(meta.device));
});

test('11. Speculative cascade execution mode', async () => {
  const meta = await hev.detailed('Checkout crashed with 500 internal server error', [
    'bug: 500 error crash',
    'billing: payment invoice'
  ], { cascade: true });

  assert.equal(meta.label, 'bug: 500 error crash');
  assert.equal(meta.engine, 'cascade');
  assert.equal(typeof meta.fastPath, 'boolean');
  assert.equal(meta.fastPath, true);
});

test('12. Module exports both default jevish and hev', async () => {
  assert.equal(typeof jevish, 'function');
  assert.equal(typeof namedJevish, 'function');
  assert.equal(typeof hev, 'function');
  assert.equal(jevish, hev);
});

test('13. webml-kit decision engine option', async () => {
  // Classification via webml engine option
  const result = await hev('Internal 500 error in database queries on checkout', [
    'bug',
    'billing',
    'feature',
  ], { engine: 'webml' });
  assert.equal(result, 'bug');

  // Metadata via detailed with webml engine
  const meta = await hev.detailed('Customer requested refund for duplicate charge on invoice', [
    'bug',
    'billing',
    'feature',
  ], { engine: 'webml' });
  assert.equal(meta.label, 'billing');
  assert.equal(meta.engine, 'webml');
  assert.ok(meta.score > 0.5);
  assert.ok(meta.probs['billing'] > meta.probs['bug']);

  // Pattern matching with confidence guard via webml engine
  const action = await hev('Production database crash 500 error!', {
    'bug @ >0.7': () => 'p0-bug',
    'bug': () => 'review-bug',
    _: () => 'fallback',
  }, { engine: 'webml' });
  assert.equal(action, 'p0-bug');

  // Boolean predicate via webml engine
  const isBug = await hev('System crashed with fatal exception', 'is this a software bug?', { engine: 'webml' });
  assert.equal(isBug, true);
});

test('14. Multi-word and compound labels resolve clusters properly', async () => {
  // Phrases with non-first-word cluster keywords
  const result = await hev('I was charged twice on my invoice for this service', [
    'customer billing',
    'software bug',
  ]);
  assert.equal(result, 'customer billing');

  const compound = await hev('New microprocessor quantum computing architecture announced', [
    'sci/tech',
    'sports',
    'world',
  ]);
  assert.equal(compound, 'sci/tech');
});

test('15. Custom user-injected semantic clusters via options.clusters', async () => {
  // Domain outside built-ins: e-commerce logistics
  const customClusters = {
    shipping: ['tracking', 'parcel', 'courier', 'fedex', 'ups', 'package', 'delivered', 'dispatch'],
    returns: ['refund', 'exchange', 'rma', 'sendback', 'receipt'],
  };

  // Zero-shot array mode with custom clusters
  const label = await hev('Where is my parcel right now? It was dispatched yesterday.', [
    'shipping',
    'returns',
    'support',
  ], { clusters: customClusters });
  assert.equal(label, 'shipping');

  // Object pattern matching with custom clusters
  let handled = '';
  await hev('Please process my refund for item exchange', {
    'shipping': () => { handled = 'shipping'; },
    'returns': () => { handled = 'returns'; },
    _: () => { handled = 'fallback'; },
  }, { clusters: customClusters });
  assert.equal(handled, 'returns');

  // Boolean predicate with custom clusters
  const isShipping = await hev('My package is delayed at the courier facility', 'is shipping inquiry', {
    clusters: customClusters,
  });
  assert.equal(isShipping, true);
});

test('16. Canonical news and emotion clusters are active in zero-dep', async () => {
  const news = await hev('The prime minister signed international peace treaty after talks', [
    'world',
    'sports',
    'business',
  ]);
  assert.equal(news, 'world');

  const emotion = await hev('I feel so exhausted, heartbroken and discouraged today', [
    'sadness',
    'joy',
    'anger',
  ]);
  assert.equal(emotion, 'sadness');
});

test('17. SEMANTIC_CLUSTERS export is available and extensible', () => {
  assert.ok(typeof SEMANTIC_CLUSTERS === 'object');
  assert.ok(Array.isArray(SEMANTIC_CLUSTERS.bug));
  assert.ok(Array.isArray(SEMANTIC_CLUSTERS.billing));
  assert.ok(Array.isArray(SEMANTIC_CLUSTERS.world));
  assert.ok(Array.isArray(SEMANTIC_CLUSTERS.sadness));
  assert.equal(jevish.SEMANTIC_CLUSTERS, SEMANTIC_CLUSTERS);
});

test('18. GLiNER local neural cascade escalation on ambiguous inputs', async () => {
  const glinerMock = {
    async classify(input, labels) {
      return {
        label: 'cardiology',
        score: 0.95,
        probs: { cardiology: 0.95, orthopedics: 0.05 },
        margin: 0.90,
      };
    },
    async predicate(input, condition) {
      return {
        value: true,
        score: 0.92,
      };
    }
  };

  // Ambiguous out-of-vocabulary input triggers cascade to GLiNER
  const meta = await hev.detailed('Patient has acute myocardial infarction', ['cardiology', 'orthopedics'], {
    cascade: true,
    gliner: glinerMock,
  });

  assert.equal(meta.label, 'cardiology');
  assert.equal(meta.engine, 'cascade');
  assert.equal(meta.fastPath, false);
});


