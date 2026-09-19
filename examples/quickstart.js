import hev from '../index.js';

console.log('--- 1. Pattern Matching with Handlers ---');
const action = await hev('Checkout button is completely broken and throwing 500s', {
  'bug @ >0.8': (t, meta) => `🚨 Filed high-severity Jira ticket (score: ${meta.score})`,
  'bug': (t, meta) => `📝 Queued bug for review`,
  'feature': (t) => `💡 Added to product backlog`,
  _: (t) => `📥 Logged to general inbox`,
});
console.log(action);

console.log('\n--- 2. Zero-Shot Classification (Array Mode) ---');
const category = await hev('Can you provide an invoice for last month charges?', [
  'bug',
  'feature',
  'billing',
]);
console.log(`Category: ${category}`);

console.log('\n--- 3. Boolean Predicate (Noul Mode) ---');
const isSpam = await hev('Claim your free luxury crypto reward now!!', 'is spam');
console.log(`Is spam: ${isSpam}`);

console.log('\n--- 4. Functional Currying in Array Pipelines ---');
const triage = hev(['bug', 'feature', 'billing']);
const tickets = [
  'Database query deadlocks on concurrent updates',
  'Please add keyboard shortcut for quick navigation',
  'Credit card payment failed to process',
];

const results = await Promise.all(tickets.map(triage));
console.log(results);
