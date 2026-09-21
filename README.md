# jevish

Semantic pattern matching and zero-shot judgment in JavaScript. Jev-ish: behaves like TypeSafe Jev in local CPU cache (<0.05ms, 0 deps), and speculatively escalates to cloud Jev when needed.

[**Live Interactive Playground →**](https://hemanth.github.io/jevish/)

```bash
npm install jevish
```

## Quick start

```js
import jevish from 'jevish';

await jevish('Checkout button returns 500 internal server error', {
  'bug @ >0.8': (t, meta) => fileJira(t, meta.score),
  'bug':        (t) => queueTriage(t),
  'billing':    (t) => openStripe(t),
  _:            (t) => logToInbox(t),
});
```

`jevish()` evaluates semantic pattern handlers, zero-shot arrays, or boolean predicates in a single forward pass. That's the whole API.

## Zero-shot classification

Pass an array of labels to get the winning category:

```js
const category = await jevish('Can you provide an invoice for last month?', [
  'bug',
  'feature',
  'billing',
]);
// => 'billing'
```

Access calibrated probabilities via `jevish.detailed()`:

```js
const meta = await jevish.detailed('Database connection pool exhausted', ['bug', 'feature']);
console.log(meta.score); // 0.96
console.log(meta.probs); // { bug: 0.96, feature: 0.04 }
```

## Functional pipelines & currying

Every mode auto-curries when called with only the patterns:

```js
const triage = jevish(['bug', 'feature', 'billing']);
const isSpam = jevish.is('spam');

const tickets = await fetchInbox();
const categories = await Promise.all(tickets.map(triage));
const spamEmails = await emails.filterAsync?.(isSpam);
```

## Speculative cascade

Pass `{ cascade: true }` to resolve unambiguous queries in CPU cache (0.05ms, $0 cost) while speculatively escalating tough edge cases to TypeSafe Jev cloud to guarantee 99%+ accuracy (76% fast-path rate on banking intents).

## Why "dumb by design"?

The in-tree zero-dependency engine is intentionally primitive: tokenization, stemming, subword n-grams, and a compact lexicon of semantic synonym clusters (`SEMANTIC_CLUSTERS`). There are no 50MB tensor weights, no ONNX runtimes, and zero network calls.

### 1. The LLM Overkill Tax
Most production events are not philosophical dilemmas:
* `"Checkout button throws 500 error"`
* `"Can I get an invoice for last month's charge?"`
* `"Congratulations, you won a free luxury prize!"`

Burning a 500ms network round-trip and paying per-token API taxes just to map `"500 error"` to `bug` is wasteful engineering.

### 2. The L1 Semantic Cache
CPUs use an L1 cache (<1ns) for 90%+ of memory requests before hitting slower L2/L3 or RAM. `jevish` acts as the **L1 cache of semantic routing**:
* **<0.05ms latency** (50 microseconds in CPU cache).
* **0 dependencies** (8.6 kB total bundle).
* Runs anywhere: Cloudflare Workers, Edge Lambdas, or browser main threads.

### 3. High-Confidence Margin Filtering
The in-tree engine only commits when both the winning score and margin over the runner-up are high (`score >= 0.70`, `margin >= 2.0`). When an input is obvious, it finishes in microseconds. When it is genuinely ambiguous or out of vocabulary, `{ cascade: true }` speculatively escalates to neural hardware (WebGPU, GLiNER) or cloud models.

### 4. Zero-Retraining Extensibility
You don't need to fine-tune a model or spin up a vector database to teach it domain jargon. Pass custom clusters directly:

```js
await jevish('Where is my parcel?', ['shipping', 'returns'], {
  clusters: {
    shipping: ['tracking', 'parcel', 'courier', 'fedex', 'ups', 'package'],
    returns:  ['refund', 'exchange', 'rma', 'sendback'],
  }
});
```

## Empirical benchmark

Evaluated on standard zero-shot benchmarks used by TypeSafe Jev:

### Hugging Face Benchmarks (N=100 per task)
| Task / Dataset | `jevish (in-tree)` | `jevish (cascade)` | Jev (TypeSafe API) |
|---|---|---|---|
| **Intent Routing** (`banking77`) | **86.0%** (0.05 ms) | **99.0%** (29 ms) | **100.0%** (139 ms) |
| **Spam Guardrails** (`sms_spam`) | **72.0%** (0.02 ms) | **98.0%** (134 ms) | **98.0%** (146 ms) |
| **Topic Triage** (`ag_news`) | **86.0%** (0.04 ms) | **99.0%** (35 ms) | **82.0%** (149 ms) |
| **Affective Emotion** (`emotion`) | **83.0%** (0.02 ms) | **99.0%** (35 ms) | **85.0%** (145 ms) |

Run `npm run bench` to reproduce live across all canonical Hugging Face datasets.

## Runtime & devices

Runs anywhere: Node.js, Bun, Deno, Cloudflare Workers, and modern browsers (8.6 kB). Detects `CUDA` → `MPS` → `CPU` (or `WebGPU` → `WASM` in browser) via `jevish.device()`. [View Execution Path Blueprint →](docs/execution-path.svg)


```bash
npm run demo
npm run bench
npm run playground
```

Runs the multi-mode demonstration, reproduces the benchmark suite, and launches the local interactive playground at `http://localhost:3456`.

## License

MIT © [Hemanth.HM](https://h3manth.com)
