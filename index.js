import { defaultEngine, webmlEngine, Engine, detectDevice } from './lib/engine.js';
import { matchObject } from './lib/matcher.js';

/**
 * Main hev function: Semantic pattern matching and zero-shot judgment.
 * Detects intent from input types and supports auto-currying.
 *
 * @param {string|Array|Record<string, any>} arg1 - Input text OR patterns if currying
 * @param {Array|Record<string, any>|string} [arg2] - Patterns (Array, Object) or Condition (string)
 * @param {object} [options]
 */
function hev(arg1, arg2, options = {}) {
  // 1. Currying: hev(['bug', 'feature']) or hev({ bug: fn, _: fn })
  if (arguments.length === 1 && (Array.isArray(arg1) || (arg1 && typeof arg1 === 'object'))) {
    return (text, opts) => hev(text, arg1, { ...options, ...opts });
  }

  // If called as hev(text, patterns, options)
  const input = String(arg1 || '');
  const patterns = arg2;

  // 2. Pattern Matching: hev(text, { bug: fn, feature: fn, _: fallback })
  if (patterns && typeof patterns === 'object' && !Array.isArray(patterns)) {
    return matchObject(input, patterns, options);
  }

  // 3. Zero-shot Classification: hev(text, ['bug', 'feature'])
  if (Array.isArray(patterns)) {
    let engine = options.engine || defaultEngine;
    if (typeof engine === 'string') {
      engine = new Engine({ type: engine, ...options });
    }
    return engine.classify(input, patterns, options).then(result => {
      if (options.meta) {
        return result;
      }
      return result.label;
    });
  }

  // 4. Boolean Predicate (Noul): hev(text, 'is spam')
  if (typeof patterns === 'string') {
    let engine = options.engine || defaultEngine;
    if (typeof engine === 'string') {
      engine = new Engine({ type: engine, ...options });
    }
    return engine.predicate(input, patterns, options).then(result => {
      if (options.meta) {
        return result;
      }
      return result.value;
    });
  }

  throw new TypeError('hev: invalid arguments. Expected (text, patterns) or (patterns)(text)');
}

/**
 * Get full classification metadata ({ label, score, probs, engine })
 */
hev.detailed = async function detailed(input, labels, options = {}) {
  let engine = options.engine || defaultEngine;
  if (typeof engine === 'string') {
    engine = new Engine({ type: engine, ...options });
  }
  return engine.classify(String(input || ''), labels, options);
};

/**
 * Get raw probability score [0.0 - 1.0] for a condition or label
 */
hev.score = async function score(input, condition, options = {}) {
  let engine = options.engine || defaultEngine;
  if (typeof engine === 'string') {
    engine = new Engine({ type: engine, ...options });
  }
  if (Array.isArray(condition)) {
    const res = await engine.classify(String(input || ''), condition);
    return res.score;
  }
  const res = await engine.predicate(String(input || ''), condition);
  return res.score;
};

/**
 * Curried boolean predicate helper
 */
hev.is = function is(condition, options = {}) {
  return (text, opts) => hev(text, condition, { ...options, ...opts });
};

/**
 * Curried choice classifier helper
 */
hev.pick = function pick(labels, options = {}) {
  return (text, opts) => hev(text, labels, { ...options, ...opts });
};

/**
 * Curried pattern matcher helper
 */
hev.match = function match(branches, options = {}) {
  return (text, opts) => hev(text, branches, { ...options, ...opts });
};

/**
 * Current active compute device ('cuda' | 'mps' | 'cpu' | 'webgpu' | 'wasm')
 */
hev.device = function device() {
  return defaultEngine.activeDevice;
};

hev.Engine = Engine;
hev.detectDevice = detectDevice;
hev.webmlEngine = webmlEngine;
const jevish = hev;
export default jevish;
export { jevish, hev, matchObject, Engine, detectDevice, webmlEngine };

