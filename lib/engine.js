import { classifyZeroDep, judgePredicateZeroDep } from './zero-dep.js';

let glinerModule = null;
let glinerLoadAttempted = false;

/**
 * Probe optional hardware / neural model package dynamically
 */
async function loadGliner() {
  if (glinerLoadAttempted) return glinerModule;
  glinerLoadAttempted = true;
  try {
    // Dynamic import for optional Fastino / GLiNER2.5 package
    glinerModule = await import('@fastino/gliner25-long-context');
  } catch {
    glinerModule = null;
  }
  return glinerModule;
}

/**
 * Unified execution engine supporting 3 tiers:
 * 1. Fastino / GLiNER2.5 (local neural, if installed)
 * 2. TypeSafe System One (cloud, if TYPESAFE_API_KEY present)
 * 3. Zero-dep in-tree engine (default, always available, 0ms, 0 external deps)
 */
export class Engine {
  constructor(options = {}) {
    this.apiKey = options.apiKey || (typeof process !== 'undefined' ? process.env?.TYPESAFE_API_KEY : undefined);
    this.endpoint = options.endpoint || 'https://api.typesafe.ai/v1/systemone';
  }

  async classify(input, labels) {
    // Tier 1: Local Neural GLiNER2.5 if installed
    const gliner = await loadGliner();
    if (gliner && typeof gliner.classify === 'function') {
      try {
        const result = await gliner.classify(input, labels);
        return { ...result, engine: 'gliner' };
      } catch {
        // Fall through on error
      }
    }

    // Tier 2: Cloud System One if API key is provided
    if (this.apiKey) {
      try {
        const res = await fetch(this.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({
            model: 'jev-latest',
            questions: [{
              id: 'choice',
              type: 'choice',
              instructions: 'Select the label that best fits the input text.',
              criteria: labels.reduce((acc, l) => ({ ...acc, [l]: l }), {}),
            }],
            state: { text: input },
          }),
        });

        if (res.ok) {
          const data = await res.json();
          const q = data.answers?.choice;
          if (q) {
            return {
              label: q.choice,
              score: q.confidence ?? 1.0,
              probs: q.probabilities ?? {},
              engine: 'typesafe',
            };
          }
        }
      } catch {
        // Fall through to in-tree on network/API failure
      }
    }

    // Tier 3: Zero-dependency built-in engine (always reliable, 0ms, offline)
    const result = classifyZeroDep(input, labels);
    return { ...result, engine: 'builtin' };
  }

  async predicate(input, condition) {
    if (this.apiKey) {
      try {
        const res = await fetch(this.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({
            model: 'jev-latest',
            questions: [{
              id: 'noul',
              type: 'noul',
              instructions: `Is the statement true about the text: ${condition}`,
            }],
            state: { text: input },
          }),
        });

        if (res.ok) {
          const data = await res.json();
          const q = data.answers?.noul;
          if (q) {
            return {
              value: q.noul >= 0.5,
              score: q.noul,
              engine: 'typesafe',
            };
          }
        }
      } catch {
        // Fall through
      }
    }

    const result = judgePredicateZeroDep(input, condition);
    return { ...result, engine: 'builtin' };
  }
}

export const defaultEngine = new Engine();
