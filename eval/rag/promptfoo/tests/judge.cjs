// A deterministic contract stub, not a substitute for a semantic LLM judge.
const assert = require('node:assert/strict');

module.exports = class FixtureJudge {
  id() { return 'offline-contract-judge'; }

  async callApi(prompt) {
    const metric = /^You evaluate (\w+)\./.exec(prompt)?.[1];
    const data = JSON.parse(prompt.split('\nDATA\n')[1].split('\nEND_DATA\n')[0]);
    const expected = {
      correctness: ['answer', 'question', 'reference'],
      relevance: ['answer', 'question'],
      groundedness: ['answer', 'documents'],
      retrieval_relevance: ['documents', 'question'],
    };
    assert.deepEqual(Object.keys(data).sort(), expected[metric]);
    if (metric !== 'correctness') assert(!prompt.includes('REFERENCE_ONLY'));
    if (metric === 'retrieval_relevance') assert(!prompt.includes('ANSWER_ONLY'));
    const pass = !(metric === 'groundedness' && data.answer.includes('negative'));
    return { output: JSON.stringify({ pass, score: Number(pass), reason: 'Synthetic contract verdict' }) };
  }
};
