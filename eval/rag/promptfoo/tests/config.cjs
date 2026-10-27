const path = require('node:path');
const { buildConfig } = require('../config.cjs');

module.exports = buildConfig({
  description: 'OFFLINE scaffold contract test — synthetic scores, not RAG quality',
  provider: `file://${path.join(__dirname, 'provider.py')}`,
  judge: `file://${path.join(__dirname, 'judge.cjs')}`,
  tests: ['positive', 'negative', 'missing-trace'].map(question => ({
    vars: { question, reference: 'REFERENCE_ONLY' },
  })),
});
