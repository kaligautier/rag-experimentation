const assert = require('node:assert/strict');
const fs = require('node:fs');

const report = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const rows = report.results.results;
assert.equal(rows.length, 3);
const positive = rows.find(row => row.vars.question === 'positive');
const negative = rows.find(row => row.vars.question === 'negative');
const missing = rows.find(row => row.vars.question === 'missing-trace');
assert.equal(positive.success, true);
assert.equal(negative.success, false);
assert.equal(missing.success, false);
assert.match(missing.error, /missing completed search/);
for (const row of [positive, negative]) {
  const grades = row.gradingResult.componentResults;
  assert.equal(grades.length, 4);
  for (const grade of grades) {
    const expected = row === negative && grade.assertion.metric === 'groundedness' ? 0 : 1;
    assert.equal(grade.score, expected, grade.assertion.metric);
    assert.equal(grade.pass, Boolean(expected), grade.assertion.metric);
  }
}
console.log('Verified four separate scores, field isolation, a failing axis and a provider error.');
