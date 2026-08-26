/* Structural checks that app.js's assumptions hold against the real curriculum.
   Run: node tools/app-test.js */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const sandbox = { window: {}, console };
sandbox.global = sandbox;
vm.createContext(sandbox);

vm.runInContext(fs.readFileSync(path.join(ROOT, 'js/render.js'), 'utf8'), sandbox);
const dir = path.join(ROOT, 'js/curriculum');
const files = fs.readdirSync(dir).filter(f => f.endsWith('.js')).sort();
for (const f of files) {
  vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), sandbox, { filename: f });
}

const TRACKS = sandbox.window.CURRICULUM;
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const appjs = fs.readFileSync(path.join(ROOT, 'js/app.js'), 'utf8');

let problems = 0;
const fail = m => { console.log('  FAIL  ' + m); problems++; };
const ok = m => console.log('   ok   ' + m);

// ---- 1. every curriculum file is actually loaded by index.html -------
for (const f of files) {
  if (!html.includes('js/curriculum/' + f)) fail(`${f} is not referenced in index.html`);
}
const referenced = [...html.matchAll(/js\/curriculum\/([\w.-]+\.js)/g)].map(m => m[1]);
for (const r of referenced) {
  if (!files.includes(r)) fail(`index.html references missing file ${r}`);
}
ok(`${files.length} curriculum files, all referenced by index.html`);

// ---- 2. ids are unique and URL-safe ---------------------------------
const trackIds = new Set();
for (const tr of TRACKS) {
  if (trackIds.has(tr.id)) fail(`duplicate track id: ${tr.id}`);
  trackIds.add(tr.id);
  if (!/^[a-z0-9-]+$/.test(tr.id)) fail(`track id not URL-safe: ${tr.id}`);

  const topicIds = new Set();
  for (const tp of tr.topics) {
    if (topicIds.has(tp.id)) fail(`duplicate topic id in ${tr.id}: ${tp.id}`);
    topicIds.add(tp.id);
    if (!/^[a-z0-9-]+$/.test(tp.id)) fail(`topic id not URL-safe: ${tr.id}/${tp.id}`);
  }
}
ok(`${trackIds.size} unique track ids, all topic ids unique and URL-safe`);

// ---- 3. the hard-coded home-page link resolves ----------------------
const hardLinks = [...appjs.matchAll(/data-go="#\/t\/([\w-]+)"/g)].map(m => m[1]);
for (const l of hardLinks) {
  if (!trackIds.has(l)) fail(`app.js links to #/t/${l} but no such track exists`);
}
ok(`hard-coded track links resolve: ${[...new Set(hardLinks)].join(', ') || 'none'}`);

// ---- 4. level filter values match the track levels ------------------
const VALID_LEVELS = ['Beginner', 'Intermediate', 'Advanced'];
for (const tr of TRACKS) {
  if (!VALID_LEVELS.includes(tr.level)) fail(`${tr.id} has level "${tr.level}"`);
}
ok('all track levels are Beginner / Intermediate / Advanced');

// ---- 5. quizzes are answerable --------------------------------------
let quizCount = 0;
for (const tr of TRACKS) for (const tp of tr.topics) {
  for (const [i, q] of (tp.quiz || []).entries()) {
    quizCount++;
    if (!Array.isArray(q.options) || q.options.length < 2)
      fail(`${tr.id}/${tp.id} quiz ${i}: needs 2+ options`);
    if (!Number.isInteger(q.answer) || q.answer < 0 || q.answer >= q.options.length)
      fail(`${tr.id}/${tp.id} quiz ${i}: answer index out of range`);
    if (q.options.length > 5)
      fail(`${tr.id}/${tp.id} quiz ${i}: more than 5 options (labels are A-E)`);
    if (!q.why) fail(`${tr.id}/${tp.id} quiz ${i}: missing explanation`);
  }
}
ok(`${quizCount} quiz questions, all answerable with an explanation`);

// ---- 6. every topic renders to non-trivial HTML ---------------------
const MD = sandbox.window.MD;
let thinnest = { chars: Infinity, id: '' };
for (const tr of TRACKS) for (const tp of tr.topics) {
  const total = MD.render(tp.intro || '').length +
    (tp.levels || []).reduce((a, l) => a + MD.render(l.md || '').length, 0);
  if (total < 2000) fail(`${tr.id}/${tp.id} renders only ${total} chars`);
  if (total < thinnest.chars) thinnest = { chars: total, id: `${tr.id}/${tp.id}` };
  if (!tp.summary || tp.summary.length < 30)
    fail(`${tr.id}/${tp.id} summary too short`);
}
ok(`every topic renders substantial HTML (thinnest: ${thinnest.id}, ${thinnest.chars} chars)`);

// ---- 7. search index works ------------------------------------------
function search(needle) {
  needle = needle.toLowerCase();
  let hits = 0;
  for (const tr of TRACKS) for (const tp of tr.topics) {
    const hay = [tp.title, tp.summary, (tp.tags || []).join(' '), MD.plain(tp.intro),
      (tp.levels || []).map(l => l.name + ' ' + MD.plain(l.md)).join(' ')]
      .join(' ').toLowerCase();
    if (hay.includes(needle)) hits++;
  }
  return hits;
}
const probes = ['gradient descent', 'overfitting', 'lstm', 'shap', 'leakage',
                'attention', 'diffusion', 'a*', 'docker', 'psi'];
const misses = probes.filter(p => search(p) === 0);
if (misses.length) fail(`search finds nothing for: ${misses.join(', ')}`);
else ok(`search returns hits for all ${probes.length} probe terms`);

// ---- 8. prev/next chain covers every lesson --------------------------
const flat = [];
for (const tr of TRACKS) for (const tp of tr.topics) flat.push(`${tr.id}/${tp.id}`);
if (new Set(flat).size !== flat.length) fail('duplicate lesson keys in the flat list');
ok(`prev/next chain covers ${flat.length} lessons with unique keys`);

// ---- summary ---------------------------------------------------------
const levels = TRACKS.reduce((a, t) => a + t.topics.reduce((b, p) => b + p.levels.length, 0), 0);
console.log('='.repeat(70));
console.log(`${TRACKS.length} tracks | ${flat.length} topics | ${levels} levels | ${quizCount} quiz questions`);
console.log(problems === 0 ? 'APP STRUCTURE OK' : `${problems} PROBLEM(S)`);
process.exit(problems === 0 ? 0 : 1);
