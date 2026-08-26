/* Renders every piece of markdown in the curriculum through the real
   renderer and reports anything suspicious. Run: node tools/render-test.js */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const sandbox = { window: {}, console };
sandbox.global = sandbox;
vm.createContext(sandbox);

// load the renderer
vm.runInContext(fs.readFileSync(path.join(ROOT, 'js/render.js'), 'utf8'), sandbox,
                { filename: 'render.js' });

// load all curriculum files
const dir = path.join(ROOT, 'js/curriculum');
const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.js')).sort() : [];
for (const f of files) {
  vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), sandbox, { filename: f });
}

const MD = sandbox.window.MD;
const TRACKS = sandbox.window.CURRICULUM || [];
const MARK = String.fromCharCode(1);

let problems = 0, blocks = 0, chars = 0, codeBlocks = 0;

function checkPiece(where, md) {
  if (!md) return;
  blocks++;
  chars += md.length;
  let html;
  try {
    html = MD.render(md);
  } catch (e) {
    console.log('  RENDER THREW  ' + where + ' :: ' + e.message);
    problems++;
    return;
  }
  codeBlocks += (html.match(/<pre class="code">/g) || []).length;

  if (html.indexOf(MARK) !== -1) {
    console.log('  leftover placeholder  ' + where); problems++;
  }
  if (/~~~/.test(html)) {
    console.log('  unrendered fence      ' + where); problems++;
  }
  if (/&lt;span class=/.test(html)) {
    console.log('  double-escaped span   ' + where); problems++;
  }
  // every opened codewrap must be closed
  const open = (html.match(/<div class="codewrap">/g) || []).length;
  const pre = (html.match(/<\/pre><\/div>/g) || []).length;
  if (open !== pre) {
    console.log('  unbalanced code block ' + where + ' (' + open + ' open, ' + pre + ' closed)');
    problems++;
  }
}

for (const tr of TRACKS) {
  checkPiece(tr.id + ' [track intro]', tr.intro);
  for (const tp of tr.topics || []) {
    checkPiece(tr.id + '/' + tp.id + ' [intro]', tp.intro);
    (tp.levels || []).forEach((lv, i) => {
      checkPiece(tr.id + '/' + tp.id + ' [level ' + (i + 1) + ' ' + lv.name + ']', lv.md);
    });
  }
}

console.log('='.repeat(70));
console.log('rendered ' + blocks + ' markdown blocks, ' +
            (chars / 1024).toFixed(0) + ' KB of content, ' +
            codeBlocks + ' code blocks');
console.log(problems === 0 ? 'RENDER OK - no problems' : problems + ' PROBLEM(S)');
process.exit(problems === 0 ? 0 : 1);
