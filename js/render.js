/* ============================================================
   AI Academy - rendering engine
     MD.render(markdown)  -> HTML string
     MD.inline(text)      -> inline HTML
     MD.plain(markdown)   -> plain text (for search)

   Authoring notes (content files are JS template literals, so we
   deliberately support tilde syntax as well as backtick syntax):
     ~~~python  ... ~~~     fenced code block
     ~like this~            inline code
     :::tip Title ... :::   callout  (tip | warn | danger | math | note)
   No dependencies. Runs straight from file:// with no server.
   ============================================================ */
(function (global) {
  'use strict';

  var MARK = String.fromCharCode(1);   // private placeholder, never in content
  var TICK = String.fromCharCode(96);  // backtick, written this way on purpose

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* =========================================================
     Python syntax highlighting
     ========================================================= */
  var PY_KW = {};
  ('False None True and as assert async await break class continue def del elif else ' +
   'except finally for from global if import in is lambda nonlocal not or pass raise ' +
   'return try while with yield match case').split(' ').forEach(function (k) { PY_KW[k] = 1; });

  var PY_BI = {};
  ('abs all any bool bytes callable chr complex dict dir divmod enumerate eval filter float ' +
   'format frozenset getattr hasattr hash help id input int isinstance issubclass iter len ' +
   'list map max min next object open ord pow print property range repr reversed round set ' +
   'setattr slice sorted staticmethod classmethod str sum super tuple type vars zip ' +
   'Exception ValueError TypeError KeyError IndexError RuntimeError AttributeError ' +
   'NotImplementedError StopIteration ZeroDivisionError FileNotFoundError ' +
   'ImportError AssertionError OverflowError').split(' ').forEach(function (k) { PY_BI[k] = 1; });

  var PY_RE = new RegExp(
    '(#[^\\n]*)' +                                                              // 1 comment
    '|([rbfuRBFU]{0,2}"""[\\s\\S]*?"""|[rbfuRBFU]{0,2}' + "'''[\\s\\S]*?''')" + // 2 triple string
    '|([rbfuRBFU]{0,2}"(?:\\\\.|[^"\\\\\\n])*"' +
      "|[rbfuRBFU]{0,2}'(?:\\\\.|[^'\\\\\\n])*')" +                             // 3 string
    '|(@[A-Za-z_][A-Za-z0-9_.]*)' +                                             // 4 decorator
    '|(\\b\\d[\\d_]*\\.?\\d*(?:[eE][-+]?\\d+)?j?\\b|\\b0[xXbBoO][0-9a-fA-F_]+\\b)' + // 5 number
    '|(\\b[A-Za-z_][A-Za-z0-9_]*\\b)' +                                         // 6 word
    '|([+\\-*/%=<>!&|^~:]+)',                                                   // 7 operator
    'g');

  function pyHighlight(src) {
    var out = '', last = 0, m;
    PY_RE.lastIndex = 0;
    while ((m = PY_RE.exec(src)) !== null) {
      if (m[0] === '') { PY_RE.lastIndex++; continue; }
      out += esc(src.slice(last, m.index));
      last = PY_RE.lastIndex;
      if (m[1])      out += '<span class="t-cm">' + esc(m[1]) + '</span>';
      else if (m[2]) out += '<span class="t-str">' + esc(m[2]) + '</span>';
      else if (m[3]) out += '<span class="t-str">' + esc(m[3]) + '</span>';
      else if (m[4]) out += '<span class="t-dec">' + esc(m[4]) + '</span>';
      else if (m[5]) out += '<span class="t-num">' + esc(m[5]) + '</span>';
      else if (m[6]) {
        var w = m[6];
        var before = src.slice(0, m.index);
        var after = src.slice(PY_RE.lastIndex);
        if (PY_KW[w])                         out += '<span class="t-kw">' + w + '</span>';
        else if (w === 'self' || w === 'cls') out += '<span class="t-self">' + w + '</span>';
        else if (/\bdef\s+$/.test(before))    out += '<span class="t-fn">' + w + '</span>';
        else if (/\bclass\s+$/.test(before))  out += '<span class="t-cls">' + w + '</span>';
        else if (PY_BI[w])                    out += '<span class="t-bi">' + w + '</span>';
        else if (/^\(/.test(after))           out += '<span class="t-fn">' + w + '</span>';
        else                                  out += esc(w);
      }
      else if (m[7]) out += '<span class="t-op">' + esc(m[7]) + '</span>';
    }
    out += esc(src.slice(last));
    return out;
  }

  function shHighlight(src) {
    return esc(src)
      .replace(/^(#.*)$/gm, '<span class="t-cm">$1</span>')
      .replace(/^(\s*)(pip|python|conda|docker|docker-compose|git|uvicorn|curl|export|cd|mkdir|jupyter|kaggle|wget|apt|npm|pytest|mlflow)\b/gm,
               '$1<span class="t-kw">$2</span>');
  }

  function highlight(src, lang) {
    if (!lang || lang === 'python' || lang === 'py') return pyHighlight(src);
    if (lang === 'bash' || lang === 'sh' || lang === 'shell') return shHighlight(src);
    return esc(src);
  }

  /* =========================================================
     Inline markdown
     ========================================================= */
  var CODE_RE_TILDE = /~([^~\n]+)~/g;
  var CODE_RE_TICK  = new RegExp(TICK + '([^' + TICK + ']+)' + TICK, 'g');
  var MARK_RE       = new RegExp(MARK + '(\\d+)' + MARK, 'g');

  function inline(s) {
    var codes = [];
    function stash(_, c) { codes.push(c); return MARK + (codes.length - 1) + MARK; }

    s = String(s).replace(CODE_RE_TICK, stash).replace(CODE_RE_TILDE, stash);
    s = esc(s);
    s = s.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(MARK_RE, function (_, i) { return '<code>' + esc(codes[+i]) + '</code>'; });
    return s;
  }

  /* =========================================================
     Block markdown
     ========================================================= */
  var CALLOUT_ICON = { tip: 'OK', warn: '!', danger: 'X', math: 'fx', note: 'i' };
  var FENCE = new RegExp('^(' + TICK + TICK + TICK + '|~~~)');

  function render(md) {
    if (!md) return '';
    var lines = String(md).replace(/\r\n/g, '\n').split('\n');
    var html = '', i = 0;

    while (i < lines.length) {
      var line = lines[i];

      /* ---- fenced code ---- */
      if (FENCE.test(line)) {
        var head = line.slice(3).trim().split(/\s+/);
        var lang = (head.shift() || 'python').toLowerCase();
        var label = head.join(' ');
        var buf = [];
        i++;
        while (i < lines.length && !FENCE.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        html += '<div class="codewrap"><div class="codebar">' +
                '<span class="lang">' + esc(lang) + '</span>' +
                (label ? '<span class="fn">' + esc(label) + '</span>' : '') +
                '<button class="copy-btn" type="button">copy</button></div>' +
                '<pre class="code"><code>' + highlight(buf.join('\n'), lang) + '</code></pre></div>';
        continue;
      }

      /* ---- callout ---- */
      if (/^:::/.test(line)) {
        var mm = line.match(/^:::\s*([A-Za-z]+)?\s*(.*)$/);
        var kind = (mm && mm[1] ? mm[1] : 'note').toLowerCase();
        if (!CALLOUT_ICON[kind]) kind = 'note';
        var title = (mm && mm[2]) ? mm[2] : kind.toUpperCase();
        var cbuf = [];
        i++;
        while (i < lines.length && !/^:::\s*$/.test(lines[i])) { cbuf.push(lines[i]); i++; }
        i++;
        html += '<div class="callout ' + kind + '">' +
                '<span class="ci">' + esc(CALLOUT_ICON[kind]) + '</span>' +
                '<div class="cb"><span class="ct">' + esc(title) + '</span>' +
                render(cbuf.join('\n')) + '</div></div>';
        continue;
      }

      /* ---- table ---- */
      if (/^\|/.test(line) && i + 1 < lines.length && /^\|[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
        var cells = function (row) {
          return row.replace(/^\||\|$/g, '').split('|').map(function (c) { return c.trim(); });
        };
        var headRow = cells(line);
        i += 2;
        var body = '';
        while (i < lines.length && /^\|/.test(lines[i])) {
          body += '<tr>' + cells(lines[i]).map(function (c) { return '<td>' + inline(c) + '</td>'; }).join('') + '</tr>';
          i++;
        }
        html += '<table><thead><tr>' +
                headRow.map(function (c) { return '<th>' + inline(c) + '</th>'; }).join('') +
                '</tr></thead><tbody>' + body + '</tbody></table>';
        continue;
      }

      /* ---- headings ---- */
      var h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        var lvl = Math.min(h[1].length + 1, 4);
        html += '<h' + lvl + '>' + inline(h[2]) + '</h' + lvl + '>';
        i++; continue;
      }

      /* ---- horizontal rule ---- */
      if (/^(-{3,}|\*{3,})\s*$/.test(line)) { html += '<hr>'; i++; continue; }

      /* ---- blockquote ---- */
      if (/^>\s?/.test(line)) {
        var qbuf = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) { qbuf.push(lines[i].replace(/^>\s?/, '')); i++; }
        html += '<blockquote>' + render(qbuf.join('\n')) + '</blockquote>';
        continue;
      }

      /* ---- unordered list ---- */
      if (/^\s*[-*]\s+/.test(line)) {
        var ui = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
          ui.push(lines[i].replace(/^\s*[-*]\s+/, '')); i++;
        }
        html += '<ul>' + ui.map(function (t) { return '<li>' + inline(t) + '</li>'; }).join('') + '</ul>';
        continue;
      }

      /* ---- ordered list ---- */
      if (/^\s*\d+[.)]\s+/.test(line)) {
        var oi = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          oi.push(lines[i].replace(/^\s*\d+[.)]\s+/, '')); i++;
        }
        html += '<ol>' + oi.map(function (t) { return '<li>' + inline(t) + '</li>'; }).join('') + '</ol>';
        continue;
      }

      /* ---- blank ---- */
      if (!line.trim()) { i++; continue; }

      /* ---- paragraph ---- */
      var pbuf = [];
      while (i < lines.length && lines[i].trim() &&
             !FENCE.test(lines[i]) &&
             !/^(:::|>|#{1,4}\s|\s*[-*]\s|\s*\d+[.)]\s|\|)/.test(lines[i]) &&
             !/^(-{3,}|\*{3,})\s*$/.test(lines[i])) {
        pbuf.push(lines[i]); i++;
      }
      if (pbuf.length) html += '<p>' + inline(pbuf.join(' ')) + '</p>';
      else i++;
    }
    return html;
  }

  function plain(md) {
    var t3 = TICK + TICK + TICK;
    return String(md || '')
      .split(t3).filter(function (_, k) { return k % 2 === 0; }).join(' ')
      .split('~~~').filter(function (_, k) { return k % 2 === 0; }).join(' ')
      .replace(/:::[A-Za-z]*/g, ' ')
      .replace(/[#>*|_]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  global.MD = { render: render, inline: inline, plain: plain, esc: esc };
  global.Highlight = { py: pyHighlight, sh: shHighlight };
})(window);
