/* ============================================================
   AI Academy - application
   Routing, navigation, progress, search, topic rendering.
   ============================================================ */
(function () {
  'use strict';

  var TRACKS = window.CURRICULUM || [];
  var LS_PROG = 'aiacademy.progress.v1';
  var LS_THEME = 'aiacademy.theme';
  var LS_LAST = 'aiacademy.last';

  /* ---------- state ---------- */
  var progress = load(LS_PROG, {});
  var levelFilter = 'All';

  function load(k, dflt) {
    try { return JSON.parse(localStorage.getItem(k)) || dflt; } catch (e) { return dflt; }
  }
  function save(k, v) {
    try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {}
  }

  /* ---------- flat lesson list (for prev/next) ---------- */
  var FLAT = [];
  TRACKS.forEach(function (tr) {
    (tr.topics || []).forEach(function (tp) { FLAT.push({ track: tr, topic: tp }); });
  });

  function key(trackId, topicId) { return trackId + '/' + topicId; }
  function isDone(trackId, topicId) { return !!progress[key(trackId, topicId)]; }

  function trackStats(tr) {
    var total = (tr.topics || []).length, done = 0;
    (tr.topics || []).forEach(function (tp) { if (isDone(tr.id, tp.id)) done++; });
    return { total: total, done: done, pct: total ? Math.round(done / total * 100) : 0 };
  }
  function globalStats() {
    var total = FLAT.length, done = 0;
    FLAT.forEach(function (f) { if (isDone(f.track.id, f.topic.id)) done++; });
    return { total: total, done: done, pct: total ? Math.round(done / total * 100) : 0 };
  }

  /* ---------- dom helpers ---------- */
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var view = $('#view');
  var esc = window.MD.esc;

  function go(hash) { location.hash = hash; }

  /* ============================================================
     SIDEBAR
     ============================================================ */
  function buildNav() {
    var nav = $('#trackNav');
    nav.innerHTML = TRACKS.map(function (tr, ti) {
      var st = trackStats(tr);
      var topics = (tr.topics || []).map(function (tp) {
        return '<div class="nav-topic' + (isDone(tr.id, tp.id) ? ' done' : '') + '" ' +
               'data-go="#/l/' + tr.id + '/' + tp.id + '" data-k="' + key(tr.id, tp.id) + '">' +
               '<span class="dot">●</span><span>' + esc(tp.title) + '</span></div>';
      }).join('');
      return '<div class="nav-track" data-track="' + tr.id + '">' +
             '<div class="nav-track-head" data-toggle="' + tr.id + '">' +
             '<span class="car">▶</span>' +
             '<span class="ico">' + tr.icon + '</span>' +
             '<span class="nm">' + esc(tr.title) + '</span>' +
             '<span class="cnt">' + st.done + '/' + st.total + '</span>' +
             '</div><div class="nav-topics">' + topics + '</div></div>';
    }).join('');
  }

  function syncNav(activeKey) {
    document.querySelectorAll('.nav-topic').forEach(function (el) {
      var k = el.dataset.k;
      el.classList.toggle('active', k === activeKey);
      el.classList.toggle('done', !!progress[k]);
    });
    document.querySelectorAll('.nav-track').forEach(function (el) {
      var tr = TRACKS.find(function (t) { return t.id === el.dataset.track; });
      if (!tr) return;
      var st = trackStats(tr);
      var cnt = el.querySelector('.cnt');
      if (cnt) cnt.textContent = st.done + '/' + st.total;
    });
    var g = globalStats();
    $('#pgBar').style.width = g.pct + '%';
    $('#pgPct').textContent = g.pct + '%';
    $('#pgCount').textContent = g.done + ' / ' + g.total + ' lessons complete';
  }

  function openNavTrack(trackId) {
    document.querySelectorAll('.nav-track').forEach(function (el) {
      if (el.dataset.track === trackId) el.classList.add('open');
    });
  }

  /* ============================================================
     HOME
     ============================================================ */
  function renderHome() {
    var g = globalStats();
    var levels = ['All', 'Beginner', 'Intermediate', 'Advanced'];
    var totalLevels = 0;
    FLAT.forEach(function (f) { totalLevels += (f.topic.levels || []).length; });

    var last = load(LS_LAST, null);
    var resume = '';
    if (last && last.trackId) {
      var lt = TRACKS.find(function (t) { return t.id === last.trackId; });
      var lp = lt && (lt.topics || []).find(function (t) { return t.id === last.topicId; });
      if (lp) resume = '<button class="btn primary" data-go="#/l/' + lt.id + '/' + lp.id + '">' +
                       '▶ Resume: ' + esc(lp.title) + '</button>';
    }
    if (!resume && FLAT.length) {
      resume = '<button class="btn primary" data-go="#/l/' + FLAT[0].track.id + '/' + FLAT[0].topic.id + '">' +
               '▶ Start from the beginning</button>';
    }

    var cards = TRACKS.filter(function (tr) {
      return levelFilter === 'All' || tr.level === levelFilter;
    }).map(function (tr) {
      var st = trackStats(tr);
      return '<div class="card" data-go="#/t/' + tr.id + '">' +
        '<div class="c-top"><span class="c-ico">' + tr.icon + '</span>' +
        '<span class="c-lvl ' + tr.level.toLowerCase() + '">' + tr.level + '</span></div>' +
        '<h3>' + esc(tr.title) + '</h3>' +
        '<p>' + esc(tr.blurb) + '</p>' +
        '<div class="c-foot"><span>' + st.total + ' lessons</span>' +
        '<span class="bar"><i style="width:' + st.pct + '%"></i></span>' +
        '<span>' + st.pct + '%</span></div></div>';
    }).join('');

    view.innerHTML =
      '<section class="hero">' +
        '<h1>Learn Data Science, Machine Learning &amp; AI<br>from zero to genuinely advanced.</h1>' +
        '<p>Every topic starts with a plain-English explanation of what it is and how it actually works, ' +
        'then walks you up through progressively harder Python implementations - from the simplest ' +
        'possible version to production-grade code with real data engineering.</p>' +
        '<div class="hero-stats">' +
          '<div class="hstat"><b>' + TRACKS.length + '</b><span>Tracks</span></div>' +
          '<div class="hstat"><b>' + FLAT.length + '</b><span>Lessons</span></div>' +
          '<div class="hstat"><b>' + totalLevels + '</b><span>Guided levels</span></div>' +
          '<div class="hstat"><b>' + g.pct + '%</b><span>Complete</span></div>' +
        '</div>' +
        '<div class="hero-cta">' + resume +
          '<button class="btn sec" data-go="#/t/projects">See the learning paths</button>' +
        '</div>' +
      '</section>' +

      '<div class="sec-head"><h2>Choose what you want to learn</h2>' +
      '<span>Tracks are ordered, but you can start anywhere.</span></div>' +

      '<div class="filters">' + levels.map(function (l) {
        return '<span class="chip' + (levelFilter === l ? ' on' : '') + '" data-filter="' + l + '">' + l + '</span>';
      }).join('') + '</div>' +

      '<div class="grid">' + cards + '</div>';

    $('#crumbs').innerHTML = '<b>Home</b>';
    syncNav(null);
  }

  /* ============================================================
     TRACK PAGE
     ============================================================ */
  function renderTrack(trackId) {
    var tr = TRACKS.find(function (t) { return t.id === trackId; });
    if (!tr) return renderHome();
    var st = trackStats(tr);

    var rows = (tr.topics || []).map(function (tp, i) {
      var n = (tp.levels || []).length;
      return '<div class="topic-row' + (isDone(tr.id, tp.id) ? ' done' : '') + '" ' +
             'data-go="#/l/' + tr.id + '/' + tp.id + '">' +
             '<div class="num">' + (isDone(tr.id, tp.id) ? '✓' : (i + 1)) + '</div>' +
             '<div class="tx"><b>' + esc(tp.title) + '</b><span>' + esc(tp.summary || '') + '</span></div>' +
             '<div class="meta">' + n + ' level' + (n === 1 ? '' : 's') + '</div></div>';
    }).join('');

    view.innerHTML =
      '<div class="track-hero"><div class="t-ico">' + tr.icon + '</div><div>' +
      '<h1>' + esc(tr.title) + '</h1><p>' + esc(tr.blurb) + '</p></div></div>' +
      '<div class="c-foot" style="margin-top:18px;display:flex;gap:12px;align-items:center;font-size:12px;color:var(--txt-3)">' +
      '<span>' + st.done + ' / ' + st.total + ' complete</span>' +
      '<span class="bar" style="max-width:260px;flex:1"><i style="width:' + st.pct + '%"></i></span></div>' +
      (tr.intro ? '<div class="section prose">' + window.MD.render(tr.intro) + '</div>' : '') +
      '<div class="topic-list">' + rows + '</div>';

    $('#crumbs').innerHTML = '<a data-go="#/">Home</a> &nbsp;/&nbsp; <b>' + esc(tr.title) + '</b>';
    openNavTrack(trackId);
    syncNav(null);
  }

  /* ============================================================
     TOPIC PAGE
     ============================================================ */
  var curLevel = 0;

  function renderTopic(trackId, topicId, levelIdx) {
    var tr = TRACKS.find(function (t) { return t.id === trackId; });
    if (!tr) return renderHome();
    var idx = (tr.topics || []).findIndex(function (t) { return t.id === topicId; });
    if (idx < 0) return renderTrack(trackId);
    var tp = tr.topics[idx];
    var levels = tp.levels || [];
    curLevel = Math.max(0, Math.min(levelIdx || 0, levels.length - 1));
    save(LS_LAST, { trackId: trackId, topicId: topicId });

    var flatIdx = FLAT.findIndex(function (f) { return f.track.id === trackId && f.topic.id === topicId; });
    var nxt = FLAT[flatIdx + 1];

    var tags = (tp.tags || []).map(function (t) { return '<span class="tag">' + esc(t) + '</span>'; }).join('');

    var lvlNav = levels.length > 1
      ? '<div class="level-nav">' + levels.map(function (lv, i) {
          return '<button class="lvl-btn' + (i === curLevel ? ' on' : '') + '" data-lvl="' + i + '">' +
                 '<span class="n">' + (i + 1) + '</span>' + esc(lv.name) + '</button>';
        }).join('') + '</div>'
      : '';

    var kp = '';
    if ((tp.keyPoints && tp.keyPoints.length) || (tp.pitfalls && tp.pitfalls.length)) {
      kp = '<div class="kp-grid">' +
        (tp.keyPoints && tp.keyPoints.length
          ? '<div class="kp good"><h4>Remember this</h4><ul>' +
            tp.keyPoints.map(function (t) { return '<li>' + window.MD.inline(t) + '</li>'; }).join('') + '</ul></div>' : '') +
        (tp.pitfalls && tp.pitfalls.length
          ? '<div class="kp bad"><h4>Common mistakes</h4><ul>' +
            tp.pitfalls.map(function (t) { return '<li>' + window.MD.inline(t) + '</li>'; }).join('') + '</ul></div>' : '') +
        '</div>';
    }

    var quiz = '';
    if (tp.quiz && tp.quiz.length) {
      quiz = '<div class="section"><h2>Check yourself</h2><div class="quiz">' +
        tp.quiz.map(function (q, qi) {
          return '<div class="qitem" data-q="' + qi + '" data-a="' + q.answer + '">' +
            '<div class="q">' + window.MD.inline(q.q) + '</div>' +
            q.options.map(function (o, oi) {
              return '<div class="qopt" data-o="' + oi + '"><span class="k">' +
                     'ABCDE'[oi] + '</span><span>' + window.MD.inline(o) + '</span></div>';
            }).join('') +
            '<div class="qwhy"><b>Why:</b> ' + window.MD.inline(q.why || '') + '</div></div>';
        }).join('') + '</div></div>';
    }

    var doneNow = isDone(trackId, topicId);

    view.innerHTML =
      '<div class="topic-head">' +
        '<div class="eyebrow">' + tr.icon + ' &nbsp;' + esc(tr.title) + ' &nbsp;·&nbsp; Lesson ' + (idx + 1) + ' of ' + tr.topics.length + '</div>' +
        '<h1>' + esc(tp.title) + '</h1>' +
        '<p class="lede">' + esc(tp.summary || '') + '</p>' +
        (tags ? '<div class="tagrow">' + tags + '</div>' : '') +
      '</div>' +

      (tp.intro ? '<div class="section prose">' + window.MD.render(tp.intro) + '</div>' : '') +
      kp +
      lvlNav +
      '<div id="levelBox"></div>' +
      quiz +

      '<div class="topic-foot">' +
        '<button class="done-btn' + (doneNow ? ' on' : '') + '" id="doneBtn">' +
        (doneNow ? '✓ Completed' : '○ Mark as complete') + '</button>' +
        (nxt ? '<div class="nextlink">Up next<b data-go="#/l/' + nxt.track.id + '/' + nxt.topic.id + '">' +
               esc(nxt.topic.title) + ' →</b></div>' : '') +
      '</div>';

    renderLevel(tp, curLevel);

    $('#crumbs').innerHTML =
      '<a data-go="#/">Home</a> &nbsp;/&nbsp; <a data-go="#/t/' + tr.id + '">' + esc(tr.title) + '</a>' +
      ' &nbsp;/&nbsp; <b>' + esc(tp.title) + '</b>';

    openNavTrack(trackId);
    syncNav(key(trackId, topicId));
    window.scrollTo(0, 0);

    // prev / next buttons
    $('#prevBtn').disabled = flatIdx <= 0;
    $('#nextBtn').disabled = flatIdx >= FLAT.length - 1;
    $('#prevBtn').onclick = function () {
      var p = FLAT[flatIdx - 1]; if (p) go('#/l/' + p.track.id + '/' + p.topic.id);
    };
    $('#nextBtn').onclick = function () {
      var n = FLAT[flatIdx + 1]; if (n) go('#/l/' + n.track.id + '/' + n.topic.id);
    };

    $('#doneBtn').onclick = function () {
      var k = key(trackId, topicId);
      if (progress[k]) delete progress[k]; else progress[k] = true;
      save(LS_PROG, progress);
      this.classList.toggle('on', !!progress[k]);
      this.textContent = progress[k] ? '✓ Completed' : '○ Mark as complete';
      syncNav(k);
    };
  }

  function renderLevel(tp, i) {
    var levels = tp.levels || [];
    var lv = levels[i];
    var box = $('#levelBox');
    if (!box) return;
    if (!lv) { box.innerHTML = ''; return; }
    box.innerHTML =
      (lv.goal ? '<div class="level-goal"><span>◎</span><div><b>Goal of this level:</b> ' +
                 window.MD.inline(lv.goal) + '</div></div>' : '') +
      '<div class="section prose">' + window.MD.render(lv.md || '') + '</div>';
  }

  /* ============================================================
     SEARCH
     ============================================================ */
  function renderSearch(q) {
    var needle = q.toLowerCase().trim();
    var hits = [];
    TRACKS.forEach(function (tr) {
      (tr.topics || []).forEach(function (tp) {
        var hay = [tp.title, tp.summary, (tp.tags || []).join(' '),
                   window.MD.plain(tp.intro),
                   (tp.levels || []).map(function (l) { return l.name + ' ' + window.MD.plain(l.md); }).join(' ')
                  ].join(' ').toLowerCase();
        var pos = hay.indexOf(needle);
        if (pos >= 0) {
          var score = tp.title.toLowerCase().indexOf(needle) >= 0 ? 0 : 1;
          hits.push({ tr: tr, tp: tp, pos: pos, score: score, hay: hay });
        }
      });
    });
    hits.sort(function (a, b) { return a.score - b.score || a.pos - b.pos; });

    view.innerHTML = '<div class="sec-head"><h2>Search</h2><span>' + hits.length +
      ' result' + (hits.length === 1 ? '' : 's') + ' for "' + esc(q) + '"</span></div>' +
      (hits.length ? hits.slice(0, 60).map(function (h) {
        var snip = h.hay.slice(Math.max(0, h.pos - 70), h.pos + 130).replace(/^\S*\s/, '');
        snip = esc(snip).replace(new RegExp('(' + needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
        return '<div class="sr" data-go="#/l/' + h.tr.id + '/' + h.tp.id + '">' +
               '<span class="path">' + h.tr.icon + ' ' + esc(h.tr.title) + '</span>' +
               '<b>' + esc(h.tp.title) + '</b><p>…' + snip + '…</p></div>';
      }).join('') : '<div class="empty">Nothing found. Try “gradient”, “overfitting”, “LSTM”, “scaling”…</div>');

    $('#crumbs').innerHTML = '<a data-go="#/">Home</a> &nbsp;/&nbsp; <b>Search</b>';
    syncNav(null);
  }

  /* ============================================================
     ROUTER
     ============================================================ */
  function route() {
    var h = location.hash.replace(/^#/, '') || '/';
    var parts = h.split('/').filter(Boolean);
    document.getElementById('app').classList.remove('nav-open');
    if (parts[0] === 't' && parts[1]) return renderTrack(parts[1]);
    if (parts[0] === 'l' && parts[1] && parts[2]) return renderTopic(parts[1], parts[2], 0);
    if (parts[0] === 's' && parts[1]) return renderSearch(decodeURIComponent(parts[1]));
    return renderHome();
  }

  /* ============================================================
     GLOBAL EVENTS
     ============================================================ */
  document.addEventListener('click', function (e) {
    var goEl = e.target.closest('[data-go]');
    if (goEl) { go(goEl.dataset.go); return; }

    var nav = e.target.closest('[data-nav="home"]');
    if (nav) { go('#/'); return; }

    var tgl = e.target.closest('[data-toggle]');
    if (tgl) { tgl.parentElement.classList.toggle('open'); return; }

    var chip = e.target.closest('[data-filter]');
    if (chip) { levelFilter = chip.dataset.filter; renderHome(); return; }

    var lvl = e.target.closest('[data-lvl]');
    if (lvl) {
      var i = +lvl.dataset.lvl;
      document.querySelectorAll('.lvl-btn').forEach(function (b) { b.classList.toggle('on', +b.dataset.lvl === i); });
      var p = location.hash.split('/').filter(Boolean);
      var tr = TRACKS.find(function (t) { return t.id === p[1]; });
      var tp = tr && tr.topics.find(function (t) { return t.id === p[2]; });
      if (tp) { curLevel = i; renderLevel(tp, i); lvl.scrollIntoView({ block: 'nearest' }); }
      return;
    }

    var copy = e.target.closest('.copy-btn');
    if (copy) {
      var pre = copy.closest('.codewrap').querySelector('pre.code');
      var txt = pre.innerText;
      var done = function () { copy.textContent = 'copied'; copy.classList.add('ok');
        setTimeout(function () { copy.textContent = 'copy'; copy.classList.remove('ok'); }, 1400); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, fallback);
      else fallback();
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = txt; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (err) {}
        document.body.removeChild(ta);
      }
      return;
    }

    var opt = e.target.closest('.qopt');
    if (opt) {
      var item = opt.closest('.qitem');
      if (item.dataset.answered) return;
      item.dataset.answered = '1';
      var ans = +item.dataset.a;
      item.querySelectorAll('.qopt').forEach(function (o, oi) {
        if (oi === ans) o.classList.add('right');
        else if (o === opt) o.classList.add('wrong');
      });
      item.querySelector('.qwhy').classList.add('show');
      return;
    }

    if (e.target.id === 'scrim') document.getElementById('app').classList.remove('nav-open');
  });

  $('#menuBtn').onclick = function () { document.getElementById('app').classList.toggle('nav-open'); };

  $('#themeBtn').onclick = function () {
    var cur = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = cur;
    save(LS_THEME, cur);
  };

  $('#resetBtn').onclick = function () {
    if (!confirm('Reset all lesson progress?')) return;
    progress = {}; save(LS_PROG, progress);
    buildNav(); route();
  };

  var searchBox = $('#sideSearch');
  var stimer;
  searchBox.addEventListener('input', function () {
    clearTimeout(stimer);
    var v = this.value;
    stimer = setTimeout(function () {
      if (v.trim().length >= 2) go('#/s/' + encodeURIComponent(v.trim()));
      else if (location.hash.indexOf('#/s/') === 0) go('#/');
    }, 260);
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== searchBox) { e.preventDefault(); searchBox.focus(); }
    if (e.key === 'Escape') { searchBox.blur(); document.getElementById('app').classList.remove('nav-open'); }
    if (document.activeElement === searchBox) return;
    if (e.key === 'ArrowRight' && e.altKey) $('#nextBtn').click();
    if (e.key === 'ArrowLeft' && e.altKey) $('#prevBtn').click();
  });

  window.addEventListener('hashchange', route);

  /* ---------- boot ---------- */
  var savedTheme = load(LS_THEME, 'dark');
  document.documentElement.dataset.theme = savedTheme;
  buildNav();
  route();
})();
