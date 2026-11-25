(function () {
  'use strict';

  var MAX_CHARS = 2000;
  var COUNTER_FROM = 1700;
  var LETTERS = ['R', 'I', 'A', 'S', 'E', 'C'];

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    transcript: $('transcript'), results: $('results'), msg: $('msg'), form: $('composer'),
    send: $('sendBtn'), stop: $('stopBtn'), counter: $('counter'), error: $('errorBox'),
    chip: $('codeChip'), turnText: $('turnText'), barFill: $('barFill'), progress: $('progress'),
    layout: $('layout'), chatScroll: $('chatScroll'),
    rProfile: $('rProfile'), rPolicy: $('rPolicy'),
    mapCard: $('mapCard'), scoresCard: $('scoresCard'), rankCard: $('rankCard'), mapNotesBody: $('mapNotesBody'),
    feed: $('feed'), feedCount: $('feedCount'), compsList: $('compsList'), rawCard: $('rawCard'), rawPre: $('rawPre')
  };

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Restart a one-shot CSS animation class on an element.
  function replay(node, cls) {
    node.classList.remove(cls);
    void node.offsetWidth;
    node.classList.add(cls);
  }

  var state = {
    sessionId: null, code: null, maxTurns: 10, turn: 0,
    busy: false, finished: false, typingEl: null, lastText: null, lastFailedText: null, messageId: null
  };

  /* ---------- DOM helpers (textContent only; never innerHTML) ---------- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function scrollToEnd() {
    var behavior = reduceMotion ? 'auto' : 'smooth';
    var box = els.chatScroll;
    // Wide layouts scroll the chat column itself; narrow layouts scroll the page.
    if (box && box.scrollHeight > box.clientHeight + 1 && getComputedStyle(box).overflowY !== 'visible') {
      box.scrollTo({ top: box.scrollHeight, behavior: behavior });
      return;
    }
    var last = els.results.lastElementChild || els.transcript.lastElementChild;
    if (last && last.scrollIntoView) last.scrollIntoView({ block: 'end', behavior: behavior });
  }

  /* ---------- transcript ---------- */
  function addUser(text) {
    var m = el('div', 'msg user');
    m.appendChild(el('span', 'who', 'You'));
    m.appendChild(el('p', 'bubble', text));
    els.transcript.appendChild(m);
    scrollToEnd();
  }

  function quoteFigure(caption, text) {
    var f = el('figure');
    f.appendChild(el('figcaption', null, caption));
    f.appendChild(el('blockquote', null, '“' + text + '”'));
    return f;
  }

  function addBot(data) {
    var q = data.quotes;
    var m;
    if (q && q.now && data.action === 'CHALLENGE' && q.earlier) {
      m = el('div', 'msg card');
      m.appendChild(el('span', 'who', 'Matcher · checking something you said'));
      var box = el('div', 'card-box');
      var grid = el('div', 'quotes');
      grid.appendChild(quoteFigure('Earlier, turn ' + q.earlier_turn, q.earlier));
      grid.appendChild(quoteFigure('Just now', q.now));
      box.appendChild(grid);
      box.appendChild(el('p', null, data.reply));
      m.appendChild(box);
    } else if (q && q.now && data.action === 'PROBE') {
      m = el('div', 'msg card');
      m.appendChild(el('span', 'who', 'Matcher · checking something you said'));
      var box2 = el('div', 'card-box');
      var one = el('div', 'quotes one');
      one.appendChild(quoteFigure('Just now', q.now));
      box2.appendChild(one);
      box2.appendChild(el('p', null, data.reply));
      m.appendChild(box2);
    } else {
      m = el('div', 'msg');
      m.appendChild(el('span', 'who', 'Matcher'));
      m.appendChild(el('p', 'bubble', data.reply));
    }
    els.transcript.appendChild(m);
    scrollToEnd();
  }

  function addNotice(text) {
    els.transcript.appendChild(el('p', 'notice', text));
    scrollToEnd();
  }

  function showTyping() {
    hideTyping();
    var t = el('div', 'typing');
    t.setAttribute('role', 'status');
    t.setAttribute('aria-label', 'Matcher is typing');
    for (var i = 0; i < 3; i++) {
      var d = el('span');
      d.setAttribute('aria-hidden', 'true');
      t.appendChild(d);
    }
    state.typingEl = t;
    els.transcript.appendChild(t);
    scrollToEnd();
  }
  function hideTyping() {
    if (state.typingEl && state.typingEl.parentNode) state.typingEl.parentNode.removeChild(state.typingEl);
    state.typingEl = null;
  }

  /* ---------- header / composer state ---------- */
  function updateHeader(turn, max) {
    if (turn !== state.turn) replay(els.progress, 'tick');
    state.turn = turn;
    state.maxTurns = max || state.maxTurns;
    var label = 'Turn ' + state.turn + ' of up to ' + state.maxTurns;
    els.turnText.textContent = label;
    els.progress.setAttribute('aria-label', label);
    els.progress.setAttribute('aria-valuemax', String(state.maxTurns));
    els.progress.setAttribute('aria-valuenow', String(Math.min(state.turn, state.maxTurns)));
    var pct = state.maxTurns > 0 ? Math.min(100, (state.turn / state.maxTurns) * 100) : 0;
    els.barFill.style.width = pct + '%';
  }

  function setCode(code) {
    state.code = code;
    els.chip.textContent = code;
    els.chip.hidden = false;
  }

  function syncComposer() {
    var off = state.busy || state.finished || !state.sessionId;
    els.msg.disabled = off;
    els.send.disabled = off;
  }

  function updateCounter() {
    var n = els.msg.value.length;
    if (n >= COUNTER_FROM) {
      els.counter.hidden = false;
      els.counter.textContent = n + ' / ' + MAX_CHARS;
      els.counter.classList.toggle('over', n >= MAX_CHARS);
    } else {
      els.counter.hidden = true;
    }
  }

  function autosize() {
    els.msg.style.height = 'auto';
    els.msg.style.height = Math.min(els.msg.scrollHeight + 2, 200) + 'px';
  }

  function clearError() {
    els.error.hidden = true;
    while (els.error.firstChild) els.error.removeChild(els.error.firstChild);
  }

  function showError(message, action) {
    clearError();
    els.error.appendChild(el('span', null, message));
    if (action) {
      var b = el('button', null, action.label);
      b.type = 'button';
      b.addEventListener('click', action.onClick);
      els.error.appendChild(b);
    }
    els.error.hidden = false;
  }

  function focusInput() {
    if (!els.msg.disabled) els.msg.focus();
  }

  /* ---------- API ---------- */
  function api(path, body) {
    return fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().catch(function () { return null; }).then(function (data) {
        if (!res.ok) {
          var err = new Error('HTTP ' + res.status);
          err.status = res.status;
          throw err;
        }
        return data;
      });
    });
  }

  function handleError(err, text, retryable) {
    var status = err && err.status;
    if (status === 404) {
      state.finished = true;
      showError('Your session expired, so this conversation can’t continue.', {
        label: 'Start again', onClick: function () { window.location.reload(); }
      });
    } else if (status === 409) {
      state.finished = true;
      showError('This conversation has already finished.');
    } else if (status === 422) {
      showError('That message couldn’t be sent. Please check it and try again.');
    } else {
      var msg = 'Something went wrong reaching the server. Your message was not lost.';
      showError(msg, retryable ? {
        label: 'Retry', onClick: function () { send(text, true); }
      } : null);
    }
    syncComposer();
  }

  /* ---------- session start ---------- */
  function startSession() {
    var params = new URLSearchParams(window.location.search);
    var p = params.get('p');
    var debug = params.get('debug') !== '0';
    state.busy = true;
    syncComposer();
    showTyping();
    api('/api/session', { participant_code: p ? p : null, debug: debug }).then(function (data) {
      hideTyping();
      state.busy = false;
      state.sessionId = data.session_id;
      setCode(data.participant_code);
      updateHeader(data.turn, data.max_turns);
      addBot(data);
      renderTrace(data.trace);
      syncComposer();
      focusInput();
    }).catch(function (err) {
      hideTyping();
      state.busy = false;
      var status = err && err.status;
      showError(status === 422
        ? 'That participant code isn’t valid. Use letters, numbers, - or _ (up to 32 characters).'
        : 'Couldn’t start the conversation.', {
        label: status === 422 ? 'Start without a code' : 'Try again',
        onClick: function () {
          window.location.href = status === 422 ? window.location.pathname + (debug ? '' : '?debug=0') : window.location.href;
        }
      });
    });
  }

  /* ---------- send ---------- */
  function newMessageId() {
    try { if (window.crypto && crypto.randomUUID) return crypto.randomUUID(); } catch (e) { /* fall through */ }
    var id = '';
    for (var i = 0; i < 32; i++) id += Math.floor(Math.random() * 16).toString(16);
    return id;
  }

  function send(text, isRetry) {
    if (state.busy || state.finished || !state.sessionId) return;
    text = text.trim();
    if (!text) return;
    if (text.length > MAX_CHARS) text = text.slice(0, MAX_CHARS);

    clearError();
    state.busy = true;
    state.lastText = text;
    if (!isRetry || !state.messageId) state.messageId = newMessageId();  // a retry re-sends the same id so the server can dedupe
    var messageId = state.messageId;
    if (!isRetry) {
      addUser(text);
      els.msg.value = '';
      autosize();
      updateCounter();
    }
    syncComposer();
    showTyping();

    api('/api/chat', { session_id: state.sessionId, text: text, message_id: messageId }).then(function (data) {
      hideTyping();
      state.busy = false;
      state.lastText = null;
      updateHeader(data.turn, data.max_turns);
      addBot(data);
      renderTrace(data.trace);
      if (data.done) {
        state.finished = true;
        if (data.recommendations && data.recommendations.length) renderResults(data.recommendations);
      }
      syncComposer();
      focusInput();
    }).catch(function (err) {
      hideTyping();
      state.busy = false;
      // Keep the typed text recoverable only through Retry; the bubble is already in the transcript.
      handleError(err, text, true);
    });
  }

  /* ---------- results ---------- */
  function renderResults(recs) {
    while (els.results.firstChild) els.results.removeChild(els.results.firstChild);
    var wrap = el('section', 'results');
    wrap.setAttribute('aria-labelledby', 'resultsH');

    var head = el('div');
    var semanticAssociation = recs.some(function (r) { return r.ranking_basis === 'semantic_association'; });
    var h = el('h2', 'results-h', semanticAssociation ? 'Five occupations related to what you described' : 'Five occupations that fit what you told me');
    h.id = 'resultsH';
    h.tabIndex = -1;
    head.appendChild(h);
    var hasTasks = recs.some(function (r) { return r.tasks && r.tasks.length; });
    head.appendChild(el('p', 'lead', semanticAssociation
      ? 'These experimental results reflect topic and background associations. They do not establish a career preference.'
      : hasTasks
      ? 'Under each one are the O*NET task statements that matched your messages most closely. They are the reason it is on this list.'
      : 'Each one is ranked by how well its O*NET interest profile fits the interests you described.'));
    wrap.appendChild(head);

    wrap.appendChild(el('p', 'reminder', 'A reminder: this is a research prototype, not career advice.'));

    var ol = el('ol', 'recs');
    recs.forEach(function (r, i) {
      var li = el('li', 'rec');
      li.style.setProperty('--i', String(i));
      var n =el('span', 'rec-n', r.rank || (i + 1));
      n.setAttribute('aria-hidden', 'true');
      li.appendChild(n);
      var rh = el('div', 'rec-head');
      rh.appendChild(el('h3', null, r.title));
      rh.appendChild(el('span', 'rec-code', 'O*NET ' + r.code));
      li.appendChild(rh);
      if (r.description) li.appendChild(el('p', 'rec-desc', r.description));
      li.appendChild(el('p', 'rec-why', r.why));
      if (r.tasks && r.tasks.length) {
        var tasks = el('div', 'rec-tasks');
        tasks.appendChild(el('p', null, 'Provisional task matches. Similarity alone does not confirm that you enjoy these activities.'));
        r.tasks.forEach(function (t) {
          tasks.appendChild(el('blockquote', null, '“' + t + '”'));
        });
        li.appendChild(tasks);
      }
      ol.appendChild(li);
    });
    wrap.appendChild(ol);

    var byg = el('section', 'before-you-go');
    byg.appendChild(el('h2', null, 'Before you go'));
    byg.appendChild(el('p', null, (semanticAssociation ? 'These five use experimental occupation text retrieval. ' : hasTasks ? 'These five come from a six-number interest profile and keyword matching. ' : 'These five come from a six-number interest profile. ') + 'They can’t know your grades, your circumstances or what you’d be good at. For real decisions, talk to SMU Dato’ Kho Hui Meng Career Centre.'));
    var links = el('div', 'links');
    // TODO: replace placeholder hrefs with the real rating form URL and SMU careers service URL.
    var a1 = el('a', 'primary', 'Rate these five (2 min)');
    a1.href = '#rating-form';
    var a2 = el('a', null, 'SMU careers service');
    a2.href = '#careers-service';
    links.appendChild(a1);
    links.appendChild(a2);
    byg.appendChild(links);
    var w = el('p', 'withdraw');
    w.appendChild(document.createTextNode('Changed your mind about taking part? Tell the researcher your code, '));
    w.appendChild(el('code', null, state.code || ''));
    w.appendChild(document.createTextNode(', and your transcript will be deleted.'));
    byg.appendChild(w);
    wrap.appendChild(byg);

    els.results.appendChild(wrap);
    els.msg.placeholder = 'The conversation is finished.';
    h.scrollIntoView({ block: 'start', behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  /* ---------- researcher panel ---------- */
  // The panels are built once and then updated in place, so only what changed animates.
  var rv = { built: false, slots: {}, feedCount: 0, prevProfile: null, scoresSub: null };

  function buildResearcher() {
    rv.built = true;
    els.layout.classList.add('has-researcher');
    els.rProfile.hidden = false;
    els.rPolicy.hidden = false;

    // accumulated RIASEC profile
    var sc = els.scoresCard;
    var head = el('div', 'rcard-head');
    head.appendChild(el('h2', null, 'RIASEC profile'));
    rv.scoresSub = el('span', 'rcard-sub');
    head.appendChild(rv.scoresSub);
    sc.appendChild(head);
    var list = el('div', 'scores');
    LETTERS.forEach(function (L) {
      var row = el('div', 'slot');
      row.appendChild(el('span', 'letter', L));
      var bars = el('div', 'bars');
      bars.setAttribute('aria-hidden', 'true');
      var track = el('div', 'track');
      track.appendChild(el('div', 'mid'));
      var fill = el('div', 'fill');
      track.appendChild(fill);
      var conf = el('div', 'conf');
      var confFill = el('i');
      conf.appendChild(confFill);
      bars.appendChild(track);
      bars.appendChild(conf);
      row.appendChild(bars);
      var val = el('span', 'val', '—');
      var delta = el('span', 'delta', '');
      delta.setAttribute('aria-hidden', 'true');
      row.appendChild(val);
      row.appendChild(delta);
      list.appendChild(row);
      rv.slots[L] = { row: row, fill: fill, confFill: confFill, val: val, delta: delta };
    });
    sc.appendChild(list);
    sc.appendChild(el('span', 'hint', 'Blue = likes, orange = dislikes. Dash = not yet observed. The thin line under each bar is confidence; the small number is the change since the last turn.'));

    mapView.section = els.mapCard;
    mapView.rankEl = els.rankCard;
    mapView.notesEl = els.mapNotesBody;
  }

  function renderScores(profile) {
    rv.scoresSub.textContent = 'after turn ' + state.turn;
    var prev = rv.prevProfile;
    LETTERS.forEach(function (L) {
      var s = rv.slots[L];
      var slot = profile[L] || {};
      var v = slot.value;
      var label, color;
      if (typeof v === 'number') {
        var w = Math.min(1, Math.abs(v)) * 50;
        s.fill.style.width = w + '%';
        s.fill.style.left = (v >= 0 ? 50 : 50 - w) + '%';
        s.fill.style.backgroundColor = v >= 0 ? 'var(--pos)' : 'var(--neg)';
        color = v >= 0 ? 'var(--pos-text)' : 'var(--neg-text)';
        label = (v >= 0 ? '+' : '') + v.toFixed(2);
      } else {
        s.fill.style.width = '0%';
        s.fill.style.left = '50%';
        label = '—';
        color = '#A3AEBA';
      }
      s.val.textContent = label;
      s.val.style.color = color;
      var c = Number(slot.confidence);
      s.confFill.style.width = (isFinite(c) ? Math.max(0, Math.min(1, c)) * 100 : 0) + '%';
      s.row.title = slot.reason || '';
      s.row.setAttribute('aria-label', L + ' ' + (label === '—' ? 'not yet observed' : label) +
        (isFinite(c) ? ', confidence ' + c.toFixed(2) : ''));

      // change since the previous turn
      var before = prev && prev[L];
      var d = (typeof v === 'number' ? v : 0) - (typeof before === 'number' ? before : 0);
      var moved = prev && (typeof v === 'number' || typeof before === 'number') && Math.abs(d) >= 0.005;
      s.delta.className = 'delta';
      if (moved) {
        s.delta.textContent = (d > 0 ? '+' : '−') + Math.abs(d).toFixed(2);
        s.delta.classList.add(d > 0 ? 'up' : 'down');
        replay(s.delta, 'bump');
        replay(s.row, 'changed');
      } else {
        s.delta.textContent = '';
        s.row.classList.remove('changed');
      }
    });
    rv.prevProfile = {};
    LETTERS.forEach(function (L) { rv.prevProfile[L] = profile[L] ? profile[L].value : null; });
  }

  function isPositiveAction(action) { return action === 'CLARIFY' || action === 'RECOMMEND'; }

  function renderFeed(actions) {
    var feed = els.feed;
    if (actions.length < rv.feedCount) {         // trace was reset: start over
      while (feed.firstChild) feed.removeChild(feed.firstChild);
      rv.feedCount = 0;
    }
    var empty = feed.querySelector('.feed-empty');
    if (empty && actions.length) feed.removeChild(empty);
    var fresh = actions.slice(rv.feedCount);
    fresh.forEach(function (a) {
      var li = el('li', 'feed-item is-new' + (isPositiveAction(a.action) ? ' pos' : ''));
      li.appendChild(el('span', 'feed-dot'));
      var body = el('div', 'feed-body');
      var head = el('div', 'feed-head');
      head.appendChild(el('span', 'feed-act', a.action));
      head.appendChild(el('span', 'feed-turn', 'Turn ' + a.turn));
      body.appendChild(head);
      body.appendChild(el('p', 'feed-why', a.why));
      li.appendChild(body);
      feed.insertBefore(li, feed.firstChild);   // newest first
    });
    rv.feedCount = actions.length;
    if (!actions.length && !feed.firstChild) feed.appendChild(el('li', 'feed-empty', 'Policy decisions appear here as the conversation goes on.'));
    var items = feed.querySelectorAll('.feed-item');
    for (var i = 0; i < items.length; i++) items[i].classList.toggle('is-latest', i === 0);
    els.feedCount.textContent = actions.length + (actions.length === 1 ? ' decision' : ' decisions');
    if (fresh.length && feed.scrollTop > 0) feed.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  function renderComps(comps) {
    var dl = els.compsList;
    while (dl.firstChild) dl.removeChild(dl.firstChild);
    Object.keys(comps).forEach(function (k) {
      dl.appendChild(el('dt', null, k));
      dl.appendChild(el('dd', null, comps[k]));
    });
  }

  // A researcher-panel failure must never block the conversation itself.
  function renderTrace(trace) {
    try {
      renderTraceUnsafe(trace);
    } catch (e) {
      if (window.console && console.error) console.error('Researcher panel failed to render', e);
      // Only researcher sessions receive a trace, so this notice never reaches participants.
      if (!rv.failed) {
        rv.failed = true;
        addNotice('The researcher view failed to load. Reload with Ctrl+Shift+R; if it persists, check the browser console.');
      }
    }
  }

  function renderTraceUnsafe(trace) {
    if (!trace) return;
    var first = !rv.built;
    if (first) buildResearcher();

    renderScores(trace.profile || {});
    renderFeed(trace.actions || []);
    renderComps(trace.components || {});

    var rankingRows = trace.ranking_trace || [];
    var latestRanking = rankingRows[rankingRows.length - 1];
    var semantic = latestRanking && (latestRanking.query_diagnostics || {}).occupation_semantic;
    els.rawCard.hidden = !semantic;
    els.rawPre.textContent = semantic ? JSON.stringify(semantic, null, 2) : '';

    // occupation map
    mapView.rows = rankingRows;
    mapView.top5 = trace.top5 || [];
    mapView.fresh = !first;
    loadMap();
    renderMap();
  }

  /* ---------- occupation map: per-turn rank changes on a fixed layout ---------- */
  var SVGNS = 'http://www.w3.org/2000/svg';
  var COMPONENT_LABEL = { structured: 'interest profile (M1)', semantic: 'semantic match (M4)', lexical: 'keywords (M3)' };
  var mapView = {
    map: null, loading: false, error: null, section: null, rows: [],
    tab: null, turnIdx: null, selected: null, hover: null,
    rankEl: null, notesEl: null, top5: [], fresh: false, switching: false, lastMe: null
  };

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // Live top 5 from the trace: shown when the map has no usable ranks for this turn.
  function renderTop5Fallback() {
    var box = mapView.rankEl;
    if (!box) return;
    clear(box);
    var head = el('div', 'rcard-head');
    head.appendChild(el('h2', null, 'Live top 5'));
    box.appendChild(head);
    if (!mapView.top5.length) { box.appendChild(el('p', 'hint', 'No ranking yet.')); return; }
    var ol = el('ol', 'top5');
    mapView.top5.forEach(function (o, i) {
      var li = el('li', null, o.title);
      li.style.setProperty('--i', String(i));
      ol.appendChild(li);
    });
    box.appendChild(ol);
  }

  function renderNotes(hex) {
    var box = mapView.notesEl;
    if (!box) return;
    clear(box);
    box.appendChild(legend(hex));
    var notes = hex ? [
      'Each job sits towards the types its O*NET z-scores are positive in. The diamond is your profile over observed types; the faded line is its path over earlier turns.',
      'Six dimensions squashed into two: opposite types cancel, so a strong R + S profile and a flat one both land near the centre.',
      'Two jobs at the same point can rank differently, and the marker shows only the cosine half. With fusion on, a far-away job can still rank highly from the semantic match; the details say which half moved it.'
    ] : [
      'Jobs close together have similar O*NET text under the fine-tuned encoder. t-SNE keeps neighbourhoods, not global distances: the gap between two clusters means nothing.'
    ];
    notes.forEach(function (n) { box.appendChild(el('p', 'hint', n)); });
  }

  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function loadMap() {
    if (mapView.map || mapView.loading) return;
    mapView.loading = true;
    fetch('/api/occupation_map').then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    }).then(function (map) {
      mapView.map = map;
      mapView.index = {};
      map.codes.forEach(function (c, i) { mapView.index[c] = i; });
      if (!mapView.tab) mapView.tab = map.semantic ? 'semantic' : 'hexagon';
    }).catch(function (err) {
      mapView.error = 'Occupation map unavailable (' + err.message + ').';
    }).then(function () {
      mapView.loading = false;
      renderMap();
    });
  }

  // Same formula as recommender/build_occupation_map.py hexagon_point.
  function hexPoint(values, types, corners) {
    var x = 0, y = 0, total = 0;
    types.forEach(function (t) {
      var w = Math.max(0, Number(values[t]) || 0);
      x += w * corners[t][0]; y += w * corners[t][1]; total += w;
    });
    return total > 0 ? [x / total, y / total] : null;
  }

  function logChange(prev, now) {
    return (prev == null || now == null) ? null : Math.log(prev) - Math.log(now);  // > 0 means it rose
  }

  // Rank-change category for each map point; thresholds are provisional.
  function category(i, snap, prev) {
    var r = snap.fused[i], top = r != null && r <= 10;
    if (!prev) return { top: top, kind: 'flat', d: 0 };
    var p = prev.fused[i], wasTop = p != null && p <= 10;
    var d = logChange(p, r);
    var kind = d == null ? 'flat' : d >= 0.5 ? 'rose' : d <= -0.5 ? 'fell' : 'flat';
    return { top: top, kind: kind, d: d || 0, entered: top && !wasTop, left: wasTop && !top };
  }

  function movedBy(i, snap, prev) {
    if (!prev) return null;
    var d = {};
    Object.keys(COMPONENT_LABEL).forEach(function (k) {
      if (snap[k] && prev[k]) {
        var c = logChange(prev[k][i], snap[k][i]);
        if (c != null) d[k] = Math.abs(c);
      }
    });
    var keys = Object.keys(d);
    if (!keys.length) return 'no component ranks recorded';
    if (keys.length === 1) return COMPONENT_LABEL[keys[0]] + ' (only usable component)';
    var s = d.structured || 0, m = d.semantic || 0;
    if (Math.max(s, m) < 0.1) return 'neither component moved much';
    if (s >= 1.5 * m) return COMPONENT_LABEL.structured;
    if (m >= 1.5 * s) return COMPONENT_LABEL.semantic;
    return 'both';
  }

  function usableSnapshot(row, map) {
    var s = row && row.rank_snapshot;
    return s && s.codes_sha256 === map.codes_sha256 ? s : null;
  }

  function renderMap() {
    var sec = mapView.section;
    if (!sec) return;
    var fresh = mapView.fresh, switching = mapView.switching;
    mapView.fresh = false;
    mapView.switching = false;
    clear(sec);
    sec.appendChild(el('h2', null, 'Occupation map'));
    var map = mapView.map, rows = mapView.rows || [];
    renderNotes(!map || !map.semantic || mapView.tab === 'hexagon');
    if (mapView.error || !map || !rows.length) {
      sec.appendChild(el('p', 'hint', mapView.error || (!map ? 'Loading map…' : 'The map is coloured from the first ranked turn.')));
      renderTop5Fallback();
      return;
    }

    var idx = mapView.turnIdx == null ? rows.length - 1 : Math.min(mapView.turnIdx, rows.length - 1);
    var row = rows[idx];
    var snap = usableSnapshot(row, map);
    var prev = idx > 0 ? usableSnapshot(rows[idx - 1], map) : null;
    if (mapView.tab === 'semantic' && !map.semantic) mapView.tab = 'hexagon';

    // tabs
    var tabs = el('div', 'omap-tabs');
    tabs.setAttribute('role', 'tablist');
    [['semantic', 'Semantic map'], ['hexagon', 'RIASEC hexagon']].forEach(function (t) {
      var b = el('button', 'omap-tab', t[1]);
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(mapView.tab === t[0]));
      if (t[0] === 'semantic' && !map.semantic) {
        b.disabled = true;
        b.title = map.semantic_unavailable || 'unavailable';
      }
      b.addEventListener('click', function () {
        if (mapView.tab === t[0]) return;
        mapView.tab = t[0];
        mapView.switching = true;
        renderMap();
      });
      tabs.appendChild(b);
    });
    sec.appendChild(tabs);
    if (!map.semantic) sec.appendChild(el('p', 'hint', 'Semantic map hidden: ' + map.semantic_unavailable));

    // turn slider
    var sliderRow = el('div', 'omap-slider');
    var sliderId = 'omapTurn';
    var lab = el('label', null, 'Turn ' + row.turn + (idx === rows.length - 1 ? ' (latest)' : ''));
    lab.htmlFor = sliderId;
    var slider = el('input');
    slider.type = 'range'; slider.id = sliderId; slider.min = '0'; slider.max = String(rows.length - 1);
    slider.step = '1'; slider.value = String(idx); slider.disabled = rows.length < 2;
    slider.addEventListener('input', function () {
      var v = Number(slider.value);
      mapView.turnIdx = v === rows.length - 1 ? null : v;   // the latest position follows new turns
      renderMap();
      var again = document.getElementById(sliderId);
      if (again) again.focus();
    });
    sliderRow.appendChild(lab);
    sliderRow.appendChild(slider);
    sec.appendChild(sliderRow);

    if (!snap) {
      sec.appendChild(el('p', 'hint', row.rank_snapshot
        ? 'This turn’s ranks use a different occupation order than the map, so they are not shown.'
        : row.status !== 'ranked' ? 'No ranking this turn (' + (row.reason || row.status) + ').'
        : 'No full-pool ranks were recorded for this turn.'));
    } else if (!prev) {
      sec.appendChild(el('p', 'hint', idx === 0 ? 'First ranked turn: only the top 10 are shown.'
        : 'The previous turn has no ranks to compare with: only the top 10 are shown.'));
    }

    // plot
    var hex = mapView.tab === 'hexagon';
    var pts = hex ? map.hexagon.coords : map.semantic.map(function (c) { return [c[0] * 2, c[1] * 2]; });
    var svg = svgEl('svg', { viewBox: '-1.2 -1.2 2.4 2.4', role: 'img',
      class: 'omap-svg' + (switching ? ' is-switch' : fresh ? ' is-fresh' : ''),
      'aria-label': (hex ? 'RIASEC hexagon' : 'Semantic map') + ' of ' + map.codes.length +
        ' occupations, coloured by rank change at turn ' + row.turn });
    if (hex) {
      var corners = map.hexagon.corners;
      svg.appendChild(svgEl('polygon', { class: 'omap-hex', points: LETTERS.map(function (L) { return corners[L].join(','); }).join(' ') }));
      LETTERS.forEach(function (L) {
        var t = svgEl('text', { x: corners[L][0] * 1.1, y: corners[L][1] * 1.1 + 0.035, class: 'omap-corner' });
        t.textContent = L;
        svg.appendChild(t);
      });
    }
    var cats = map.codes.map(function (_, i) { return snap ? category(i, snap, prev) : { top: false, kind: 'flat', d: 0 }; });
    var order = map.codes.map(function (_, i) { return i; }).sort(function (a, b) {
      var w = function (c) { return c.top ? 3 : c.kind !== 'flat' || c.left ? 2 : 1; };
      return w(cats[a]) - w(cats[b]);
    });
    var R = 0.03;
    order.forEach(function (i) {
      var c = cats[i], x = pts[i][0], y = pts[i][1];
      if (c.top) {
        if (c.entered) svg.appendChild(svgEl('circle', { cx: x, cy: y, r: R * 2.1, class: 'omap-ring-in' }));
        svg.appendChild(svgEl('circle', { cx: x, cy: y, r: R * 1.4, class: 'omap-top' }));
        var t = svgEl('text', { x: x + R * 1.9, y: y + 0.025, class: 'omap-label' });
        t.textContent = String(snap.fused[i]);
        svg.appendChild(t);
      } else if (c.left) {
        svg.appendChild(svgEl('circle', { cx: x, cy: y, r: R * 1.4, class: 'omap-ring-out' }));
      } else {
        var dot = svgEl('circle', { cx: x, cy: y, r: R, class: 'omap-dot ' + c.kind });
        if (c.kind !== 'flat') dot.setAttribute('fill-opacity', String(Math.min(1, 0.45 + (Math.abs(c.d) - 0.5) / 2)));
        svg.appendChild(dot);
      }
    });
    if (hex) drawProfile(svg, rows, idx, map.hexagon.corners, fresh && idx === rows.length - 1);
    var selRing = svgEl('circle', { r: R * 2.6, class: 'omap-sel' });
    selRing.style.display = 'none';
    svg.appendChild(selRing);
    sec.appendChild(svg);

    // details box (hover previews, click/tap pins)
    var info = el('div', 'omap-info');
    info.setAttribute('aria-live', 'polite');
    sec.appendChild(info);
    function show(code) {
      var i = code == null ? null : mapView.index[code];
      if (i == null) {
        selRing.style.display = 'none';
        info.textContent = '';
        info.appendChild(el('span', 'hint', 'Hover or tap an occupation for its ranks.'));
        return;
      }
      selRing.setAttribute('cx', pts[i][0]); selRing.setAttribute('cy', pts[i][1]);
      selRing.style.display = '';
      renderDetails(info, map, i, snap, prev);
    }
    function nearest(evt) {
      var ctm = svg.getScreenCTM();
      if (!ctm) return null;
      var pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      var q = pt.matrixTransform(ctm.inverse()), best = null, bestD = 0.09 * 0.09;
      pts.forEach(function (p, i) {
        var d = (p[0] - q.x) * (p[0] - q.x) + (p[1] - q.y) * (p[1] - q.y);
        if (d < bestD) { bestD = d; best = map.codes[i]; }
      });
      return best;
    }
    svg.addEventListener('pointermove', function (e) { show(nearest(e) || mapView.selected); });
    svg.addEventListener('pointerleave', function () { show(mapView.selected); });
    svg.addEventListener('click', function (e) { mapView.selected = nearest(e); show(mapView.selected); });
    show(mapView.selected);

    // keyboard-reachable top 10 (its own card; selecting one highlights it on the map)
    if (snap) {
      var rank = mapView.rankEl;
      clear(rank);
      var rh = el('div', 'rcard-head');
      rh.appendChild(el('h2', null, 'Top 10'));
      rh.appendChild(el('span', 'rcard-sub', 'turn ' + row.turn));
      rank.appendChild(rh);
      var top = map.codes.map(function (c, i) { return i; })
        .filter(function (i) { return snap.fused[i] != null && snap.fused[i] <= 10; })
        .sort(function (a, b) { return snap.fused[a] - snap.fused[b]; });
      var ol = el('ol', 'omap-top10');
      top.forEach(function (i, n) {
        var li = el('li');
        if (fresh) li.style.setProperty('--i', String(n));
        else li.style.animation = 'none';
        var b = el('button', null, map.titles[i]);
        b.type = 'button';
        if (cats[i].entered) b.appendChild(el('span', 'omap-new', ' new'));
        b.addEventListener('click', function () { mapView.selected = map.codes[i]; show(mapView.selected); });
        li.appendChild(b);
        ol.appendChild(li);
      });
      rank.appendChild(ol);
    } else {
      renderTop5Fallback();
    }
  }

  function drawProfile(svg, rows, idx, corners, animate) {
    var path = [];
    for (var k = 0; k <= idx; k++) {
      var r = rows[k];
      path.push(hexPoint(r.interests || {}, r.observed_types || [], corners) || [0, 0]);
    }
    if (path.length > 1) {
      svg.appendChild(svgEl('polyline', { class: 'omap-trail', points: path.map(function (p) { return p.join(','); }).join(' ') }));
      path.slice(0, -1).forEach(function (p) { svg.appendChild(svgEl('circle', { cx: p[0], cy: p[1], r: 0.022, class: 'omap-trail-pt' })); });
    }
    var cur = path[path.length - 1], s = 0.075;
    var me = svgEl('polygon', { class: 'omap-me',
      points: [[cur[0], cur[1] - s], [cur[0] + s, cur[1]], [cur[0], cur[1] + s], [cur[0] - s, cur[1]]].map(function (p) { return p.join(','); }).join(' ') });
    svg.appendChild(me);
    // glide the diamond from where it sat last turn (CSS px = user units inside the SVG)
    var last = mapView.lastMe;
    if (animate && last && !reduceMotion && me.animate && (last[0] !== cur[0] || last[1] !== cur[1])) {
      try {
        me.animate([
          { transform: 'translate(' + (last[0] - cur[0]) + 'px,' + (last[1] - cur[1]) + 'px)' },
          { transform: 'translate(0px,0px)' }
        ], { duration: 900, easing: 'cubic-bezier(.2,.8,.2,1)' });
      } catch (e) { /* animation is decorative */ }
    }
    if (idx === rows.length - 1) mapView.lastMe = cur;
  }

  function rankPair(now, before, hasPrev) {
    var a = now == null ? '—' : '#' + now;
    if (!hasPrev) return a;
    return a + ' (was ' + (before == null ? '—' : '#' + before) + ')';
  }

  function renderDetails(box, map, i, snap, prev) {
    box.textContent = '';
    box.appendChild(el('strong', null, map.titles[i]));
    box.appendChild(el('span', 'omap-code', ' ' + map.codes[i]));
    if (!snap) return;
    var dl = el('dl', 'comps');
    var add = function (k, v) { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', null, v)); };
    add('fused', rankPair(snap.fused[i], prev && prev.fused[i], !!prev));
    Object.keys(COMPONENT_LABEL).forEach(function (k) {
      if (snap[k]) add(k === 'structured' ? 'M1' : k === 'semantic' ? 'M4' : 'M3',
        rankPair(snap[k][i], prev && prev[k] && prev[k][i], !!(prev && prev[k])));
    });
    var why = movedBy(i, snap, prev);
    if (why) add('moved by', why);
    box.appendChild(dl);
  }

  function legend(hex) {
    var ul = el('ul', 'omap-legend');
    [['omap-top', 'Top 10 (number = rank)'], ['omap-ring-in', 'Entered top 10'], ['omap-ring-out', 'Left top 10'],
     ['omap-dot rose', 'Rose'], ['omap-dot fell', 'Fell'], ['omap-dot flat', 'Little change']]
      .concat(hex ? [['omap-me', 'Your profile']] : [])
      .forEach(function (item) {
        var li = el('li');
        var s = svgEl('svg', { viewBox: '-1 -1 2 2', width: '14', height: '14', 'aria-hidden': 'true' });
        s.appendChild(item[0] === 'omap-me'
          ? svgEl('polygon', { points: '0,-0.9 0.9,0 0,0.9 -0.9,0', class: 'omap-me' })
          : svgEl('circle', { r: item[0].indexOf('ring') >= 0 ? 0.7 : 0.6, class: item[0] }));
        li.appendChild(s);
        li.appendChild(el('span', null, item[1]));
        ul.appendChild(li);
      });
    var li = el('li', 'omap-legend-note', 'Rose/fell = at least a 0.5 change in log rank since the previous turn; stronger colour, bigger move.');
    ul.appendChild(li);
    return ul;
  }


  /* ---------- events ---------- */
  els.form.addEventListener('submit', function (e) {
    e.preventDefault();
    send(els.msg.value, false);
  });

  els.msg.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send(els.msg.value, false);
    }
  });

  els.msg.addEventListener('input', function () {
    updateCounter();
    autosize();
  });

  els.stop.addEventListener('click', function () {
    if (state.finished) return;
    if (!window.confirm('Stop this session? You won’t be able to continue the conversation.')) return;
    state.finished = true;
    hideTyping();
    clearError();
    syncComposer();
    els.stop.disabled = true;
    els.msg.placeholder = 'The session has ended.';
    addNotice('Session stopped. Thank you for taking part. If you want your transcript deleted, tell the researcher your code' + (state.code ? ' (' + state.code + ')' : '') + '.');
  });

  updateHeader(0, 10);
  syncComposer();
  startSession();
})();
