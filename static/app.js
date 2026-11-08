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
    researcher: $('researcher')
  };

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
    var last = els.results.lastElementChild || els.transcript.lastElementChild;
    if (last && last.scrollIntoView) last.scrollIntoView({ block: 'end', behavior: 'auto' });
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
    var h = el('h2', 'results-h', 'Five occupations that fit what you told me');
    h.id = 'resultsH';
    h.tabIndex = -1;
    head.appendChild(h);
    var hasTasks = recs.some(function (r) { return r.tasks && r.tasks.length; });
    head.appendChild(el('p', 'lead', hasTasks
      ? 'Under each one are the O*NET task statements that matched your messages most closely. They are the reason it is on this list.'
      : 'Each one is ranked by how well its O*NET interest profile fits the interests you described.'));
    wrap.appendChild(head);

    wrap.appendChild(el('p', 'reminder', 'A reminder: this is a research prototype, not career advice.'));

    var ol = el('ol', 'recs');
    recs.forEach(function (r, i) {
      var li = el('li', 'rec');
      var n = el('span', 'rec-n', r.rank || (i + 1));
      n.setAttribute('aria-hidden', 'true');
      li.appendChild(n);
      var rh = el('div', 'rec-head');
      rh.appendChild(el('h3', null, r.title));
      rh.appendChild(el('span', 'rec-code', 'O*NET ' + r.code));
      li.appendChild(rh);
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
    byg.appendChild(el('p', null, (hasTasks ? 'These five come from a six-number interest profile and keyword matching. ' : 'These five come from a six-number interest profile. ') + 'They can’t know your grades, your circumstances or what you’d be good at. For real decisions, talk to SMU Dato’ Kho Hui Meng Career Centre.'));
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
    h.scrollIntoView({ block: 'start', behavior: 'auto' });
  }

  /* ---------- researcher panel ---------- */
  function renderTrace(trace) {
    if (!trace) return;
    var p = els.researcher;
    while (p.firstChild) p.removeChild(p.firstChild);
    p.hidden = false;
    p.appendChild(el('strong', 'r-title', 'Researcher view'));

    // profile
    var s1 = el('section');
    s1.appendChild(el('h2', null, 'Profile after turn ' + state.turn));
    LETTERS.forEach(function (L) {
      var slot = (trace.profile && trace.profile[L]) || {};
      var v = slot.value;
      var row = el('div', 'slot');
      row.appendChild(el('span', 'letter', L));
      var track = el('div', 'track');
      track.setAttribute('aria-hidden', 'true');
      track.appendChild(el('div', 'mid'));
      var label, color;
      if (typeof v === 'number') {
        var fill = el('div', 'fill');
        fill.style.width = (Math.min(1, Math.abs(v)) * 50) + '%';
        if (v >= 0) { fill.style.left = '50%'; fill.style.background = 'var(--pos)'; color = 'var(--pos-text)'; }
        else { fill.style.right = '50%'; fill.style.background = 'var(--neg)'; color = 'var(--neg-text)'; }
        track.appendChild(fill);
        label = (v >= 0 ? '+' : '') + v.toFixed(2);
      } else {
        label = '—';
        color = '#A3AEBA';
      }
      row.appendChild(track);
      var val = el('span', 'val', label);
      val.style.color = color;
      row.appendChild(val);
      if (slot.reason) row.title = slot.reason;
      s1.appendChild(row);
    });
    s1.appendChild(el('span', 'hint', 'Blue = likes, orange = dislikes. Dash = not yet observed.'));
    p.appendChild(s1);

    // policy trace
    var s2 = el('section');
    s2.appendChild(el('h2', null, 'Policy trace'));
    (trace.actions || []).forEach(function (a) {
      var row = el('div', 'trace-row');
      row.appendChild(el('span', 't', 'T' + a.turn));
      var act = el('span', 'a', a.action);
      act.style.color = (a.action === 'CLARIFY' || a.action === 'RECOMMEND') ? 'var(--pos-text)' : 'var(--neg-text)';
      row.appendChild(act);
      row.appendChild(el('span', 'w', a.why));
      s2.appendChild(row);
    });
    p.appendChild(s2);

    // top 5
    var s3 = el('section');
    s3.appendChild(el('h2', null, 'Live top 5'));
    var ol = el('ol', 'top5');
    (trace.top5 || []).forEach(function (o) { ol.appendChild(el('li', null, o.title)); });
    s3.appendChild(ol);
    p.appendChild(s3);

    // components
    var comps = trace.components || {};
    var s4 = el('section');
    s4.appendChild(el('h2', null, 'Components'));
    var dl = el('dl', 'comps');
    Object.keys(comps).forEach(function (k) {
      dl.appendChild(el('dt', null, k));
      var dd = el('dd', null, comps[k]);
      dl.appendChild(dd);
    });
    s4.appendChild(dl);
    p.appendChild(s4);
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
