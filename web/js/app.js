/* ============================================================
   Uber_Support AI agent - dashboard controller (pure JS)
   Fully local: data views mirror the committed repo data, and
   Live Chat talks to the real agent over POST /api/chat.
   ============================================================ */
(function () {
  "use strict";

  var DATA = window.WEB_DATA || {};
  var SYSTEMS = {
    our_agent: "Our Agent",
    simple: "Keyword + RAG",
    trivial: "Trivial"
  };
  var SYS_KEYS = ["our_agent", "simple", "trivial"];
  var VIEWS = ["dashboard", "evaluation", "golden", "chat", "files"];
  var INTENT_LABELS = DATA.intent_labels || {};

  var ACTIVE_SYS = "our_agent";
  var ACTIVE_VIEW = "dashboard";
  var GOLDEN_FILTER = { q: "", show: "all" };
  var CHAT_STARTED = false;

  /* ---------------- helpers ---------------- */
  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function pct(x) { return (Math.round(x * 1000) / 10) + "%"; }
  function trunc(s, n) {
    s = String(s == null ? "" : s);
    return s.length > n ? s.slice(0, n) + "\u2026" : s;
  }
  function pill(t, kind) {
    var cls = { ok: "pill pill--ok", bad: "pill pill--bad", warn: "pill pill--warn", info: "pill pill--info" }[kind] || "pill pill--info";
    return '<span class="' + cls + '">' + esc(t) + "</span>";
  }
  function intentLabel(k) { return esc(INTENT_LABELS[k] || k || "?"); }
  function escPillLabel(k) { return pill(intentLabel(k), "info"); }
  function routePill(escalate, route) {
    if (escalate) return pill("HUMAN AGENT", route === "admin" ? "warn" : "bad");
    return pill("AUTO-HANDLED", "ok");
  }
  function sysMetrics(sys) {
    var m = (DATA.metrics || {})[sys];
    if (!m) return { intent: 0, esc: { precision: 0, recall: 0, f1: 0 }, gr: { mean: 0, pct_ge_0_7: 0 } };
    var escM = m.escalation || {};
    var gr = m.reply_grounding || {};
    return {
      intent: m.intent_accuracy || 0,
      esc: { precision: escM.precision || 0, recall: escM.recall || 0, f1: escM.f1 || 0 },
      gr: { mean: gr.mean || 0, pct_ge_0_7: gr.pct_ge_0_7 || 0 }
    };
  }

  /* ---------------- model toggle ---------------- */
  function renderModelToggle() {
    var el = $("dash-model-toggle");
    if (!el) return;
    el.innerHTML = SYS_KEYS.map(function (k) {
      var on = k === ACTIVE_SYS;
      return '<label class="model-toggle__label' + (on ? " model-toggle__label--on" : "") + '">' +
        '<input type="radio" name="model" value="' + k + '"' + (on ? " checked" : "") + " />" +
        "<span>" + esc(SYSTEMS[k]) + "</span></label>";
    }).join("");
    el.querySelectorAll("input").forEach(function (inp) {
      inp.addEventListener("change", function () {
        ACTIVE_SYS = this.value;
        renderAll();
      });
    });
  }

  /* ---------------- view switching ---------------- */
  function showView(name) {
    if (VIEWS.indexOf(name) === -1) name = "dashboard";
    ACTIVE_VIEW = name;
    VIEWS.forEach(function (v) {
      var el = $("view-" + v);
      if (el) el.classList.toggle("view--active", v === name);
    });
    document.querySelectorAll(".nav__link").forEach(function (a) {
      a.classList.toggle("nav__link--active", a.dataset.tab === name);
    });
    closeSidebar();
    render();
    if (name === "chat") focusChat();
  }

  function closeSidebar() {
    $("sidebar").classList.remove("sidebar--open");
    $("overlay").classList.remove("overlay--visible");
  }

  /* ============================================================
     DASHBOARD
     ============================================================ */
  function renderDashboard() {
    var opt = $("view-dashboard");
    var m = sysMetrics(ACTIVE_SYS);
    opt.innerHTML =
      '<section class="cards">' +
      '<div class="card card--accent"><span class="card__label">Intent accuracy</span>' +
      '<span class="card__value">' + pct(m.intent) + '</span>' +
      '<span class="card__sub">golden set: ' + (DATA.golden || []).length + " hand-labeled examples</span></div>" +
      '<div class="card"><span class="card__label">Escalation F1</span>' +
      '<span class="card__value">' + m.esc.f1.toFixed(2) + "</span>" +
      '<span class="card__sub">precision ' + m.esc.precision.toFixed(2) +
      " / recall " + m.esc.recall.toFixed(2) + "</span></div>" +
      '<div class="card"><span class="card__label">Reply grounding</span>' +
      '<span class="card__value">' + m.gr.mean.toFixed(3) + "</span>" +
      '<span class="card__sub">' + pct(m.gr.pct_ge_0_7) + " \u2265 0.7 similarity to real Uber replies</span></div>" +
      "</section>" +

      '<section class="panel"><h2 class="panel__title">How the agent works</h2>' +
      '<div class="pipeline">' +
      [
        ["1", "Understand", "A sentence-embedding model (mxbai-embed-large, local) turns the customer text into a 1024-dim vector - meaning in, numbers out."],
        ["2", "Classify intent", "Nearest of 10 intents among 25 hand-picked anchors, with an explainable keyword fallback. Non-Uber requests are caught first and routed to a human."],
        ["3", "Draft reply", "RAG retrieval returns the real Uber reply that resolved the most similar historical request - grounded, retrieved, never hallucinated."],
        ["4", "Decide escalation", "12 word-boundary safety / fraud / legal rules decide auto-handle vs human - always with a stated reason."]
      ].map(function (s) {
        return '<div class="step"><span class="step__n">' + s[0] + "</span>" +
          "<div><h3>" + esc(s[1]) + "</h3><p>" + esc(s[2]) + "</p></div></div>";
      }).join("") +
      "</div></section>" +

      '<section class="panel"><h2 class="panel__title">The 10 intents</h2>' +
      '<div class="chips">' +
      (DATA.intents || []).map(function (i) {
        return '<span class="chip">' + intentLabel(i) + "</span>";
      }).join("") +
      "</div></section>";

    $("agent-ready").textContent = "AGENT READY";
    $("emp-data-note").textContent = "all local \u00b7 no API keys";
  }

  /* ============================================================
     EVALUATION
     ============================================================ */
  function renderEvaluation() {
    var opt = $("view-evaluation");
    var m = sysMetrics(ACTIVE_SYS);
    var rows = SYS_KEYS.map(function (k) {
      var mm = sysMetrics(k);
      var nm = k === ACTIVE_SYS ? "<strong>" + esc(SYSTEMS[k]) + "</strong>" : esc(SYSTEMS[k]);
      return "<tr" + (k === ACTIVE_SYS ? ' class="row--on"' : "") + "><td>" + nm + "</td><td>" +
        pct(mm.intent) + "</td><td>" + mm.esc.precision.toFixed(2) + "</td><td>" +
        mm.esc.recall.toFixed(2) + "</td><td><b>" + mm.esc.f1.toFixed(2) + "</b></td><td>" +
        mm.gr.mean.toFixed(3) + "</td></tr>";
    }).join("");

    var perRows = ((DATA.per_intent || {})[ACTIVE_SYS] || []).map(function (b) {
      return '<div class="per-intent"><span class="per-intent__label">' + intentLabel(b.intent) + "</span>" +
        '<span class="per-intent__bar"><span style="width:' + Math.max(3, Math.round(b.accuracy * 100)) + '%' +
        (b.accuracy < 0.5 ? ' class="bar--low"' : "") + '"></span></span>' +
        '<span class="per-intent__pct">' + b.correct + "/" + b.total + " \u00b7 " + pct(b.accuracy) + "</span></div>";
    }).join("");

    opt.innerHTML =
      '<section class="cards">' +
      '<div class="card card--accent"><span class="card__label">Intent accuracy</span>' +
      '<span class="card__value">' + pct(m.intent) + "</span>" +
      '<span class="card__sub">' + esc(SYSTEMS[ACTIVE_SYS]) + " on the golden set</span></div>" +
      '<div class="card"><span class="card__label">Escalation P / R / F1</span>' +
      '<span class="card__value">' + m.esc.precision.toFixed(2) + " / " + m.esc.recall.toFixed(2) + " / " + m.esc.f1.toFixed(2) + "</span>" +
      '<span class="card__sub">recall is what matters: never miss a real human case</span></div>' +
      '<div class="card"><span class="card__label">Reply grounding</span>' +
      '<span class="card__value">' + m.gr.mean.toFixed(3) + "</span>" +
      '<span class="card__sub">mean cosine similarity vs real historical Uber replies</span></div>' +
      "</section>" +

      '<section class="panel"><h2 class="panel__title">Systems compared</h2>' +
      '<div class="table-wrap"><table class="tbl">' +
      "<thead><tr><th>System</th><th>Intent acc</th><th>Esc P</th><th>Esc R</th><th>Esc F1</th><th>Grounding</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table></div></section>" +

      '<section class="panel"><h2 class="panel__title">Per-intent accuracy \u00b7 ' + esc(SYSTEMS[ACTIVE_SYS]) + "</h2>" +
      '<div class="per-intent__list">' + (perRows || '<p class="muted">No data.</p>') + "</div></section>";
  }

  /* ============================================================
     GOLDEN SET
     ============================================================ */
  function goldenItems() {
    var all = (DATA.golden || []).slice();
    var f = GOLDEN_FILTER;
    if (f.show === "esc") all = all.filter(function (g) { return g.escalation; });
    if (f.show === "auto") all = all.filter(function (g) { return !g.escalation; });
    if (f.q) {
      var q = f.q.toLowerCase();
      all = all.filter(function (g) {
        return (g.message + " " + g.agent_reply + " " + (g.escalation_reason || "")).toLowerCase().indexOf(q) !== -1;
      });
    }
    return all;
  }

  function renderGolden() {
    var opt = $("view-golden");
    var items = goldenItems();
    var tabs = [["all", "All"], ["esc", "Escalated"], ["auto", "Auto-handled"]].map(function (t) {
      var on = GOLDEN_FILTER.show === t[0];
      return '<button class="tb' + (on ? " tb--on" : "") + '" data-f="' + t[0] + '">' + t[1] + "</button>";
    }).join("");

    var list = items.map(function (g, i) {
      var escTxt = g.escalation
        ? '<div class="gb__esc">' + pill("ESCALATED \u00b7 " + (g.escalation_reason || "human needed"), "warn") + "</div>"
        : '<div class="gb__esc">' + pill("STANDARD SUPPORT", "ok") + "</div>";
      return '<article class="gb">' +
        '<div class="gb__head">' +
        '<span class="gb__idx">' + String(i + 1).padStart(3, "0") + "</span>" +
        escPillLabel(g.intent) +
        '<span class="gb__route">' + routePill(g.escalation) + "</span>" +
        "</div>" +
        '<p class="gb__msg">' + esc(g.message) + "</p>" +
        '<div class="gb__reply"><span class="gb__reply-label">Uber replied</span><p>' + esc(g.agent_reply) + "</p></div>" +
        escTxt +
        "</article>";
    }).join("");

    opt.innerHTML =
      '<section class="panel">' +
      '<div class="gb__top">' +
      "<div>" + '<h2 class="panel__title">Golden set</h2>' +
      '<p class="muted">' + (DATA.golden || []).length + " hand-labeled dialogues, the ground truth the metrics are measured against.</p></div>" +
      '<input class="gb__search" type="search" placeholder="Search messages\u2026" value="' + esc(GOLDEN_FILTER.q) + '" />' +
      "</div>" +
      '<div class="tb__row">' + tabs + "</div>" +
      '<div class="gb__list">' + (list || '<p class="muted">No matches.</p>') + "</div>" +
      "</section>";

    var search = opt.querySelector(".gb__search");
    search.addEventListener("input", function () {
      GOLDEN_FILTER.q = this.value;
      renderGolden();
      var el = $("view-golden");
      var s = el.querySelector(".gb__search");
      if (s) { s.value = GOLDEN_FILTER.q; s.focus(); }
    });
    opt.querySelectorAll(".tb").forEach(function (b) {
      b.addEventListener("click", function () {
        GOLDEN_FILTER.show = this.dataset.f;
        renderGolden();
        var el = $("view-golden");
        var s = el.querySelector(".gb__search");
        if (s) s.value = GOLDEN_FILTER.q;
      });
    });
  }

  /* ============================================================
     LIVE CHAT
     ============================================================ */
  function chatLog() {
    var el = $("chat-log");
    el = el || $("view-chat").querySelector(".chat__log");
    return el;
  }
  function focusChat() {
    var inp = $("chat-input");
    if (inp) inp.focus();
  }

  function addMsg(kind, html) {
    var log = chatLog();
    var div = document.createElement("div");
    div.className = "chat__msg chat__msg--" + kind;
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function sendMessage(text) {
    var msg = String(text || "").trim();
    if (!msg) return;
    if (!CHAT_STARTED) { chatLog().innerHTML = ""; CHAT_STARTED = true; }

    addMsg("user", '<div class="chat__bubble">' + esc(msg) + "</div>");
    var input = $("chat-input");
    var btn = $("chat-send");
    if (input) input.value = "";
    if (btn) btn.disabled = true;

    var pending = document.createElement("div");
    pending.className = "chat__msg chat__msg--agent";
    pending.innerHTML = '<div class="chat__bubble chat__bubble--pending"><span class="spinner"></span> thinking\u2026</div>';
    chatLog().appendChild(pending);
    chatLog().scrollTop = chatLog().scrollHeight;

    fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    }).then(function (res) {
      return res.json().then(function (j) { return { ok: res.ok, j: j }; });
    }).then(function (r) {
      if (r.ok) renderAgentReply(r.j); else renderAgentError(r.j);
    }).catch(function () {
      renderAgentError({ error: "Could not reach the agent server. Is web_server.py running?" });
    }).finally(function () {
      pending.remove();
      if (btn) btn.disabled = false;
    });
  }

  function renderAgentReply(p) {
    var oos = !!p.out_of_scope;
    var escTrue = !!p.escalate;
    var head = oos
      ? escPillLabel("out_of_scope") + ' <span class="chat__meta-sep"></span>' + routePill(true, "admin")
      : escPillLabel(p.intent) + ' <span class="chat__meta-sep"></span>' + routePill(escTrue, p.route);

    var reasonHtml = oos
      ? '<div class="chat__reason">' + pill(p.escalation_reason || "Non-Uber request routed to human", "warn") + "</div>"
      : escTrue
        ? '<div class="chat__reason">' + pill(esc(p.escalation_reason || "Escalated to a human agent"), "warn") + "</div>"
        : '<div class="chat__reason">' + pill("Auto-handled \u00b7 routine resolution", "ok") + "</div>";

    var simBar = "";
    var sim = parseFloat(p.reply_source_sim) || 0;
    if (p.reply && !oos) {
      simBar = '<div class="chat__sim"><span style="width:' + Math.max(4, Math.round(sim * 100)) + '%"></span>' +
        '<em>' + sim.toFixed(3) + " grounding</em></div>";
    }

    var replyHtml = oos || !p.reply
      ? '<div class="chat__bubble chat__bubble--oos">' + esc(p.reply || "No reply generated.") + "</div>"
      : '<div class="chat__bubble chat__bubble--agent">' + esc(p.reply) + "</div>" + simBar;

    addMsg("agent",
      '<div class="chat__meta">' + head + "</div>" +
      replyHtml +
      reasonHtml +
      '<div class="chat__note">retrieved from real Uber_Support history \u00b7 not generated</div>');
  }

  function renderAgentError(j) {
    addMsg("agent", '<div class="chat__bubble chat__bubble--error">' + esc((j && j.error) || "Agent error.") + "</div>");
  }

  function renderChat() {
    var opt = $("view-chat");
    var samples = (DATA.sample_questions || []).map(function (q, i) {
      return '<button class="chip chip--click' + (i === 0 ? " chip--on" : "") + '" data-q="' + i + '">' + esc(q) + "</button>";
    }).join("");

    opt.innerHTML =
      '<section class="chat">' +
      '<div class="chat__head"><div><h2 class="panel__title">Live chat with the agent</h2>' +
      '<p class="muted">Type anything a customer might tweet @Uber_Support - the real agent pipeline replies.</p></div>' +
      '<span class="tag tag--live" id="chat-status">' + (navigator.onLine ? "ONLINE" : "OFFLINE") + "</span></div>" +
      '<div class="chat__samples">' + samples + "</div>" +
      '<div class="chat__log" id="chat-log">' +
      '<div class="chat__welcome"><span class="chat__welcome-caret">@Uber_Support</span>' +
      "<p>Hi, I'm the Uber_Support AI assistant. I classify your intent, draft a grounded reply from real Uber history, and tell you if a human should step in.</p>" +
      '<p class="muted">Try one of the sample messages above, or type your own.</p></div>' +
      "</div>" +
      '<div class="chat__input">' +
      '<input id="chat-input" type="text" placeholder="Type a customer message\u2026" autocomplete="off" />' +
      '<button id="chat-send">Send</button></div>' +
      "</section>";

    var input = $("chat-input");
    var send = $("chat-send");
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") sendMessage(input.value);
    });
    send.addEventListener("click", function () { sendMessage(input.value); });

    $("view-chat").querySelectorAll(".chip--click").forEach(function (c) {
      c.addEventListener("click", function () {
        sendMessage(DATA.sample_questions[parseInt(this.dataset.q, 10)]);
      });
    });
  }

  /* ============================================================
     FILES
     ============================================================ */
  function renderFiles() {
    var opt = $("view-files");
    var rows = (DATA.files || []).map(function (f) {
      var folder = f.path.split("/")[0];
      var fg = f.path.indexOf("/") !== -1 ? folder : "root";
      return '<button class="file" data-path="' + esc(f.path) + '">' +
        '<span class="file__icon">' + esc(fg === "root" ? "fi" : fg.charAt(0)) + "</span>" +
        '<span class="file__body"><span class="file__name">' + esc(f.path) + '</span>' +
        '<span class="file__note">' + esc(f.note) + "</span></span>" +
        '<span class="file__size">' + esc(f.size) + "</span></button>";
    }).join("");

    opt.innerHTML =
      '<section class="panel">' +
      '<div class="gb__top"><div><h2 class="panel__title">Data &amp; files</h2>' +
      '<p class="muted">Everything the dashboard and the agent run from - click a file for its purpose.</p></div></div>' +
      '<div class="file__list">' + rows + "</div></section>";

    opt.querySelectorAll(".file").forEach(function (b) {
      b.addEventListener("click", function () { openFile(this.dataset.path); });
    });
  }

  function openFile(path) {
    var f = (DATA.files || []).filter(function (x) { return x.path === path; })[0];
    if (!f) return;
    $("modal-title").textContent = f.path;
    $("modal-body").innerHTML =
      '<p class="muted">' + esc(f.size) + "</p>" +
      "<p>" + esc(f.note) + "</p>" +
      '<p class="muted">See README.md \u00a7 9 for the full file-by-file guide.</p>';
    $("modal").classList.add("modal--open");
    $("overlay").classList.add("overlay--visible");
  }

  /* ---------------- top-level render ---------------- */
  function render() {
    var active = ACTIVE_VIEW;
    var suppress = { dashboard: [renderDashboard], evaluation: [renderEvaluation], golden: [renderGolden], chat: [renderChat], files: [renderFiles] };
    (suppress[active] || [function () {}])[0]();
    renderModelToggle();
  }

  function renderAll() { render(); }

  /* ---------------- nav / static wiring ---------------- */
  function wireNav() {
    document.querySelectorAll(".nav__link").forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        showView(this.dataset.tab || "dashboard");
      });
    });
    $("burger").addEventListener("click", function () {
      var sb = $("sidebar"), ov = $("overlay");
      sb.classList.toggle("sidebar--open");
      ov.classList.toggle("overlay--visible");
    });
    $("overlay").addEventListener("click", closeSidebar);
    $("modal-close").addEventListener("click", function () {
      $("modal").classList.remove("modal--open");
      closeSidebar();
    });
    window.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { $("modal").classList.remove("modal--open"); closeSidebar(); }
    });
  }

  function boot() {
    var msg = $("emp-data-note");
    if (!(DATA.metrics && DATA.golden && DATA.predictions)) {
      if (msg) msg.textContent = "data.js is stale - run web/tools/generate_web_data.py";
    }
    wireNav();
    renderAll();
    $("agent-ready").textContent = "AGENT READY";
  }

  boot();
})();