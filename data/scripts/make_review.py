import json
O = "/home/claude/data/out"
L = lambda p: [json.loads(l) for l in open(p)]
ext = [{"t": r["text"], "s": {k: v[0] for k, v in r["slots"].items() if v != "neutral"}, "r": r["reason"][0],
        "k": r["kind"], "sp": r["split"], "id": r["seed_id"], "c": r.get("context", ""), "g": r.get("strength_label", {})} for r in L(f"{O}/extractor.jsonl")]
prof = L(f"{O}/profiles.jsonl") + L(f"{O}/dev_profiles.jsonl")
nli = L(f"{O}/nli_pairs.jsonl")
bank = json.load(open(f"{O}/question_bank.json"))
import csv
titles = {r["code"]: r["Title"] for r in csv.DictReader(open(f"{O}/occupations.csv"))}
for p in prof: p["title"] = titles.get(p["target"], "")
DATA = json.dumps({"ext": ext, "prof": prof, "nli": nli, "bank": bank}).replace("</", "<\\/")

html = r'''<title>Career Matcher Data Summary</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a reviewer's workbench — summary ledger on top, one tab per data file, dense filterable rows */
:root{
  --bg:#f6f7f4; --panel:#ffffff; --fg:#1d2321; --muted:#5d6762; --rule:#dde2dd;
  --accent:#2f6b5a; --accent-soft:#e2eee9; --pos:#2f7d4f; --pos-soft:#e3f2e8; --neg:#b2443a; --neg-soft:#f7e4e1; --warn:#9a6a12; --warn-soft:#f6edd9;
  --display:"Fraunces",Georgia,serif; --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#141917; --panel:#1b2220; --fg:#e4e9e6; --muted:#98a39e; --rule:#2c3532;
  --accent:#7cc2ab; --accent-soft:#1f3530; --pos:#78c896; --pos-soft:#1c3326; --neg:#ec8f84; --neg-soft:#3a2220; --warn:#e3b45c; --warn-soft:#352b17; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#141917; --panel:#1b2220; --fg:#e4e9e6; --muted:#98a39e; --rule:#2c3532;
  --accent:#7cc2ab; --accent-soft:#1f3530; --pos:#78c896; --pos-soft:#1c3326; --neg:#ec8f84; --neg-soft:#3a2220; --warn:#e3b45c; --warn-soft:#352b17; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--body)}
.wrap{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
h1{font:600 clamp(26px,4vw,34px)/1.15 var(--display);margin:0 0 6px;text-wrap:balance}
.lede{color:var(--muted);max-width:68ch;margin:0 0 22px}
.ledger{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:8px;overflow:hidden;margin-bottom:22px}
.ledger div{background:var(--panel);padding:12px 14px}
.ledger b{display:block;font:500 22px/1.2 var(--mono);font-variant-numeric:tabular-nums}
.ledger span{color:var(--muted);font-size:12.5px}
h2{font:600 19px/1.3 var(--display);margin:26px 0 10px}
.checks{display:grid;gap:8px;margin:0;padding:0;list-style:none}
.checks li{display:flex;gap:10px;align-items:baseline;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:9px 12px}
.checks li>div{min-width:0}
.tag{font:500 11px/1 var(--mono);letter-spacing:.04em;text-transform:uppercase;padding:4px 7px;border-radius:4px;white-space:nowrap}
.ok{background:var(--pos-soft);color:var(--pos)} .fix{background:var(--accent-soft);color:var(--accent)} .look{background:var(--warn-soft);color:var(--warn)}
.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--rule);margin:30px 0 14px;position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);z-index:2;padding-top:8px}
.tabs button{font:500 14px var(--body);background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);padding:8px 12px;cursor:pointer}
.tabs button[aria-selected="true"]{color:var(--fg);border-color:var(--accent)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center}
select,input[type=search]{font:14px var(--body);background:var(--panel);color:var(--fg);border:1px solid var(--rule);border-radius:6px;padding:6px 8px}
input[type=search]{flex:1;min-width:160px}
.count{color:var(--muted);font:13px var(--mono)}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font:500 12px var(--body);color:var(--muted);letter-spacing:.03em;text-transform:uppercase;padding:9px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td.m{font:12.5px var(--mono);color:var(--muted);white-space:nowrap}
.chip{display:inline-block;font:500 12px/1 var(--mono);padding:3px 6px;border-radius:4px;margin:0 3px 3px 0}
.chip.p{background:var(--pos-soft);color:var(--pos)} .chip.n{background:var(--neg-soft);color:var(--neg)} .chip.z{background:var(--rule);color:var(--muted)}
.more{margin-top:10px}
.more button{font:500 14px var(--body);background:var(--panel);color:var(--accent);border:1px solid var(--rule);border-radius:6px;padding:7px 12px;cursor:pointer}
.cards{display:grid;gap:12px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:16px}
@media (max-width:700px){.card{grid-template-columns:minmax(0,1fr)}}
.card h3{font:600 15px var(--body);margin:0 0 4px;display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}
.card .sub{color:var(--muted);font-size:13px;margin-bottom:8px}
.card p{margin:0 0 8px}
.hl{background:var(--warn-soft);border-radius:3px;padding:0 2px}
.bars{display:grid;gap:5px;font:12.5px var(--mono);font-variant-numeric:tabular-nums}
.bar{display:grid;grid-template-columns:16px 1fr 46px;align-items:center;gap:6px}
.track{position:relative;height:10px;background:var(--bg);border-radius:2px}
.track::after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.fill{position:absolute;top:0;bottom:0;border-radius:2px}
details{margin-top:6px;font-size:13.5px}
summary{cursor:pointer;color:var(--accent)}
dl.res{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:8px 0 0}
dl.res dt{font:12.5px var(--mono);color:var(--muted)} dl.res dd{margin:0}
.q{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 16px;margin-bottom:10px}
.q h3{font:600 14px var(--body);margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.q ol{margin:0;padding-left:20px} .q li{margin-bottom:6px}
.ans{font-size:13.5px;color:var(--muted)}
.glance{max-width:75ch;padding-left:20px;display:grid;gap:6px;margin:0}
.note2{color:var(--muted);font-size:13.5px;max-width:75ch;margin:8px 0 0}
h3{font:600 15px var(--body);margin:18px 0 8px}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
</style>
<div class="wrap">
<h1>Career Matcher Data Summary</h1>
<p class="lede">The data for the CS425 career matcher (9 October): what is there, whether it is enough to train, and how it was checked. Browse every row at the bottom.</p>
<div class="ledger" id="ledger"></div>

__SUMMARY__
<h2>Browse the data</h2>
<div class="tabs" role="tablist">
<button role="tab" id="t-ext" aria-selected="true">Extractor</button>
<button role="tab" id="t-prof" aria-selected="false">Test profiles</button>
<button role="tab" id="t-nli" aria-selected="false">NLI pairs</button>
<button role="tab" id="t-bank" aria-selected="false">Question bank</button>
</div>

<section id="p-ext">
<div class="filters">
<input type="search" id="f-q" placeholder="Search sentences" aria-label="Search sentences">
<select id="f-k" aria-label="Kind"><option value="">All kinds</option><option>seed</option><option>rephrase</option><option>negation</option><option>frame</option><option>combo</option><option>qa</option></select>
<select id="f-t" aria-label="Type"><option value="">All types</option><option value="R">R Realistic</option><option value="I">I Investigative</option><option value="A">A Artistic</option><option value="S">S Social</option><option value="E">E Enterprising</option><option value="C">C Conventional</option><option value="-">All neutral</option></select>
<select id="f-p" aria-label="Polarity"><option value="">Any polarity</option><option value="p">positive</option><option value="n">negative</option></select>
<select id="f-r" aria-label="Reason"><option value="">All reasons</option><option value="e">enjoyment</option><option value="f">familiarity</option><option value="n">neither</option></select>
<select id="f-s" aria-label="Split"><option value="">All splits</option><option>train</option><option>val</option><option>test</option></select>
<span class="count" id="f-c"></span>
</div>
<div class="tw"><table><thead><tr><th>Sentence</th><th>Labels</th><th>Reason</th><th>Kind</th><th>Split</th><th>Seed</th></tr></thead><tbody id="ext-body"></tbody></table></div>
<div class="more"><button id="ext-more">Show 100 more</button></div>
</section>

<section id="p-prof" hidden>
<div class="filters">
<select id="g-t" aria-label="Tier"><option value="">All profiles</option><option value="1">Tier 1 · clean</option><option value="2">Tier 2 · planted</option><option value="D">Dev set only</option></select>
<input type="search" id="g-q" placeholder="Search text or ID" aria-label="Search profiles">
<span class="count" id="g-c"></span>
</div>
<div class="cards" id="prof-body"></div>
</section>

<section id="p-nli" hidden>
<div class="filters">
<select id="h-l" aria-label="Label"><option value="">All labels</option><option>contradiction</option><option>elaboration</option><option>unrelated</option></select>
<select id="h-s" aria-label="Split"><option value="">Dev and test</option><option>dev</option><option>test</option></select>
<span class="count" id="h-c"></span>
</div>
<div class="tw"><table><thead><tr><th>ID</th><th>Statement a</th><th>Statement b</th><th>Slot</th><th>Label</th><th>Split</th></tr></thead><tbody id="nli-body"></tbody></table></div>
</section>

<section id="p-bank" hidden><div id="bank-body"></div></section>
</div>
<script>
const D = __DATA__;
const T = {R:"Realistic",I:"Investigative",A:"Artistic",S:"Social",E:"Enterprising",C:"Conventional"};
const RN = {e:"enjoyment",f:"familiarity",n:"neither"};
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const $ = id => document.getElementById(id);

// ledger
const cnt = (a, f) => a.filter(f).length;
const L = [
  [D.ext.length.toLocaleString(), "extractor sentences"],
  [cnt(D.ext, x => x.k === "seed"), "seeds"],
  [cnt(D.ext, x => x.sp === "train") + " / " + cnt(D.ext, x => x.sp === "val") + " / " + cnt(D.ext, x => x.sp === "test"), "train / val / test"],
  [cnt(D.prof, p => p.id[0] === "P"), "locked test profiles"],
  [cnt(D.prof, p => p.id[0] === "D"), "dev profiles"],
  [D.nli.length, "NLI pairs"],
  ["0.82", "hit@5 ceiling (perfect reader)"],
];
$("ledger").innerHTML = L.map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");

// tabs
const tabs = ["ext", "prof", "nli", "bank"];
function show(t){ tabs.forEach(x => { $("t-" + x).setAttribute("aria-selected", x === t); $("p-" + x).hidden = x !== t; });
  try { localStorage.setItem("tab", t) } catch(e) {} }
tabs.forEach(t => $("t-" + t).onclick = () => show(t));
try { const s = localStorage.getItem("tab"); if (tabs.includes(s)) show(s) } catch(e) {}

// extractor
let limit = 100;
const chips = (s, g) => { const k = Object.keys(s); return k.length ? k.map(t => `<span class="chip ${s[t]}" title="${T[t]} ${s[t]==="p"?"positive":"negative"}">${t} ${s[t]==="p"?"+":"−"}${g && g[t] && g[t]!=="strong" ? " " + (g[t]==="mild" ? "mild" : "don't mind") : ""}</span>`).join("") : `<span class="chip z">all neutral</span>` };
function renderExt(){
  const q = $("f-q").value.toLowerCase(), k = $("f-k").value, t = $("f-t").value, p = $("f-p").value, r = $("f-r").value, sp = $("f-s").value;
  const rows = D.ext.filter(x => (!q || x.t.toLowerCase().includes(q)) && (!k || x.k === k) && (!r || x.r === r) && (!sp || x.sp === sp)
    && (!t || (t === "-" ? !Object.keys(x.s).length : x.s[t] && (!p || x.s[t] === p))) && (t && t !== "-" || !p || Object.values(x.s).includes(p)));
  $("f-c").textContent = rows.length.toLocaleString() + " rows";
  $("ext-body").innerHTML = rows.slice(0, limit).map(x => `<tr><td>${x.c ? `<div class="ans">Q: ${esc(x.c)}</div>` : ""}${esc(x.t)}</td><td>${chips(x.s, x.g)}</td><td class="m">${RN[x.r]}</td><td class="m">${x.k}</td><td class="m">${x.sp}</td><td class="m">${esc(x.id)}</td></tr>`).join("");
  $("ext-more").parentElement.hidden = rows.length <= limit;
}
["f-q","f-k","f-t","f-p","f-r","f-s"].forEach(id => $(id).addEventListener("input", () => { limit = 100; renderExt() }));
$("ext-more").onclick = () => { limit += 100; renderExt() };
renderExt();

// profiles
const bar = (t, v) => { const w = Math.min(Math.abs(v), 2.5) / 2.5 * 50; const pos = v >= 0;
  return `<div class="bar"><span>${t}</span><div class="track"><div class="fill" style="${pos ? "left:50%" : "right:50%"};width:${w}%;background:var(${pos ? "--pos" : "--neg"})"></div></div><span style="text-align:right">${v > 0 ? "+" : ""}${v.toFixed(2)}</span></div>` };
function hl(p){ let t = esc(p.text); if (p.contradiction) [p.contradiction.statement_a, p.contradiction.statement_b].forEach(s => t = t.replace(esc(s), `<span class="hl">${esc(s)}</span>`)); return t }
function renderProf(){
  const f = $("g-t").value, q = $("g-q").value.toLowerCase();
  const ps = D.prof.filter(p => (!f || (f === "D" ? p.id[0] === "D" : String(p.tier) === f)) && (!q || p.text.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)));
  $("g-c").textContent = ps.length + " profiles";
  $("prof-body").innerHTML = ps.map(p => `<article class="card"><div>
    <h3>${p.id}<span class="tag ${p.tier === 2 ? "look" : "ok"}">Tier ${p.tier}</span>${p.planted.map(x => `<span class="chip z">${esc(x)}</span>`).join("")}</h3>
    <div class="sub">Target ${p.target} · ${esc(p.title)}</div>
    <p>${hl(p)}</p>
    ${p.contradiction ? `<div class="ans">Genuine side of the contradiction: statement ${p.contradiction.genuine}</div>` : ""}
    ${p.latent_reveal ? `<div class="ans">Latent reveal: “${esc(p.latent_reveal)}”</div>` : ""}
    <details><summary>Resolutions (simulated user's answer to a CHALLENGE)</summary><dl class="res">${Object.entries(p.resolutions).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl></details>
  </div><div class="bars" aria-label="Hidden RIASEC z-scores">${"RIASEC".split("").map(t => bar(t, p.hidden_slots[t])).join("")}</div></article>`).join("");
}
["g-t","g-q"].forEach(id => $(id).addEventListener("input", renderProf));
renderProf();

// nli
function renderNli(){
  const l = $("h-l").value, s = $("h-s").value;
  const r = D.nli.filter(x => (!l || x.label === l) && (!s || x.split === s));
  $("h-c").textContent = r.length + " pairs";
  $("nli-body").innerHTML = r.map(x => `<tr><td class="m">${x.id}</td><td>${esc(x.a)}</td><td>${esc(x.b)}</td><td class="m">${x.slot}</td><td class="m">${x.label}</td><td class="m">${x.split}</td></tr>`).join("");
}
["h-l","h-s"].forEach(id => $(id).addEventListener("input", renderNli));
renderNli();

// bank
const B = D.bank;
const list = a => `<ol>${a.map(x => `<li>${esc(x)}</li>`).join("")}</ol>`;
$("bank-body").innerHTML =
  `<div class="q"><h3>Opening</h3>${list(B.OPENING)}</div>` +
  Object.entries(B.CLARIFY).map(([t, qs]) => `<div class="q"><h3>CLARIFY · ${t}</h3><ol>${qs.map((q, i) => `<li>${esc(q)}<div class="ans">strong yes: ${esc(B.ANSWERS[t][i].pos)}<br>mild yes: ${esc(B.ANSWERS[t][i].pos_mild||"")}<br>don&rsquo;t mind: ${esc(B.ANSWERS[t][i].dontmind||"")}<br>mild no: ${esc(B.ANSWERS[t][i].neg_mild||"")}<br>strong no: ${esc(B.ANSWERS[t][i].neg)}</div></li>`).join("")}</ol></div>`).join("") +
  ["CHALLENGE", "PROBE", "ABSTAIN", "RECOMMEND"].map(k => `<div class="q"><h3>${k}</h3>${list(B[k])}</div>`).join("");
</script>
'''
open("/mnt/user-data/outputs/career-matcher-data-review.html", "w").write(html.replace("__DATA__", DATA).replace("__SUMMARY__", open("/home/claude/data/work/summary_block.html").read()))
print("ok")
