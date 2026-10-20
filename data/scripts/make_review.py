import json, csv
O = "/home/claude/data/out"
L = lambda p: [json.loads(l) for l in open(p)]
def jl(name):
    rows = L(f"{O}/{name}"); cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    return {"kind": "table", "cols": cols, "rows": [[r.get(c) for c in cols] for r in rows]}
def cs(name):
    with open(f"{O}/{name}", newline="") as f:
        rd = csv.reader(f); cols = next(rd); rows = list(rd)
    return {"kind": "table", "cols": cols, "rows": rows, "csv": True}
def js(name):
    return {"kind": "json", "text": open(f"{O}/{name}").read()}
FILES = [
 ("extractor.jsonl", jl), ("extractor_stress.jsonl", jl), ("profiles.jsonl", jl), ("profiles_v4.jsonl", jl), ("dev_profiles.jsonl", jl), ("nli_pairs.jsonl", jl),
 ("question_bank.json", js), ("labels.json", js), ("occupations.csv", cs), ("tasks.csv", cs),
 ("related_occupations.csv", cs), ("riasec_lexicon.json", js), ("activities.jsonl", jl),
 ("synthetic_seeds.jsonl", jl), ("nli_pairs_blind.jsonl", jl),
]
FD = {}
for n, f in FILES:
    FD[n] = f(n)
ext = L(f"{O}/extractor.jsonl"); prof = L(f"{O}/profiles.jsonl"); dev = L(f"{O}/dev_profiles.jsonl"); nli = L(f"{O}/nli_pairs.jsonl")
LEDGER = [[f"{len(ext):,}", "extractor sentences"], [str(sum(1 for r in ext if r["kind"] == "seed")), "seeds"],
          [" / ".join(str(sum(1 for r in ext if r["split"] == s)) for s in ("train", "val", "test")), "train / val / test"],
          [str(len(prof)) + " + " + str(len(L(f"{O}/profiles_v4.jsonl"))), "locked test profiles (v3 + v4)"], [str(len(dev)), "dev profiles"], ["400", "stress-test sentences"], [str(len(nli)), "NLI pairs"],
          ["0.82", "hit@5 ceiling (perfect reader)"]]
DATA = json.dumps({"files": FD, "order": [n for n, _ in FILES], "ledger": LEDGER}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

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
.scroll{height:72vh;min-height:320px;overflow:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px;overscroll-behavior:contain}
table.raw{width:max-content;min-width:100%;font:12.5px/1.45 var(--mono)}
table.raw th{position:sticky;top:0;background:var(--panel);z-index:1;font:500 12px var(--mono);text-transform:none;letter-spacing:0;color:var(--fg);box-shadow:0 1px 0 var(--rule)}
table.raw td{max-width:520px;white-space:pre-wrap;word-break:break-word;color:var(--fg)}
table.raw td.obj{color:var(--accent)}
table.raw .rn{color:var(--muted);text-align:right;position:sticky;left:0;background:var(--panel)}
pre.raw{margin:0;padding:12px 14px;font:12.5px/1.5 var(--mono);white-space:pre-wrap;word-break:break-word}
</style>
<div class="wrap">
<h1>Career Matcher Data Summary</h1>
<p class="lede">The data for the CS425 career matcher (9 October): what is there, whether it is enough to train, and how it was checked. Browse every row at the bottom.</p>
<div class="ledger" id="ledger"></div>

__SUMMARY__
<h2>Browse the data</h2>
<p class="lede">Every file in the package, exactly as stored: one column per field, in file order, with nested values shown as JSON. Scroll inside the box; more rows load as you reach the bottom.</p>
<div class="tabs" role="tablist" id="tabs"></div>
<div class="filters">
<input type="search" id="q" placeholder="Search this file" aria-label="Search this file">
<select id="sp" aria-label="Split"></select>
<span class="count" id="cnt"></span>
</div>
<div class="scroll" id="box" tabindex="0" aria-label="File contents"></div>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("ledger").innerHTML = D.ledger.map(([b, s]) => `<div><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
const show = v => v === null || v === undefined ? "" : typeof v === "string" ? v : JSON.stringify(v);
let cur = D.order[0], rows = [], shown = 0;
const STEP = 150;
$("tabs").innerHTML = D.order.map(n => `<button role="tab" data-f="${n}">${n}</button>`).join("");
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => open_(b.dataset.f));
function open_(n){
  cur = n; try { localStorage.setItem("file", n) } catch(e) {}
  document.querySelectorAll("#tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.f === n));
  const f = D.files[n]; $("q").value = "";
  const si = f.kind === "table" ? f.cols.indexOf("split") : -1;
  const splits = si < 0 ? [] : [...new Set(f.rows.map(r => r[si]))];
  $("sp").hidden = !splits.length;
  $("sp").innerHTML = `<option value="">All splits</option>` + splits.map(s => `<option>${esc(s)}</option>`).join("");
  render();
}
function render(){
  const f = D.files[cur], q = $("q").value.toLowerCase(), box = $("box"); box.scrollTop = 0;
  if (f.kind === "json") {
    $("sp").hidden = true;
    const lines = f.text.split("\n"), keep = q ? lines.filter(l => l.toLowerCase().includes(q)) : lines;
    $("cnt").textContent = q ? `${keep.length} matching lines` : `${lines.length} lines`;
    box.innerHTML = `<pre class="raw">${esc(keep.join("\n"))}</pre>`; return;
  }
  const si = f.cols.indexOf("split"), sp = $("sp").value;
  rows = f.rows.filter(r => (!sp || r[si] === sp) && (!q || r.some(v => show(v).toLowerCase().includes(q))));
  $("cnt").textContent = `${rows.length.toLocaleString()} of ${f.rows.length.toLocaleString()} rows · ${f.cols.length} fields`;
  box.innerHTML = `<table class="raw"><thead><tr><th class="rn">#</th>${f.cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody id="tb"></tbody></table>`;
  shown = 0; more();
}
function more(){
  const end = Math.min(rows.length, shown + STEP), f = D.files[cur];
  $("tb").insertAdjacentHTML("beforeend", rows.slice(shown, end).map(r =>
    `<tr><td class="rn">${f.rows.indexOf(r) + 1}</td>${r.map(v => `<td class="${typeof v === "object" && v !== null ? "obj" : ""}">${esc(show(v))}</td>`).join("")}</tr>`).join(""));
  shown = end;
}
$("box").addEventListener("scroll", e => { const b = e.target; if (D.files[cur].kind === "table" && shown < rows.length && b.scrollTop + b.clientHeight > b.scrollHeight - 400) more(); });
$("q").addEventListener("input", render); $("sp").addEventListener("input", render);
let start = D.order[0]; try { const s = localStorage.getItem("file"); if (D.order.includes(s)) start = s } catch(e) {}
open_(start);
</script>
'''
open("/mnt/user-data/outputs/career-matcher-data-review.html", "w").write(html.replace("__DATA__", DATA).replace("__SUMMARY__", open("/home/claude/data/work/summary_block.html").read()))
print("ok")
