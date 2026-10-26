#!/usr/bin/env python3
"""Build REPORT.html from REPORT.md — one self-contained file, figures inlined.

Why a generator instead of hand-written HTML: the Markdown file is the single
source of truth (it is what gets diffed and reviewed). If the HTML were hand
maintained, the two would drift and the pretty one would win by default, which
is exactly backwards for a report about measurement discipline.

What this does
  1. replaces every `![alt](figures/x.png)` with a <figure> whose <img> carries
     the PNG as a base64 data URI, so the HTML opens offline with no asset dir;
  2. runs python-markdown (tables / fenced_code / sane_lists / attr_list / toc);
  3. builds a two-level sidebar TOC out of toc_tokens;
  4. post-processes tables (scroll wrapper, verdict chips in cells);
  5. wraps everything in one HTML shell with screen + print CSS.

Usage:  python tools/build_report_html.py [--src REPORT.md] [--out REPORT.html]
Exit 0 on success; non-zero with a message if a referenced figure is missing.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root{
  --ink:#1b1c1e; --ink-2:#4a4d52; --ink-3:#74797f;
  --bg:#f4f4f2; --card:#ffffff; --line:#e2e1dd; --line-2:#cfcec9;
  --accent:#2f5d8a; --accent-soft:#eaf1f8;
  --ok:#1f6b45; --ok-bg:#e8f3ec;
  --bad:#9b2c2c; --bad-bg:#fbeceb;
  --warn:#8a5a1a; --warn-bg:#fbf1e2;
  --explore:#5b4b8a; --explore-bg:#efecf7;
  --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  --sans:"Inter","Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans SC",system-ui,sans-serif;
  --serif:"Source Han Serif SC","Songti SC","Georgia","Noto Serif SC",serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.78;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1240px;margin:0 auto;padding:0 28px 96px;}
.layout{display:grid;grid-template-columns:212px minmax(0,1fr);gap:52px;align-items:start;}
main{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:46px 56px 60px;box-shadow:0 1px 2px rgba(0,0,0,.03);min-width:0;}

/* ---------- header ---------- */
header.masthead{padding:56px 0 30px;}
.eyebrow{font-size:12.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3);margin-bottom:14px}
h1.title{font-family:var(--serif);font-size:34px;line-height:1.3;margin:0 0 10px;letter-spacing:-.01em;font-weight:700}
.subtitle{font-size:16px;color:var(--ink-2);margin:0 0 22px;max-width:78ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:4px}
.chip{font-size:12.5px;color:var(--ink-2);background:#fff;border:1px solid var(--line);
      border-radius:999px;padding:3px 11px;white-space:nowrap}
.chip b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:18px;font-size:12.5px;color:var(--ink-3)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}

/* ---------- toc ---------- */
nav.toc{position:sticky;top:26px;max-height:calc(100vh - 52px);overflow:auto;
        font-size:13px;padding:8px 4px 8px 0;}
nav.toc .toc-h{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;
               color:var(--ink-3);margin:0 0 12px;padding-left:10px}
nav.toc ul{list-style:none;margin:0;padding:0}
nav.toc li{margin:0}
nav.toc a{display:block;padding:5px 10px;color:var(--ink-2);text-decoration:none;
          border-left:2px solid transparent;line-height:1.45}
nav.toc a:hover{color:var(--accent);background:var(--accent-soft)}
nav.toc a.active{color:var(--accent);border-left-color:var(--accent);
                 background:var(--accent-soft);font-weight:600}
nav.toc ul ul a{padding-left:22px;font-size:12.4px;color:var(--ink-3)}

/* ---------- typography ---------- */
main h2{font-family:var(--serif);font-size:25px;font-weight:700;margin:54px 0 16px;
        padding-top:22px;border-top:1px solid var(--line);scroll-margin-top:20px}
main h2:first-of-type{border-top:0;padding-top:0;margin-top:8px}
main h3{font-size:18px;font-weight:650;margin:34px 0 10px;color:#26282b;scroll-margin-top:20px}
main h4{font-size:15.5px;margin:24px 0 8px}
main p{margin:0 0 15px}
main a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c9d8e8}
main a:hover{border-bottom-color:var(--accent)}
strong{font-weight:650;color:#111}
hr{border:0;border-top:1px solid var(--line);margin:38px 0}
ul,ol{margin:0 0 16px;padding-left:24px}
li{margin:5px 0}
code{font-family:var(--mono);font-size:.87em;background:#f2f2ef;border:1px solid #e6e5e1;
     border-radius:4px;padding:1px 5px;color:#3d4046;word-break:break-word}
pre{background:#1f2226;color:#e8e8e6;padding:16px 18px;border-radius:8px;overflow:auto;
    font-size:13px;line-height:1.65;margin:0 0 18px}
pre code{background:none;border:0;color:inherit;padding:0;font-size:13px}

/* ---------- blockquote / callout ---------- */
blockquote{margin:0 0 18px;padding:14px 18px;background:#faf9f6;border-left:3px solid var(--line-2);
           color:var(--ink-2);border-radius:0 6px 6px 0;font-size:15px}
blockquote p:last-child{margin-bottom:0}
blockquote.lead{background:var(--accent-soft);border-left-color:var(--accent);color:#2b3d4f}

/* ---------- tables ---------- */
.tw{overflow-x:auto;margin:0 0 22px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px;line-height:1.6;background:#fff}
thead th{background:#f7f7f4;text-align:left;font-weight:650;color:#33363a;
         padding:9px 12px;border-bottom:1px solid var(--line-2);white-space:nowrap;
         position:sticky;top:0}
tbody td{padding:8px 12px;border-bottom:1px solid #eeedea;vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:#fbfbf9}
td.num,th.num{font-variant-numeric:tabular-nums}

/* ---------- verdict chips inside cells ---------- */
.v{display:inline-block;font-size:11.5px;font-weight:650;line-height:1.5;
   padding:1px 7px;border-radius:4px;border:1px solid transparent;white-space:nowrap}
.v-ok{color:var(--ok);background:var(--ok-bg);border-color:#cfe4d7}
.v-bad{color:var(--bad);background:var(--bad-bg);border-color:#f0d3d0}
.v-warn{color:var(--warn);background:var(--warn-bg);border-color:#f0dfc4}
.v-explore{color:var(--explore);background:var(--explore-bg);border-color:#ddd6ef}

/* ---------- figures ---------- */
figure.fig{margin:26px 0 26px;text-align:center}
figure.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figure.fig figcaption{font-size:13px;color:var(--ink-3);margin-top:9px}

/* ---------- footer ---------- */
footer.foot{border-top:1px solid var(--line);margin-top:56px;padding-top:18px;
            font-size:12.5px;color:var(--ink-3);display:flex;flex-wrap:wrap;gap:16px;
            justify-content:space-between}

/* ---------- responsive ---------- */
@media (max-width:1080px){
  .layout{grid-template-columns:1fr}
  nav.toc{display:none}
  main{padding:32px 24px 40px}
  .wrap{padding:0 14px 60px}
}
@media print{
  body{background:#fff;font-size:10.5pt}
  nav.toc,.legend,footer.foot .build{display:none}
  .wrap{max-width:none;padding:0}
  .layout{display:block}
  main{border:0;box-shadow:none;padding:0}
  main h2{border-top:0;page-break-after:avoid}
  main h3{page-break-after:avoid}
  .tw{overflow:visible;border:0}
  table{font-size:9.5pt}
  thead th{position:static}
  figure.fig{page-break-inside:avoid}
  tr,blockquote{page-break-inside:avoid}
}
"""

JS = """
(function(){
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a'));
  var targets = links.map(function(a){
    var id = decodeURIComponent(a.getAttribute('href').slice(1));
    return document.getElementById(id);
  }).filter(Boolean);
  if(!targets.length) return;
  function setActive(id){
    links.forEach(function(a){
      var on = decodeURIComponent(a.getAttribute('href').slice(1)) === id;
      a.classList.toggle('active', on);
    });
  }
  var visible = {};
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting) visible[e.target.id] = e.boundingClientRect.top;
      else delete visible[e.target.id];
    });
    var ids = Object.keys(visible);
    if(ids.length){
      ids.sort(function(a,b){ return visible[a] - visible[b]; });
      setActive(ids[0]);
    }
  }, {rootMargin:'-10% 0px -70% 0px', threshold:[0,1]});
  targets.forEach(function(t){ io.observe(t); });
})();
"""

VERDICTS = {
    "通过": "ok", "成立": "ok", "verified": "ok",
    "相反": "bad", "证伪": "bad", "不成立": "bad", "未复现": "bad",
    "部分": "warn", "未检验": "warn", "不可识别": "warn", "不可触发": "warn",
    "待定": "warn", "不可判定": "warn", "未缓解": "warn",
    "探索性": "explore", "非预注册": "explore", "未预注册": "explore",
}

FIG_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$", re.M)


def inline_figures(src: str, base: Path) -> tuple[str, list[tuple[str, int]]]:
    """Replace markdown images with <figure> blocks carrying base64 data URIs."""
    used: list[tuple[str, int]] = []

    def repl(m: re.Match) -> str:
        alt, rel = m.group("alt").strip(), m.group("path").strip()
        p = (base / rel).resolve()
        if not p.is_file():
            raise SystemExit(f"[fatal] figure missing: {rel} (looked at {p})")
        raw = p.read_bytes()
        used.append((rel, len(raw)))
        b64 = base64.b64encode(raw).decode("ascii")
        return (
            f'\n<figure class="fig">'
            f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'
            f'<figcaption>{html.escape(alt)}</figcaption></figure>\n'
        )

    return FIG_RE.sub(repl, src), used


def wrap_tables(body: str) -> str:
    return body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")


def chipify_cells(body: str) -> str:
    """Colour whole-cell verdict words (<strong>通过</strong>) as small chips."""

    def repl(m: re.Match) -> str:
        word = m.group(1)
        cls = VERDICTS.get(word)
        if not cls:
            return m.group(0)
        return f'<strong class="v v-{cls}">{word}</strong>'

    # only inside <td>…</td> chunks
    def per_cell(cm: re.Match) -> str:
        return re.sub(r"<strong>([^<]{1,8})</strong>", repl, cm.group(0))

    return re.sub(r"<td[^>]*>.*?</td>", per_cell, body, flags=re.S)


def mark_lead_blockquotes(body: str) -> str:
    """The first blockquote run (报告对象/性质/状态) gets the accent style."""
    blocks = list(re.finditer(r"<blockquote>", body))
    if not blocks:
        return body
    first = blocks[0]
    end = body.find("</blockquote>", first.start())
    if end == -1:
        return body
    return body[: first.start()] + '<blockquote class="lead">' + body[first.end() :]


def render_toc(tokens: list[dict]) -> str:
    out = ['<nav class="toc" aria-label="目录"><p class="toc-h">目录</p><ul>']
    for t in tokens:
        if t["level"] > 2:
            continue
        out.append(f'<li><a href="#sec-{t["id"]}">{html.escape(t["name"])}</a>')
        kids = [c for c in t.get("children", []) if c["level"] == 3]
        if kids:
            out.append("<ul>")
            for c in kids:
                out.append(f'<li><a href="#sec-{c["id"]}">{html.escape(c["name"])}</a></li>')
            out.append("</ul>")
        out.append("</li>")
    out.append("</ul></nav>")
    return "".join(out)


def prefix_heading_ids(body: str) -> str:
    """python-markdown derives ids straight from the text ('5','51'); prefix them
    so the fragment is a valid, readable anchor ('#sec-51')."""
    return re.sub(r'<(h[1-6]) id="([^"]+)"', r'<\1 id="sec-\2"', body)


def git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=path, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="REPORT.md")
    ap.add_argument("--out", default="REPORT.html")
    args = ap.parse_args()

    import markdown  # noqa: WPS433 (deliberate: clear error if not installed)

    src_path = (ROOT / args.src).resolve()
    out_path = (ROOT / args.out).resolve()
    src = src_path.read_text(encoding="utf-8")
    md_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]

    src_md, used = inline_figures(src, ROOT)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc", "md_in_html"],
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
    )
    body = md.convert(src_md)
    body = prefix_heading_ids(body)
    body = wrap_tables(body)
    body = chipify_cells(body)
    body = mark_lead_blockquotes(body)

    # pull the H1 + lead blockquote out as the masthead
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    title = re.sub(r"<[^>]+>", "", h1.group(1)).strip() if h1 else "REPORT"
    if h1:
        body = body[: h1.start()] + body[h1.end() :]

    toc_html = render_toc(md.toc_tokens or [])
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = git(ROOT, "rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git(ROOT, "status", "--porcelain"))
    fig_note = " · ".join(f"{Path(p).name} {n // 1024} KB" for p, n in used)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="µ 扫描实验报告：量化 Two Clocks 假设 A2 的代价">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="masthead">
  <div class="eyebrow">Audit report · Two Clocks / arXiv:2610.18853 · 假设 A2</div>
  <h1 class="title">{html.escape(title)}</h1>
  <p class="subtitle">固定 <code>Br=4096</code>、<code>G=8</code>、每 run 32,768 responses 的同一预算，只操纵「一个 rollout 被重复用于几次优化更新」：µ ∈ {{1, 8, 16}} × 3 seeds。测量 A2 显式忽略的陈旧性项，以及它能否被 progress clock 的判据识别。</p>
  <div class="chips">
    <span class="chip"><b>9</b> run</span>
    <span class="chip"><b>294,912</b> responses</span>
    <span class="chip"><b>15.578</b> GPU-h</span>
    <span class="chip"><b>¥32.40</b> 内部记账</span>
    <span class="chip"><b>608/608</b> 文件哈希相符</span>
    <span class="chip">Qwen3-1.7B-Base + LoRA r64</span>
  </div>
  <div class="legend">
    <span><i class="dot" style="background:#cfe4d7"></i>通过 / 成立</span>
    <span><i class="dot" style="background:#f0d3d0"></i>相反 / 未复现</span>
    <span><i class="dot" style="background:#f0dfc4"></i>部分 / 未检验 / 不可识别</span>
    <span><i class="dot" style="background:#ddd6ef"></i>探索性（未预注册）</span>
  </div>
</header>

<div class="layout">
{toc_html}
<main>
{body}
<footer class="foot">
  <div>事实源：<code>{args.src}</code> · sha256:{md_sha} · 由 <code>tools/build_report_html.py</code> 生成</div>
  <div class="build">commit {commit}{" (dirty)" if dirty else ""} · built {built} · 图内嵌：{fig_note}</div>
</footer>
</main>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    # newline="\n": on Windows the default would write CRLF, and .gitattributes
    # (* text=auto eol=lf) normalises it away on commit — so the working copy
    # would keep a CRLF file that no diff ever shows. Same trap the yaml configs
    # hit on 2026-10-25. Pin LF here instead.
    out_path.write_text(doc, encoding="utf-8", newline="\n")
    print(f"[ok] {out_path.relative_to(ROOT)}  {len(doc) / 1024:.0f} KB")
    print(f"     figures inlined: {len(used)} ({sum(n for _, n in used) / 1024:.0f} KB raw)")
    print(f"     toc entries: {len(md.toc_tokens or [])} top-level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
