"""
ui.py — shared structured CLI formatting for every pipeline script.

Pure stdlib + optional tqdm (progress bars). ASCII-only so it renders cleanly
in any Windows console / PowerShell.

Helpers:
    banner(title)          big step header (use at the start of a phase)
    rule([title])          inline horizontal rule, optionally with a caption
    subheader(title)       smaller section header
    kv_pairs(items)        aligned  key: value  block (ordered dict or [(k,v)])
    table(headers, rows)   async boxed table with auto-sized, wrapped columns
    status(label, text)    [ OK ] [WARN] [FAIL] [ .... ] prefix line
    pbar(seq, **kw)        tqdm wrapper with safe fallback
    progress_while(seq)    yields with periodic "  12/160" lines (fallback)
"""

import sys
import time


def _line(char, width):
    return char * width


def rule(title=None, char="=", width=72):
    t = f" {title} " if title else ""
    space = max(0, width - len(t))
    left = space // 2
    print(char * left + t + char * (space - left))


def banner(title, width=72):
    print()
    print(_line("=", width))
    print(title.center(width))
    print(_line("=", width))


def subheader(title, char="-", width=72):
    print()
    rule(title, char=char, width=width)


def hsep(char="-", width=72):
    print(char * width)


def kv_pairs(items, indent=2, width=72, sep=":"):
    """Print aligned 'key: value' lines. items may be a dict or [(k, v), ...]."""
    if isinstance(items, dict):
        items = list(items.items())
    if not items:
        return
    label_w = max(len(str(k)) for k, _ in items)
    for k, v in items:
        print(f"{' ' * indent}{str(k):<{label_w}}{sep} {v}")


def status(label, text, width=72):
    """label is one of: ok, warn, fail, info."""
    tag = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "info": "[ .. ]"}.get(label, "[ ? ]")
    print(f"  {tag}  {text}")


def _cell(text, limit):
    """Wrap text to lines not exceeding `limit` chars (word-safe)."""
    text = str(text)
    if len(text) <= limit:
        return [text]
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(w) > limit:
            w = w[: limit - 1] + "\u2026"
        if len(cur) + len(w) + 1 <= limit:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit_widths(widths, width):
    """Shrink widest columns (proportionally) so the table fits `width`."""
    border = 3 * (len(widths) + 1)  # +--+--+ -> 3 chars per column + walls
    total = sum(widths) + border
    if total <= width:
        return widths
    excess = total - width
    for _ in range(len(widths)):
        idx = max(range(len(widths)), key=lambda i: widths[i])
        if widths[idx] <= 6:
            break
        shrink = min(widths[idx] - 6, excess)
        widths[idx] -= shrink
        excess -= shrink
        if excess <= 0:
            break
    return widths


def table(headers, rows, title=None, max_col=34, width=72):
    """Print a boxed table. Rows are lists of cell values; long cells wrap."""
    if title:
        rule(title, char="=", width=width)
    headers = [str(h) for h in headers]
    body = [[_cell(c, max_col) for c in r] for r in rows]

    ncols = len(headers)
    widths = [len(headers[i]) for i in range(ncols)]
    for row in body:
        for i in range(ncols):
            w = max((len(line) for line in row[i]), default=0)
            widths[i] = max(widths[i], w)
    widths = _fit_widths(widths, width)

    def sep():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def render(cells):
        nlines = max((len(c) for c in cells), default=0)
        for ln in range(nlines):
            row = [cells[i][ln] if ln < len(cells[i]) else "" for i in range(ncols)]
            print("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")

    print(sep())
    render([_cell(h, max_col) for h in headers])
    for row in body:
        print(sep())
        render(row)
    print(sep())
    print()


def pbar(seq, desc="", total=None, unit=""):
    """tqdm progress bar over `seq`; falls back to plain iteration if missing."""
    try:
        from tqdm import tqdm

        return tqdm(
            seq,
            desc=desc,
            total=total,
            unit=unit,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    except Exception:
        return progress_while(seq, desc=desc, total=total)


def progress_while(seq, desc="", total=None):
    """Fallback iterable that prints a progress line every 10 items."""
    n = 0
    start = time.time()
    try:
        total = total if total is not None else len(seq)
    except TypeError:
        total = None
    for item in seq:
        yield item
        n += 1
        if total is None or n % 10 == 0:
            if desc:
                print(f"  {desc}: {n}/{total}" if total else f"  {desc}: {n}")
            else:
                print(f"  {n}/{total}" if total else f"  {n}")
    if total is None:
        print(f"  done ({time.time() - start:.1f}s)")