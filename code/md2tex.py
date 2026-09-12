# -*- coding: utf-8 -*-
import re, sys

UNI = {
 '—':'---', '–':'--', '×':'$\\times$', '≈':'$\\approx$', '≥':'$\\ge$', '≤':'$\\le$',
 '−':'$-$', '…':'\\dots', '“':'``', '”':"''", '‘':'`', '’':"'", '·':'$\\cdot$',
 '∩':'$\\cap$', '⊆':'$\\subseteq$', '⌈':'$\\lceil$', '⌉':'$\\rceil$', 'Θ':'$\\Theta$',
 '≠':'$\\neq$', '∅':'$\\emptyset$', '∞':'$\\infty$', '→':'$\\to$', '⊋':'$\\supsetneq$',
}
SPECIAL = {'%':'\\%','&':'\\&','#':'\\#','_':'\\_'}

def esc_text(s):
    out=[]
    for ch in s:
        if ch in SPECIAL: out.append(SPECIAL[ch])
        elif ch in UNI:   out.append(UNI[ch])
        else:             out.append(ch)
    return ''.join(out)

def inline(s):
    """hide inline math behind placeholders, then run emphasis + escaping, then restore"""
    s = s.replace('$\\square$','@@QED@@')
    math=[]
    def stash(m):
        math.append(m.group(0)); return '\x00%d\x00'%(len(math)-1)
    s = re.sub(r'\$[^$]*\$', stash, s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<![\w*\\])\*(?!\*)(.+?)(?<!\*)\*(?![\w*])', r'\\emph{\1}', s)
    code=[]
    def stash_code(m):
        inner=m.group(1)
        for a,b in (('_','\\_'),('%','\\%'),('#','\\#'),('&','\\&')):
            inner=inner.replace(a,b)
        code.append('\\texttt{'+inner+'}'); return '\x01%d\x01'%(len(code)-1)
    s = re.sub(r'`([^`]*)`', stash_code, s)
    s = esc_text(s)
    s = re.sub(r'\x01(\d+)\x01', lambda m: code[int(m.group(1))], s)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\\href{\2}{\1}', s)
    s = re.sub(r'(?<=\d) (?=\d\d\d\b)', r'\\,', s)
    s = re.sub(r'\x00(\d+)\x00', lambda m: math[int(m.group(1))], s)
    return s

def convert(md):
    lines = md.split('\n')
    out=[]; i=0; in_list=None; in_table=False
    def close_list():
        nonlocal in_list
        if in_list: out.append('\\end{%s}'%in_list); in_list=None
    while i < len(lines):
        L = lines[i]
        # tables
        if L.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:\-|]+\|$', lines[i+1]):
            close_list()
            hdr=[c.strip() for c in L.strip('|').split('|')]
            align=[c.strip() for c in lines[i+1].strip('|').split('|')]
            spec=''.join('r' if a.endswith(':') and not a.startswith(':') else ('c' if a.startswith(':') and a.endswith(':') else 'l') for a in align)
            rows=[]; i+=2
            while i<len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip('|').split('|')]); i+=1
            out.append('\\begin{center}\\small')
            out.append('\\begin{tabular}{%s}\\toprule'%spec)
            out.append(' & '.join(inline(c) for c in hdr)+' \\\\ \\midrule')
            for r in rows:
                r=(r+['']*len(hdr))[:len(hdr)]
                out.append(' & '.join(inline(c) for c in r)+' \\\\')
            out.append('\\bottomrule\\end{tabular}\\end{center}')
            continue
        if L.strip().startswith('$$'):
            close_list()
            chunk=L.strip()[2:]
            if chunk.rstrip().endswith('$$'):
                out.append('\\begin{equation}\n'+chunk.rstrip()[:-2]+'\n\\end{equation}'); i+=1; continue
            body=[chunk]; i+=1
            while i<len(lines):
                t=lines[i]
                if '$$' in t:
                    body.append(t[:t.index('$$')]); i+=1; break
                body.append(t); i+=1
            out.append('\\begin{equation}\n\\begin{split}\n'+'\n'.join(body)+'\n\\end{split}\n\\end{equation}'); continue
        if L.strip().startswith('```'):
            close_list(); i+=1; body=[]
            while i<len(lines) and not lines[i].strip().startswith('```'):
                body.append(lines[i]); i+=1
            i+=1
            out.append('{\\footnotesize\\begin{verbatim}\n'+'\n'.join(body)+'\n\\end{verbatim}}'); continue
        if L.strip()=='---': close_list(); i+=1; continue
        m=re.match(r'^(#{1,4})\s+(.*)$', L)
        if m:
            close_list(); lvl=len(m.group(1)); t=inline(m.group(2))
            if lvl==1: out.append('\\title{%s}'%t)
            elif lvl==2: out.append('\\section{%s}'%t)
            elif lvl==3: out.append('\\subsection{%s}'%t)
            else: out.append('\\subsubsection{%s}'%t)
            i+=1; continue
        def take_item(text, kind):
            """gather an item plus its indented continuation lines, then convert as one unit"""
            nonlocal i, in_list
            if in_list != kind:
                close_list(); out.append('\\begin{%s}'%kind); in_list = kind
            buf=[text]; i+=1
            while i < len(lines) and re.match(r'^\s\s+\S', lines[i]) \
                  and not re.match(r'^\s*(-|\d+\.)\s', lines[i]):
                buf.append(lines[i].strip()); i+=1
            out.append('\\item '+inline(' '.join(buf)))
        m=re.match(r'^(\s*)-\s+(.*)$', L)
        if m:
            take_item(m.group(2), 'itemize'); continue
        m=re.match(r'^(\s*)(\d+)\.\s+(.*)$', L)
        if m:
            take_item(m.group(3), 'enumerate'); continue
        if L.startswith('> '):
            close_list(); body=[]
            while i<len(lines) and lines[i].startswith('> '):
                body.append(lines[i][2:]); i+=1
            out.append('\\begin{quote}\\itshape\n'+inline(' '.join(body))+'\n\\end{quote}'); continue
        if L.strip()=='':
            if in_list and i+1<len(lines) and re.match(r'^\s*(-|\d+\.)\s', lines[i+1]):
                i+=1; continue
            close_list(); out.append(''); i+=1; continue
        close_list()
        para=[L]; i+=1
        while i<len(lines):
            nxt=lines[i]
            if (nxt.strip()=='' or nxt.startswith('|') or nxt.startswith('#')
                or nxt.strip()=='---' or nxt.strip().startswith('```')
                or re.match(r'^\s*(-|\d+\.)\s', nxt) or nxt.startswith('> ')
                or nxt.strip().startswith('$$')): break
            para.append(nxt); i+=1
        out.append(inline(' '.join(x.strip() for x in para)))
    close_list()
    return '\n'.join(out)

md=open(sys.argv[1]).read()
body=convert(md)
open(sys.argv[2],'w').write(body)
print("converted ->", sys.argv[2], len(body.split(chr(10))), "lines")
