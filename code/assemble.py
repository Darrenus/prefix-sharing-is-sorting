# -*- coding: utf-8 -*-
import re, sys
body = open('/tmp/body.tex').read()
lines = body.split('\n')

# ---- split off front matter (everything up to the first \section) ----
first_sec = next(i for i,l in enumerate(lines) if l.startswith('\\section{'))
front, rest = lines[:first_sec], lines[first_sec:]
abstract = [l for l in front if l.startswith('\\textbf{Abstract.}') or
            (l.strip() and not l.startswith('\\title') and not l.startswith('\\textbf{Rong He}')
             and 'github.com' not in l)]
abstract = [a.replace('\\textbf{Abstract.} ','') for a in abstract]

# ---- section numbers are already in the text; strip them, let LaTeX number ----
out=[]
for l in rest:
    l = re.sub(r'\\(sub)?section\{(\d+(\.\d+)?)\.?\s+', lambda m:'\\'+(m.group(1) or '')+'section{', l)
    out.append(l)

# ---- theorem environments ----
NAMES = {}
res=[]; i=0
while i < len(out):
    l = out[i]
    m = re.match(r'^\\textbf\{(Theorem|Proposition) (\d+)(?: \(([^)]*)\))?\.\}\s*(.*)$', l)
    if m:
        kind, num, name, tail = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
        hdr = '\\begin{%s}%s'%(kind, '[%s]'%name if name else '')
        blk=[tail] if tail.strip() else []
        j=i+1
        while j < len(out) and not out[j].startswith('\\emph{Proof.}') \
              and not out[j].startswith('\\textbf{') and not out[j].startswith('\\section') \
              and not out[j].startswith('\\subsection'):
            blk.append(out[j]); j+=1
        while blk and blk[-1].strip()=='' : blk.pop()
        res.append(hdr); res.extend(blk); res.append('\\end{%s}'%kind)
        i=j; continue
    if l.startswith('\\emph{Proof.}'):
        first = l.replace('\\emph{Proof.}','',1).strip()
        blk=[]; j=i
        if '@@QED@@' in first:                       # QED on the very same line
            blk=[first]; j=i+1
        else:
            blk=[first]; j=i+1
            while j < len(out) and '@@QED@@' not in out[j]:
                blk.append(out[j]); j+=1
            if j < len(out):
                blk.append(out[j]); j+=1
        res.append('\\begin{proof}'); res.extend(blk); res.append('\\end{proof}')
        i=j; continue
    res.append(l); i+=1
out=res

# ---- wide tables: shrink to \textwidth ----
res=[]; open_rb=False
for l in out:
    m=re.match(r'^\\begin\{tabular\}\{([lrc]+)\}', l)
    if m and len(m.group(1))>6:
        res.append('\\resizebox{\\textwidth}{!}{%'); res.append(l); open_rb=True; continue
    if l.startswith('\\bottomrule\\end{tabular}\\end{center}') and open_rb:
        res.append('\\bottomrule\\end{tabular}}\\end{center}'); open_rb=False; continue
    res.append(l)
out=res

# ---- references -> thebibliography ----
txt='\n'.join(out)
_ref = txt.find('\\section{References}')
_head, _tail = (txt[:_ref], txt[_ref:]) if _ref>0 else (txt, '')
_head = re.sub(r'\[([A-Z]{2,4}\d{0,2}(?:\s*,\s*[A-Z]{2,4}\d{0,2})*)\]',
               lambda m: '\\cite{'+','.join(k.strip() for k in m.group(1).split(','))+'}', _head)
txt = _head + _tail
mref = re.search(r'\\section\{References\}(.*?)(\\section\{)', txt, re.S)
bib=''
if mref:
    items=[]
    for it in re.findall(r'\\item \[([A-Z]{2,4}\d{0,2})\]\s*(.*)', mref.group(1)):
        items.append('\\bibitem{%s} %s'%(it[0],it[1]))
    if not items:
        for it in re.findall(r'\\item (\[[A-Z]{2,4}\d{0,2}\])\s*(.*)', mref.group(1)):
            items.append('\\bibitem{%s} %s'%(it[0].strip('[]'),it[1]))
    bib='\\begin{thebibliography}{99}\n'+'\n'.join(items)+'\n\\end{thebibliography}\n'
    txt = txt[:mref.start()] + bib + mref.group(2) + txt[mref.end():]

PRE = r'''\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1.1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{3em}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\title{\bf Prefix Sharing Is a Sorting Problem}
\author{Rong He}
\date{\today}

\begin{document}
\maketitle
\begin{center}\small
Code, data pipeline, and verification scripts:\\
\url{https://github.com/Darrenus/prefix-sharing-is-sorting}
\end{center}
\begin{abstract}
%(ABSTRACT)s
\end{abstract}
'''
doc = PRE % {'ABSTRACT': '\n\n'.join(a for a in abstract if a.strip())} + txt + '\n\\end{document}\n'
doc = doc.replace('@@QED@@','')
open('paper.tex','w').write(doc)
print("wrote paper.tex,", doc.count('\n'), "lines")
