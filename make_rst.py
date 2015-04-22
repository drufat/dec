#!/usr/bin/env python3
'''
Convert files written in LyX or Latex to Restructured Text.
'''
import sys
import os
import subprocess
import tempfile
import re


sample_tex = r'''
\documentclass{article}
\begin{document}

The Euler Identity is given by:

\begin{equation}
e^{i\pi} + 1 = 0
\label{eq:euler}
\end{equation}

and we can refer back to the Euler Identity as Eq. \ref{eq:euler}.

\end{document}
'''

sample_rst = r'''The Euler Identity is given by:

.. math::

   e^{i\pi} + 1 = 0
   \label{eq:euler}

and we can refer back to the Euler Identity as Eq. [eq:euler].
'''

sample_rst_correct = r'''The Euler Identity is given by:

.. math::

   e^{i\pi} + 1 = 0
   :label: euler

and we can refer back to the Euler Identity as Eq. :eq:`euler`.
'''

def pandoc(source, f, t):
    source = source.encode('utf-8')
    out = subprocess.check_output(['pandoc', '-f', f, '-t', t],
                                    input=source)
    return out.decode('utf-8')

def rst2json(source):
    return pandoc(source, 'rst', 'json')

def latex2json(source):
    return pandoc(source, 'rst', 'json')

def latex2rst_(source):
    '''
    >>> latex2rst_(sample_tex) == sample_rst
    True
    '''
    return pandoc(source, 'latex', 'rst')

def latex2rst(source):
    '''
    >>> latex2rst(sample_tex) == sample_rst_correct
    True
    '''
    source = fix_tabularnewline(source)

    out = latex2rst_(source)

    out = fix_eq_labels(out)
    out = fix_pdf2png(out)
    out = fix_citations(out)
    return out

def lyx2latex(source):
    temp_lyx = tempfile.mktemp(suffix='.lyx')
    temp_name, _ = os.path.splitext(temp_lyx)
    temp_tex = '{}.tex'.format(temp_name)

    source = source
    with open(temp_lyx, 'w') as f:
        f.write(source)
    subprocess.check_call(['lyx', '--export', 'pdflatex', temp_lyx])
    assert os.path.isfile(temp_tex), 'LyX was unable to create {}.'.format(temp_tex)
    out = ''
    with open(temp_tex, 'r') as f:
        out += f.read()

    os.remove(temp_lyx)
    os.remove(temp_tex)

    return out

def lyx2rst(source):
    return latex2rst(lyx2latex(source))

sample_tabular = r'''
\documentclass{article}

\providecommand{\tabularnewline}{\\}
\begin{document}

\begin{tabular}{|c|c|}
\hline
1 & 2\tabularnewline
\hline
\hline
q & w\tabularnewline
\hline
a & s\tabularnewline
\hline
\end{tabular}

\end{document}

'''

def fix_tabularnewline(source):

    regexdef = re.compile(r'\\providecommand{\\tabularnewline}{\\\\}')
    regexuse = re.compile(r'\\tabularnewline')
    if re.search(regexdef, source):
        source = re.sub(regexdef, '', source)
        source = re.sub(regexuse, r'\\\\', source)
    return source

sample_pdf2png = r'''
.. figure:: pic/displ.pdf
   Description.

.. image:: pic/3D_lino-2.pdf
'''

def fix_pdf2png(source):
    regex = re.compile(r'([figure|image])::\s*(.*?)\.pdf')
    def f(m):
        return '{}:: {}.png'.format(m.group(1), m.group(2))
    source = re.sub(regex, f, source)
    return source

def fix_citations(source):
    regex = re.compile(r':raw-latex:\`\\cite\{(.*?)\}\`')
    def f(m):
        return ':cite:`{}`'.format(m.group(1))
    source = re.sub(regex, f, source)
    return source

def fix_eq_labels(source):

    regex = re.compile(r'\\label\{eq:(.*?)\}', re.DOTALL)
    def f(m):
        return '\n   :label: {}'.format(m.group(1))
    source = re.sub(regex, f, source)

    regex = re.compile(r'\[eq:(.*)]')
    def f(m):
        return ' :eq:`{}` '.format(m.group(1))
    source = re.sub(regex, f, source)

    return source

def process_tex(name):

    i = '{}.tex'.format(name)
    o = '{}.rst'.format(name)
    with open(i, 'r') as fin:
        with open(o, 'w') as fout:
            fout.write(latex2rst(fin.read()))
    print(o)

def process_lyx(name):

    i = '{}.lyx'.format(name)
    o = '{}.rst'.format(name)
    with open(i, 'r') as fin:
        with open(o, 'w') as fout:
            fout.write(lyx2rst(fin.read()))
    print(o)

def write_lyx2tex(name):
    i = '{}.lyx'.format(name)
    o = '{}.tex'.format(name)
    with open(i, 'r') as fin:
        with open(o, 'w') as fout:
            fout.write(lyx2latex(fin.read()))
    print(o)

def process(fname):

    assert os.path.isfile(fname), 'File {} does not exist.'.format(fname)
    name, ext = os.path.splitext(fname)
    if   ext == '.tex': process_tex(name)
    elif ext == '.lyx': process_lyx(name)
    elif ext == '.rst': pass # Do nothing
    else:
        raise Exception('Unrecognized extension {} for file {}'.format(ext, fname))

if __name__ == '__main__':
    for fname in sys.argv[1:]:
        process(fname)
