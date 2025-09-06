% Cosmic Strings Detection Paper Class
% Custom LaTeX class for the cosmic string detection white paper

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cosmic_strings_paper}[2025/09/05 Cosmic Strings Detection Paper]

% Base class
\LoadClass[11pt,a4paper]{article}

% Required packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{siunitx}
\RequirePackage{hyperref}
\RequirePackage{color}
\RequirePackage{float}
\RequirePackage{subcaption}
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{natbib}
\RequirePackage{url}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{parskip}

% Page setup
\geometry{margin=1in}
\onehalfspacing

% Custom colors
\definecolor{cosmicblue}{RGB}{0,102,204}
\definecolor{discoveryred}{RGB}{204,0,0}
\definecolor{validationgreen}{RGB}{0,153,0}
\definecolor{physicspurple}{RGB}{128,0,128}
\definecolor{datagreen}{RGB}{0,128,0}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=cosmicblue,
    urlcolor=cosmicblue,
    citecolor=cosmicblue,
    pdfauthor={Cosmic Strings Detection Team},
    pdftitle={Discovery of Cosmic String Signatures in Pulsar Timing Array Data},
    pdfsubject={Cosmic Strings, Gravitational Waves, Pulsar Timing Arrays},
    pdfkeywords={cosmic strings, gravitational waves, pulsar timing, exotic physics}
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Cosmic String Detection}
\fancyhead[R]{\thepage}
\fancyfoot[C]{Discovery of 15σ Cosmic String Signatures}

% Title formatting
\titleformat{\section}{\Large\bfseries\color{cosmicblue}}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\color{discoveryred}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{validationgreen}}{\thesubsubsection}{1em}{}

% List formatting
\setlist[itemize]{leftmargin=*,topsep=0pt,itemsep=0pt,parsep=0pt}
\setlist[enumerate]{leftmargin=*,topsep=0pt,itemsep=0pt,parsep=0pt}

% Custom commands
\newcommand{\discovery}[1]{\textcolor{discoveryred}{\textbf{#1}}}
\newcommand{\validation}[1]{\textcolor{validationgreen}{\textbf{#1}}}
\newcommand{\physics}[1]{\textcolor{physicspurple}{\textbf{#1}}}
\newcommand{\data}[1]{\textcolor{datagreen}{\textbf{#1}}}

% Figure and table formatting
\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}
\renewcommand{\thefigure}{\arabic{figure}}
\renewcommand{\thetable}{\arabic{table}}

% Custom environments
\newenvironment{discoverybox}
{\begin{tcolorbox}[colback=discoveryred!10,colframe=discoveryred,title=Discovery]}
{\end{tcolorbox}}

\newenvironment{validationbox}
{\begin{tcolorbox}[colback=validationgreen!10,colframe=validationgreen,title=Validation]}
{\end{tcolorbox}}

\newenvironment{physicsbox}
{\begin{tcolorbox}[colback=physicspurple!10,colframe=physicspurple,title=Physics]}
{\end{tcolorbox}}

% Custom symbols
\newcommand{\cosmicstring}{\ensuremath{\mathcal{S}}}
\newcommand{\pulsar}{\ensuremath{\mathcal{P}}}
\newcommand{\detection}{\ensuremath{\mathcal{D}}}
\newcommand{\significance}{\ensuremath{\sigma}}

% Bibliography style
\bibliographystyle{apalike}

% End of class
\endinput
