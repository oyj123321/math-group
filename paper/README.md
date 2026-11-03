# 编译当前论文

当前主文件是 `en/main.tex` 和 `zh/main.tex`。每个语言目录均独立包含模板、BibTeX 与配图。分别进入目录后用 XeLaTeX → BibTeX → XeLaTeX → XeLaTeX 编译，或使用 Tectonic。

上传 Overleaf 时，建议分别上传单个语言目录内的全部文件，将主文档设为 `main.tex`，编译器设为 XeLaTeX。完整文献由 `references.bib` 管理。

该版本采用官方 ICLR 2026 样式，保留课程项目页眉与作者。英文使用 TeX Gyre Termes，中文使用 Fandol，无须额外安装本机中文字体。

已编译 PDF 位于仓库根目录下的 `reports/`。历史附件内的旧论文不代表当前版本。
