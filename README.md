# Routing Robustness in Mixture of Experts

MoE 路由鲁棒性课程项目。研究在不同扰动条件下，专家输入变化与路由变化分别增加还是降低预测误差。

## 最新论文

- [中文版 PDF](reports/MoE_Report_Chinese.pdf) · [LaTeX 主文件](paper/zh/main.tex)
- [English PDF](reports/MoE_Report_English.pdf) · [LaTeX main file](paper/en/main.tex)

两版采用 ICLR 2026 官方模板。论文以本仓库 `paper/` 中的修订稿为准。历史资源包内的旧论文和旧 LaTeX 不代表当前稿件。

## 本轮核查与修订

1. 遮挡数字差异已追溯并复算。表 1 使用无标记遮挡种子 9024–9026，MSE 为 0.3228636975；配对标记对照共享另一组遮挡位置，种子 9015–9017，无标记 MSE 为 0.3223950443，经验参考风险为 0.3190434546，超额误差为 0.0033515897。正文已说明不同抽样口径，没有把不同评估的结果强行统一。
2. 两种语言中平均路由概率的符号均已统一为数学形式。
3. 14 条 BibTeX 文献补全题名、作者与已核实的出处信息。

本轮按作者要求采纳第 1、3、4 项建议；未加入第 2 项建议的新增正文段落。主要实验结论与已确认的论证结构保留。

## 仓库结构

- `paper/en/`、`paper/zh/`：独立可编译的全文工程、BibTeX、ICLR 样式与四幅原图。
- `reports/`：由上述 LaTeX 编译的当前 PDF。
- `src/`：原团队资源包中的合成实验、分析、绘图及 15 项测试；训练和分析源码保持原样。
- `results/synthetic/`：合成实验的汇总 JSON、CSV 与记录。
- `results/supplied_report_evidence/`：后续报告包所附的 SARCOS 汇总、清单及历史验证记录。
- `scripts/verify_evidence.py`：当前结果文件、正文表格和数值恒等式检查。
- `audit/`：本轮检查范围、回放结果和验证输出。
- `provenance/historical_protocols/`：历史协议，反映当时阶段状态；最新范围以本 README 和审查说明为准。

## 编译论文

分别进入 `paper/en/` 或 `paper/zh/`：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

也可运行 `tectonic main.tex`。在 Overleaf 可分别上传某个语言目录的全部文件，主文件选择 `main.tex`，编译器选择 **XeLaTeX**。中文版采用 Fandol 字体，英文采用 Times 兼容的 TeX Gyre Termes 字体，无须本机宋体。

ICLR 官方样式来源：[ICLR/Master-Template](https://github.com/ICLR/Master-Template/tree/master/iclr2026)。课程项目页眉替代会议投稿或录用声明，不表示该论文已投稿或被 ICLR 录用。第三方模板保留原文件声明。

## 运行检查

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s src -p "test_*.py" -v
python scripts/verify_evidence.py
```

历史训练环境为 NumPy 1.26.4、PyTorch 2.6.0；见 `requirements-recorded.txt`。本轮在 NumPy 2.2.6、PyTorch 2.12.1 CPU 环境运行测试，并从十组历史运行的 30 个模型检查点重建合成数据、核验测试输入哈希和回放缺失标记对照。关键均值及 1,000 次配对交叉 bootstrap 区间与存档在 1e-7 容差内一致。

## 原始资源与复现边界

完整历史团队资源包、旧报告源码包作为 [GitHub Release 附件](https://github.com/oyj123321/math-group/releases/tag/resources-2026-11-03) 保存，不进入 Git 历史。它们用于保留原始检查点、预测、数据和追溯材料；内部旧论文不能替代 `paper/` 的最新版。附件 SHA-256 见 `provenance/release_assets.sha256`。

SARCOS 的 264 行汇总与历史验证记录已作一致性检查，但所附两个包未包含 `sarcos_e5.py`、该阶段检查点和逐源样本预测。因此，本轮没有重新执行 SARCOS 的 35 次训练或其逐样本 bootstrap。不要将本仓库描述成已经独立复现全部 SARCOS 实验的完整训练代码发布。

历史 SARCOS 数据准备清单和执行清单记录了两个不同的 `sarcos_e5.py` 哈希；原文件缺失时无法进一步比对版本差异。该事实及未复算项目已列于审查说明。历史 `status: passed` 文件是原资源包记录，不等同于本轮重新执行。

合成实验训练示例（新建输出目录）：

```sh
python src/run.py --out runs/independent/seed0 --device cpu --train 80000 --val 10000 --test 20000 --width 128 --epochs 100 --batch 1024 --balance 0.1 --perturb-reps 3 --seed 0
```

其余种子重复 1–4；捷径训练增加 `--shortcut`。训练开销较大，普通检查不需要重新训练。实验为探索性课程研究，不能直接推广为大型语言 MoE 的普遍规律。
