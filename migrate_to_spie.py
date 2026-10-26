from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
paper = ROOT / "MAU-YOLO26-Paper"
source_dir = paper / "WileyDesign" / "预修改基准版本"
target = paper / "spie-journal-papers-latex" / "MAU-YOLO26"
figures = target / "figures"
figures.mkdir(parents=True, exist_ok=True)

figure_sources = {
    "challenges.png": paper / "文章配图" / "问题说明" / "image.png",
    "architecture.png": paper / "drawio" / "ai绘图" / "MAU-YOLO26-v1.png",
    "mshc.png": paper / "drawio" / "ai绘图" / "MSHC-v1.png",
    "adown.png": paper / "drawio" / "ai绘图" / "ADown-v1.png",
    "upsampleanything.png": paper / "drawio" / "UpsampleAnything-v1.png",
    "dataset_scale_distribution.png": paper / "figures" / "dataset_scale_distribution.png",
    "scale_performance.pdf": paper / "figures" / "scale_performance.pdf",
    "test_diagnostics_combined.pdf": paper / "figures" / "test_diagnostics_combined.pdf",
    "qualitative.png": paper / "文章配图" / "对比试验检测效果图" / "对比试验检测效果图.png",
}
for name, src in figure_sources.items():
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, figures / name)

COMMON = r"""\documentclass[12pt]{spieman}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{placeins}
\usepackage{setspace}
\usepackage{iftex}
\ifXeTeX
  \usepackage{fontspec}
\fi
\graphicspath{{figures/}}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.05}
"""

EN_HEAD = COMMON + r"""
\title{MAU-YOLO26: A multi-scale framework for UAV construction-site object detection}
\author[1]{[TO BE COMPLETED]}
\affil[1]{[TO BE COMPLETED]}
\begin{document}
\maketitle
\begin{abstract}
Unmanned aerial vehicle (UAV)-based object detection at construction sites is important for automated safety monitoring in complex working environments. Construction-site images exhibit substantial variation in object scale, densely distributed small objects, diverse shapes, and complex backgrounds. Existing research often focuses on improving small-object detection, with less attention to joint representation and unified detection across large, medium, and small objects. To address these challenges, this study proposes MAU-YOLO26 based on YOLO26. First, multi-scale spatial heterogeneous convolution enhances the representation of local structures at different scales and orientations. ADown then combines convolution and pooling branches to aggregate complementary information while reducing feature-map resolution, mitigating the loss of object details during downsampling. Building on these operations, UpsampleAnything performs edge-guided feature reconstruction to improve boundary recovery. Experiments on the UAV Construction dataset show that MAU-YOLO26 achieves a test-set mAP50 of 86.29\% and mAP50-95 of 58.28\%, improving on the YOLO26 baseline by 1.90 and 3.00 percentage points, respectively. These results demonstrate the effectiveness of the proposed model for multi-scale object detection at construction sites.
\end{abstract}
\keywords{computer vision, construction-site safety, object detection, YOLO26, small-object detection, multi-scale feature fusion}
{\noindent\footnotesize\textbf{*}Corresponding author: [TO BE COMPLETED], \linkable{[TO BE COMPLETED]}}
\begin{spacing}{2}
"""

ZH_HEAD = COMMON + r"""
\usepackage[UTF8,fontset=windows]{ctex}
\title{MAU-YOLO26：面向无人机施工现场的多尺度目标检测框架}
\author[1]{[待补充]}
\affil[1]{[待补充]}
\begin{document}
\maketitle
\begin{abstract}
施工现场无人机目标检测对于实现复杂作业环境的自动化安全监测具有重要意义。施工现场图像具有目标尺度差异大、小目标密集、形态多样和背景复杂等特点。现有研究往往侧重于提升小目标检测能力，对大、中、小不同尺度目标的协同表征与统一检测关注不足。针对上述问题，本文以YOLO26为基线提出MAU-YOLO26。模型通过多尺度空间异构卷积增强不同尺度和方向的局部结构表达，采用ADown聚合下采样过程中的特征信息，减缓分辨率降低造成的目标细节损失；同时，利用UpsampleAnything进行边缘引导的特征重建，以改善目标边界恢复。基于UAV Construction数据集开展实验，结果表明，MAU-YOLO26在测试集上的mAP50和mAP50-95分别达到86.29\%和58.28\%，较YOLO26基线分别提高1.90和3.00个百分点，验证了该模型对施工现场多尺度目标检测的有效性。
\end{abstract}
\keywords{计算机视觉, 施工现场安全, 目标检测, YOLO26, 小目标检测, 多尺度特征融合}
{\noindent\footnotesize\textbf{*}通讯作者：[待补充]，\linkable{[待补充]}}
\begin{spacing}{2}
"""

def body_from(src: Path) -> str:
    text = src.read_text(encoding="utf-8")
    start = text.index(r"\begin{document}") + len(r"\begin{document}")
    end = text.rfind(r"\end{document}")
    body = text[start:end]
    replacements = {
        r"../../文章配图/问题说明/image.png": "challenges.png",
        r"../../drawio/ai绘图/MAU-YOLO26-v1.png": "architecture.png",
        r"../../drawio/ai绘图/MSHC-v1.png": "mshc.png",
        r"../../drawio/ADown-v1.png": "adown.png",
        r"../../drawio/ai绘图/ADown-v1.png": "adown.png",
        r"../../drawio/UpsampleAnything-v1.png": "upsampleanything.png",
        r"../../文章配图/对比试验检测效果图/对比试验检测效果图.png": "qualitative.png",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    body = body.replace(r"\begin{figure*}", r"\begin{figure}")
    body = body.replace(r"\end{figure*}", r"\end{figure}")
    # Avoid a vertically centered float-only page for the wide diagnostic plot
    # when it is placed in the SPIE single-column manuscript.
    body = body.replace(r"\begin{figure}[p]", r"\begin{figure}[t]")
    body = body.replace(r"\begin{table*}", r"\begin{table}")
    body = body.replace(r"\end{table*}", r"\end{table}")
    body = body.replace(r"\bmsection*{Declarations}", r"\section*{Declarations}")
    body = body.replace(r"\bmsection*{声明}", r"\section*{声明}")
    body = body.replace(r"\balance", "")
    body = body.replace(r"\FloatBarrier", r"\FloatBarrier")
    return body.strip() + "\n"

for stem, head in [
    ("MAU-YOLO26_English", EN_HEAD),
    ("MAU-YOLO26_Chinese", ZH_HEAD),
]:
    source = source_dir / ("UAV_Construction_YOLO26_Final_English.tex" if stem.endswith("English") else "UAV_Construction_YOLO26_Final_Chinese.tex")
    out = target / f"{stem}.tex"
    out.write_text(head + body_from(source) + r"\end{spacing}" + "\n\\end{document}\n", encoding="utf-8")

print(f"Wrote migrated manuscripts to {target}")
