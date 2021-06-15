# Generalizing Doors in Technical Drawings using Graph Neural Networks
By Christoffer P. Simonsen, Frederik M. Thiesson, Mark P. Philipsen, and Thomas B. Moeslund

[2020/06/17: Master thesis defense & project ended]

[2021/01/29: data/code/models available]

[2021/04/20: Paper accepted at ICIP 2021]

## Introduction

This repository contains the code & links to annotation data for our ICIP 2021 submission: 'Generalizing Doors in Technical Drawings using Graph Neural Networks'. Here, we present how floor plan elements can be recognized by constructing graphs from the primitives in dxf files and performing node classification using Graph Neural Networks.

## Requirements

- OpenCV
- Python 3
- pip3 torch ezdxf

sudo apt-get install python3-tk

Our code has been tested using torch==1.5.1 & pytorch-lightning==0.8.3 on a Threadripper 2950x and Nvidia 2060 GPU machine with CUDA 10.0 installed.

## Data

Our pubic dataset is available in the [data/](data/Public) directory. The dataset consists of all of the tools and files shown in Figure

![dataset_creation_and_content](figs/dataset_creation_and_content.png)

The data is available as ready to use labeled graphs stored as .gpickle files. In addition, we provide the original dxf files as well as tools for extracting the graphs and generating rasterized floor plan images.

More details about the RUB dataset and other floorplan datasets see [datasets.pdf](data/datasets.pdf) in the data/ directory.

## Tools

**DXF -> Line data -> Graph:** To extract graphs from dxf files use DxfReader.extract_data() to create line data and DxfReader.convert_data_to_graph() to convert it to a graph - see main.py

**Annotate graphs:** Use annotation_tool.py to annotate the graphs or use our annotations in [/data/labels](/data/labels)

**Normalize graphs:** Use xx.py to scale the graphs and labels to a normalized scale.

**Convert to png:**
from https://askubuntu.com/questions/612169/can-i-convert-a-dxf-to-png-or-jpg-with-a-command-tool

sudo apt install libreoffice texlive-extra-utils

- libreoffice --headless --convert-to pdf A1322PE-0.dxf
- pdfcrop A1322PE-0.pdf
- gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -sOutputFile=A1322PE-0.png A1322PE-0-crop.pdf

or use dxf2png.py -l <list_of_dxf_files>.txt

NB: fails to render arcs correctly

## Usage

**Try network**

A pre-trained model can be downloaded using the script in [/trained_models](/trained_models). Use mode flag to decide whether to perform "node" or "element" prediction.

```bash
python predict.py --mode=node --graph=test.npy
```

**Train network**

Configure model in config.py  

```bash
python main.py
```

**Evaluate predictions**

Performs both node- and element level evaluation.

```bash
python eval.py --mode=node --model=/trained_models/model.pt
```

## Citation

Please consider citing:
---
TODO
---
