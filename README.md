# BenchIE: Benchmark for Open Information Extraction

BenchIE is a benchmark for measuring performance of Open Information Extraction (OIE) systems. Given manual annotations and a set of OIE extractions from different OIE systems, BenchIE measures precision, recall and F1 score based on our fact-based approach for evaluating OIE systems. 

You can find the details about the benchmark in our paper [*BenchIE: Open Information Extraction Evaluation Based on Facts, Not Tokens*](https://arxiv.org/abs/2109.06850). 

## Table of contents


  - [Requirements](#requirements)
  - [Demo](#demo)
  - [Data formats and data handling](#data-formats-and-data-handling)
    - [Golden annotations](#golden-annotations)
    - [OIE system extractions](#oie-system-extractions)
  - [Languages](#languages)
  - [Facets](#facets)
  - [Reproduce experiments](#reproduce-experiments)
  - [License](#license)
  - [Citing](#citing)


## Requirements

To run BenchIE, all you need to have is Python3 and NumPy (tested on version 1.21.1).

## Demo

A demo is provided in `demo.py`. In particular, for running BenchIE on the English datasets, use the following code snippet:

```python
from benchie import Benchie
import numpy as np
import pdb

# Define input files
gold_annotation_file = "data/gold/2_annotators/benchie_gold_annotations_en.txt"
clausie_extractions_file = "data/oie_systems_explicit_extractions/clausie_explicit.txt"
minie_extractions_file = "data/oie_systems_explicit_extractions/minie_explicit.txt"
stanford_extractions_file = "data/oie_systems_explicit_extractions/stanford_explicit.txt"
openie6_extractions_file = "data/oie_systems_explicit_extractions/openie6_explicit.txt"
roie_t_extractions_file = "data/oie_systems_explicit_extractions/roi_t_explicit.txt"
roie_n_extractions_file = "data/oie_systems_explicit_extractions/roi_n_explicit.txt"
naive_extractions_file = "data/oie_systems_explicit_extractions/naive_oie_extractions.txt"
m2oie_extraction_file = "data/oie_systems_explicit_extractions/m2oie_en_explicit.txt"

# Load gold annotations to BenchIE
benchie = Benchie()
benchie.load_gold_annotations(filename=gold_annotation_file)

# Add OIE systems extractions
benchie.add_oie_system_extractions(oie_system_name="clausie", filename=clausie_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="minie", filename=minie_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="stanford", filename=stanford_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="openie6", filename=openie6_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="roie_t", filename=roie_t_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="roie_n", filename=roie_n_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="naive", filename=naive_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="m2oie_en", filename=m2oie_extraction_file)

# Compute scores
benchie.compute_precision()
benchie.compute_recall()
benchie.compute_f1()

# Print scores
benchie.print_scores()
```
which outputs:

```
clausie precision: 0.5029154518950437
clausie recall: 0.25555555555555554
clausie f1: 0.33889980353634575
===============
minie precision: 0.4290617848970252
minie recall: 0.2777777777777778
minie f1: 0.33723021582733814
===============
stanford precision: 0.11082070047046524
stanford recall: 0.15703703703703703
stanford f1: 0.12994177137603433
===============
openie6 precision: 0.3110871905274489
openie6 recall: 0.21407407407407408
openie6 f1: 0.2536200087757789
===============
roie_t precision: 0.3732394366197183
roie_t recall: 0.07851851851851852
roie_t f1: 0.12974296205630356
===============
roie_n precision: 0.20287539936102236
roie_n recall: 0.09407407407407407
roie_n f1: 0.12854251012145748
===============
naive precision: 0.03336921420882669
naive recall: 0.022962962962962963
naive f1: 0.02720491443615621
===============
m2oie_en precision: 0.3924050632911392
m2oie_en recall: 0.16074074074074074
m2oie_en f1: 0.2280609563846558
```

Similarly, to run BenchIE on the datasets for Chinese and German, execute the following code:

```python
# Run BenchIE for chinese and German

# Dataset names
gold_annotation_file_zh = "data/gold/benchie_gold_annotations_zh.txt"
multi2oie_extractions_zh_file = "data/oie_systems_explicit_extractions/m2oie_zh_explicit.txt"
gold_annotation_file_de = "data/gold/benchie_gold_annotations_de.txt"
multi2oie_extractions_de_file = "data/oie_systems_explicit_extractions/m2oie_de_explicit.txt"

# BenchIE for Chinese language
benchie_zh = Benchie()
benchie_zh.load_gold_annotations(filename=gold_annotation_file_zh)
benchie_zh.add_oie_system_extractions(oie_system_name="multi2oie_zh", filename=multi2oie_extractions_zh_file)
benchie_zh.compute_precision()
benchie_zh.compute_recall()
benchie_zh.compute_f1()
benchie_zh.print_scores()

# BenchIE for German language
benchie_de = Benchie()
benchie_de.load_gold_annotations(filename=gold_annotation_file_de)
benchie_de.add_oie_system_extractions(oie_system_name="multi2oie_de", filename=multi2oie_extractions_de_file)
benchie_de.compute_precision()
benchie_de.compute_recall()
benchie_de.compute_f1()
benchie_de.print_scores()
```

## Data formats and data handling

The data in BenchIE is split in two majour groups:
   * Golden annotations: the golden annotations provided by manual annotations.
   * OIE systems extractions: the extractions generated by OIE systems.

In what follows, we explain these major data categories (i.e., their format) as well as how to handle them with the BenchIE framework.

### Golden annotations

The golden annotations are writen in `data/gold/2_annotators/benchie_gold_annotations_2annotators_with_sent_id.txt`. Each entry is written in the following format:

```
sent_id:1	He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia .
1--> Cluster 1:
He --> served as --> [the] [first] Prime Minister [of Australia]
He --> served --> as [the] [first] Prime Minister [of Australia]
1--> Cluster 2:
He --> served as [the] [first] Prime Minister of --> Australia
He --> served as [the] [first] Prime Minister --> of Australia
1--> Cluster 3:
He --> became --> [a] [founding] justice
He --> became --> [a] [founding] justice of [the] High Court [of Australia]
1--> Cluster 4:
He --> became [a] [founding] justice of --> [the] High Court [of Australia]
He --> became [a] [founding] justice --> of [the] High Court [of Australia]
1--> Cluster 5:
He --> became [a] [founding] justice of [the] High Court of --> Australia
He --> became [a] [founding] justice of [the] High Court --> of Australia

sent_id:2	Graner handcuffed him to the bars of a cell window and left him there , feet dangling off the floor , for nearly five hours .
2--> Cluster 1:
Graner --> handcuffed --> him
...
```
The first line is the input sentence, where there are two tab-separated entries:
   * `sent_id:` which is a placeholder where the sentence ID is written (in principle, this can be any string, though in our implementation we use integers)
   * sentence: the input sentence as a string
  
Next, each cluster (i.e., synset of facts) needs to be specified in a separate line in the following format:

```1--> Cluster 1:```

where the first string (in our case "1") has to match the sentence ID from the first line (`sent_id`). This needs to be followed by the string ```"--> Cluster "```, which is followed by  the cluster number (here, 1), followed by ```":"``` and a new line. 

Next, each triple is written between two consequtive clusters. For instance, the following two triples are in cluster 1, because they are written between clusters 1 and 2:
```
1--> Cluster 1:
He --> served as --> [the] [first] Prime Minister [of Australia]
He --> served --> as [the] [first] Prime Minister [of Australia]
1--> Cluster 2:
...
```
The triples are written in the format ```subject --> relation --> object```. The tokens in square brackets (```[``` and ```]```) are optional tokens. This means that when an OIE triple is evaluated for correctness, the extraction is considered to be correct even if some of the optional tokens are missing. All possible triples (with all possible combinations of optional tokens) are automatically generated in the function `gold_annotations.load_gold_annotations()`.

### OIE system extractions

The OIE systems' extractions are written in the folder `data/oie_systems_explicit_extractions`. Each file represents the extractions of one OIE system. Each line in the file is one OIE extraction and is written in the following tab-separated format:
```
sent_id <TAB> subject <TAB> relation <TAB> object
```
Note that `sent_id` should match `sent_id` from the golden extractions. For example, consider the following line:
``` 
19 <TAB> She <TAB> began <TAB> her film career
```
This means that this particular OIE triple will be evaluated w.r.t. the sentence with `sent_id=19` in the golden annotations file. 

## Languages

BenchIE provides evaluation data in several languages: English, German, Chinese, Japanese, Galician and Arabic. Some of these languages are not available yet, but they soon will be.

## Facets

BenchIE comes in multiple facets: BenchIE (regular), BenchIE-E (entity), BenchIE-M (minimality) and BenchIE-C (concatenate). For now, the code is available only on the regular BenchIE facet. The data and the framework code for the other facets will be soon available.

## Reproduce experiments

A notebook for reproducing the experiments from the paper (TBA)

## License

The software framework is licensed according to the license for academic or non-profit organization noncommercial research use only. Details are provided in the [license file](https://github.com/gkiril/benchie/blob/main/LICENSE.txt) and in the header of each source code file. The data is under the non-restrictive [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Citing
If you use BenchIE for your research, please cite the following paper:

```
@inproceedings{Gashteovski2022BenchIEAF,
  title={{BenchIE: A Framework for Multi-Faceted Fact-Based Open Information Extraction Evaluation}},
  author={Kiril Gashteovski and Mingying Yu and Bhushan Kotnis and Carolin Lawrence and Mathias Niepert and Goran Glavavs},
  booktitle={ACL},
  url={https://aclanthology.org/2022.acl-long.307/},
  year={2022}
}
```
