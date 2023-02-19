# Style Transfer in Text

Style transfer in text refers to the task of converting the style of one text to that of another text while preserving its content. The objective is to develop an approach that can take a source text as input and generate a new text in a target style.\
\
 This project has been developed as part of completion of the Neural Natural Language Generation course, IIIT-Hyderabad (2023).

## Repository Structure

```
.
├── evaluators              # contains custom metric evaluation scripts
├── scripts                 # contains training scripts
├── utils                   # contains additional scripts
├── environment.txt         # project level dependencies
└── README.md
```

### `evaluators`

```
.
├── cps_evaluator.py        # Content Preservation Score
├── sti_evaluator.py        # Style Transfer Intensity
└── style_evaluator.py      # Style Strength
```

### `scripts`

```
.
├── bart.sh                 # finetuning BART for baseline
├── bert.sh                 # finetuning BERT classifier
└── gpt.sh                  # finetuning GPT-2 for baseline
```

### `utils`

```
.
└── bert_cls.py             # train script for BERT classifier fine-tuning
```

## Setting up the environment

```
conda create --name <env> --file environment.txt
conda activate <env>
```

## Members

Padakanti Srijith (2019114002)\
Jerrin John Thomas (2019114012)\
Nikhil Bishnoi (2019114021)\
Suraj Kumar (2019900038)\
T Sreekanth (2019900090)
