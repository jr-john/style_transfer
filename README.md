# Style Transfer in Text

Style transfer in text refers to the task of converting the style of one text to that of another text while preserving its content. The objective is to develop an approach that can take a source text as input and generate a new text in a target style.\
\
 This project has been developed as part of completion of the Neural Natural Language Generation course, IIIT-Hyderabad (2023).

## Repository Structure

```
.
├── baseline_outputs        # contains generation output for baseline model
├── control_tokens          # contains files used for the approach
├── evaluators              # contains custom metric evaluation scripts
├── scripts                 # contains training scripts
├── utils                   # contains additional scripts
├── environment.txt         # project level dependencies
└── README.md
```

### `baseline_outputs`

```
.
├── generation_em.txt       # baseline generation output for GYAFC-EM
└── generation_fr.txt       # baseline generation output for GYAFC-FR
```

### `control_tokens`

```
.
├── dataset                 # contains dataset and preparation script
└── outputs                 # contains generation output for this approach
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

## Checkpoints

- BART Baseline Checkpoints - [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/jerrin_thomas_research_iiit_ac_in/EWjfmq4VxR5IsjjS83H6cA0BSb85YvHkLiEKCewOU_V9CQ?e=RjCdp9)
- BERT Classifier Checkpoint - [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/jerrin_thomas_research_iiit_ac_in/ETS5MMsUchJInSzWyBd5b_ABYIW5oSoUdqTzyhKAX2XQPg?e=ClBh6M)
- BART+ControlTokens Baseline+ Checkpoints - [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/jerrin_thomas_research_iiit_ac_in/EdjwCl5oDT9LlRlzhqrFc44BJhSpYclv4a6ZZCwVaeb9yA?e=qDUY0A)

## Members

Padakanti Srijith (2019114002)\
Jerrin John Thomas (2019114012)\
Nikhil Bishnoi (2019114021)\
Suraj Kumar (2019900038)\
T Sreekanth (2019900090)
