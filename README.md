This is the code for **SetGNER: General Named Entity Recognition as Entity Set Generation** (EMNLP 2022).

## Prerequisites
    transformers>=3.4.0 
    pytorch>=1.6.0 

## Data
Put the data in this manner:
```text
    - data/
        - CADEC
            - train.txt
            - text.txt
            - dev.txt
        - Share_2013
            ...
        - en_ace04
        - en_ace05
        - conll2003
```

For discontinuous NER datasets `CADEC` and `Share_2013`, the data should be in the following form:
```text
Upset stomach and the feeling that I may need to throw up .
0,1 ADR|10,11 ADR

The pain in my stomach is in the middle of my stomach above my belly button and its a deep constant pain .
19,21 ADR|1,4 ADR
...
```

For nested NER datasets `en_ace04` and `en_ace05`, the data should be in the following form:
```text
{'ners': [[[0, 2, 'ORG'], [4, 4, 'GPE'], [10, 10, 'PER'], [10, 12, 'PER']]], 'sentences': [['Xinhua', 'News', 'Agency', ',', 'Shanghai', ',', 'August', '31st', ',', 'by', 'reporter', 'Jierong', 'Zhou']]}
{'ners': [[[0, 0, 'GPE'], [0, 4, 'PER'], [0, 5, 'PER'], [10, 10, 'GPE'], [16, 16, 'GPE'], [18, 18, 'GPE']]], 'sentences': [['Malaysian', 'vice', '-', 'prime', 'minister', 'Anwar', 'ended', 'a', 'visit', 'to', 'China', 'this', 'afternoon', ',', 'and', 'left', 'Shanghai', 'for', 'Tokyo', '.']]}
...
```

For flat NER dataset `conll2003`, the data should be in the following form:
```text
Japan B-LOC
began O
the O
defence O
of O
their O
Asian B-MISC
Cup I-MISC
```

## Running with CPU/Single GPU
Run the code by using
```shell
python train.py --dataset_name [Name-of-Dataset]
```

## Cite
If you find our work useful, please consider citing our paper:

```
@inproceedings{he-tang-2022-setgner,
    title = "SetGNER: General Named Entity Recognition as Entity Set Generation",
    author = "He, Yuxin  and
      Tang, Buzhou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.200",
}
```
