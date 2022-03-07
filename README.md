# udstyle

Compute complexity metrics from Universal Dependencies.

```
Usage: python3 udstyle.py [OPTIONS] FILE...
  --parse=LANG          parse texts with Stanza; provide 2 letter language code
  --output=FILENAME     write result to a tab-separated file.
  --persentence         report per sentence results, not mean per document
Reported metrics:
  - LEN:  mean sentence length in words (excluding punctuation).
  - MDD:  mean dependency distance (Gibson, 1998).
  - NDD:  normalized dependency distance (Lei & Jockers, 2018).
  - ADJ:  proportion of adjacent dependencies.
  - LEFT: dependency direction: proportion of left dependents.
  - MOD:  nominal modifiers (Biber & Gray, 2010).
  - CLS:  number of clauses per sentence.
  - CLL:  average clause length (clauses/words)
  - LXD:  lexical density: ratio of content words over total number of words
  - POS/DEP tag frequencies (only with --output)
Example:
$ python3 udstyle.py UD_Dutch-LassySmall/*.conllu
                 LEN    MDD    NDD    ADJ   LEFT    MOD    CLS    CLL    LXD
dev.conllu    14.182  2.461  0.926  0.500  0.459  0.052  2.223  9.190  0.603
test.conllu   11.434  2.192  0.807  0.547  0.412  0.074  1.771  9.013  0.657
train.conllu  11.027  2.172  0.775  0.564  0.391  0.072  1.863  8.107  0.645
```

## Also see

Simple readability metrics: https://github.com/andreasvc/readability/
