# udstyle

Compute complexity metrics from [Universal Dependencies](https://universaldependencies.org/).
Input maybe be `.conllu` files, or plain text files that will be parsed by
[Stanza](https://stanfordnlp.github.io/stanza/) when `--parse` is specified.
The language must be specified with `--lang`.

```
Usage: python3 udstyle.py [OPTIONS] FILE...
  --lang=LANG           two-letter language code (required)
  --parse               parse texts with Stanza
  --chunksize=N         divide input into chunks of about N words
  --lowfreq=ZIPF        low-frequency threshold (default: 3.0 large/3.5 small)
  --highfreq=ZIPF       high-frequency Zipf threshold (default: 4.0)
  --output=FILENAME     write result to a tab-separated file.
  --persentence         report per sentence results, not mean per document
Reported metrics:
  - NSENT:  number of sentences.
  - NWORDS: number of syntactic words (excluding punctuation).
  - LEN:    mean sentence length in words (excluding punctuation).
  - LENVAR: coefficient of variation of sentence lengths
  - WORDLEN: mean word length in characters.
  - MDD:    mean dependency distance (Gibson, 1998).
  - NDD:    normalized dependency distance (Lei & Jockers, 2018).
  - ADJD:   proportion of adjacent dependencies.
  - LEFT:   dependency direction: proportion of left dependents.
  - DEPTH:  maximum number of dependency edges between any token and the root.
  - AVGMDD: mean of the maximum dependency distance in each sentence.
  - MAXMDD: maximum dependency distance in the document.
  - MOD:    nominal modifiers (Biber & Gray, 2010).
  - CLS:    number of clauses per sentence.
  - CLL:    average clause length (words/predicate)
  - PRED:   number of verbal or copular predicates per sentence.
  - VBROOT: proportion of sentence roots that are verbs.
  - PREDROOT: proportion of roots that are verbs or copular predicates.
  - VBEDGES: mean number of dependents of lexical verbs.
  - VBEDGE*: distribution of lexical verbs by number of dependents.
  - SBJPRE/SBJPOST: position of nominal subjects relative to their heads.
  - OBJPRE/OBJPOST: position of objects relative to their heads.
  - LXD:    lexical density: ratio of content words over total number of words
  - MTLD:   Measure of Textual Lexical Diversity over content-word lemmas
  - LOWF:   proportion of content-word lemmas below the Zipf threshold
  - HIGHF:  proportion of content-word lemmas at or above the Zipf threshold
  - ZIPF:   mean Zipf frequency of known content-word lemmas
  - WFCOV:  proportion of content-word lemmas known to wordfreq
  - POS/DEP tag frequencies and distributions of verb features (only with --output)
Example:
$ python3 udstyle.py --lang=en --parse README.md
          README.conllu
NSENT            75.000
NWORDS          923.000
WORDLEN           5.568
LEN              12.307
LENVAR            0.698
MDD               2.279
NDD               0.890
ADJD              0.456
LEFT              0.653
DEPTH             3.613
AVGMDD            6.413
MAXMDD           48.000
MOD               0.093
CLS               2.187
CLL               9.230
PRED              1.333
VBROOT            0.453
PREDROOT          0.560
VBEDGES           1.888
SBJPRE            1.000
SBJPOST           0.000
OBJPRE            0.000
OBJPOST           1.000
LXD               0.653
MTLD             47.181
LOWF              0.199
HIGHF             0.654
ZIPF              4.430
WFCOV             0.901

$ python3 udstyle.py --lang=nl UD_Dutch-LassySmall/*.conllu
[...]
```

MTLD, LOWF, HIGHF, ZIPF, WFCOV, and LENVAR are document-level metrics. The
lexical metrics use lowercased lemmas of nouns, verbs, adjectives, and adverbs.
MTLD is reported only for texts with at least 50 such content words and uses a
type-token ratio threshold of 0.72. LOWF is the proportion of content-word
lemmas whose [wordfreq](https://github.com/rspeer/wordfreq) Zipf frequency is
below the threshold given with `--lowfreq`. Its default is 3.0 when wordfreq
has a large list for the language and 3.5 when only a small list is available.
HIGHF is the proportion at or above the threshold given with `--highfreq`
(4.0 by default).
ZIPF is the mean frequency of lemmas known to wordfreq, while WFCOV reports the
proportion with a known frequency. A wordfreq score of zero is treated as
unknown.

Full output also includes dynamic `feat=value` columns for values observed in
the UD morphological features of verbs. Each value is reported as a
proportion of tokens annotated for that feature. `{feat}Cov` reports the
proportion of `VERB` and `AUX` tokens carrying the corresponding feature;
tokens with the feature on another part of speech are also included as
possible hosts. This separates the observed distribution from annotation
coverage, which can differ substantially between UD languages and treebanks.
Dynamic `VBEDGE0`, `VBEDGE1`, ... columns give the proportion of lexical verbs
with the corresponding number of dependents. As with `VBEDGES`, only `VERB`
heads are included, including those with zero dependents; `aux`, `punct`, and
`cop` dependents are excluded.

MDD, NDD, AVGMDD, MAXMDD, ADJD, and LEFT consistently exclude `root`,
`punct`, `fixed`, `flat`, and `conj` relations. Document MDD and NDD are means
of their sentence-level values. ADJD and LEFT are pooled over eligible links;
MOD, LXD, subject order, and object order are pooled over their corresponding
word or relation populations. CLL is the number of words divided by the number
of verbal or copular predicates.

In the full output, only the base dependency labels are reported; i.e.,
subtypes such as `nsubj:pass` are collapsed into the base relation `nsubj`.
Punctuation is removed before analysis, so `PUNCT` and `punct` columns are not
reported.

With `--chunksize=N`, each input file is divided into consecutive chunks of
approximately N words, with boundaries rounded to the nearest sentence
boundary. Sentences are never split. Each chunk is analyzed as a separate
document and reported as `filename:1`, `filename:2`, and so on.

Note that wordfreq coverage differs by language. In particular, its smaller
frequency lists have a minimum represented Zipf frequency of 3.0, while
out-of-vocabulary words receive a frequency of 0. The adaptive LOWF default
allows known words near that floor to contribute, but LOWF still includes
unknown words. WFCOV shows how much vocabulary was covered; LOWF values based
on small and large lists should not be compared directly across languages.

## Installation

Install udstyle and its dependencies with:

```
pip install .
```

For development, including the test suite:

```
pip install -e '.[dev]'
pytest
```

To use `--parse`, you need [Stanza](https://stanfordnlp.github.io/stanza/):

```
pip install stanza
```

## References

If you use this code for research, please cite this repository.

Several metrics were inspired by Profiling-UD: https://aclanthology.org/2020.lrec-1.883/

Some metrics were inspired by https://github.com/SemHuis/genai_news/blob/main/analysis/article_metrics.py

Simple readability metrics: https://github.com/andreasvc/readability/
