# udstyle

Compute complexity metrics from [Universal Dependencies](https://universaldependencies.org/).
Input maybe be `.conllu` files, or plain text files that will be parsed by
[Stanza](https://stanfordnlp.github.io/stanza/) when `--parse` is specified.
Specify a language with `--lang` to report wordfreq metrics. A language is
required when `--parse` is specified.

```
Usage: python3 udstyle.py [OPTIONS] FILE...
  --lang=LANG           language code for Stanza and wordfreq
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
  - SUBORD: proportion of clauses that are subordinate.
  - SUBDEPTH: mean maximum depth of subordinate-clause chains.
  - SUBPRE/SUBPOST: position of subordinate clauses relative to their heads.
  - SBJPRE/SBJPOST: position of nominal subjects relative to their heads.
  - OBJPRE/OBJPOST: position of objects relative to their heads.
  - CLS:    number of clauses per sentence.
  - CLL:    average clause length (words/predicate)
  - PRED:   number of verbal or copular predicates per sentence.
  - VBROOT: proportion of sentence roots that are verbs.
  - PREDROOT: proportion of roots that are verbs or copular predicates.
  - VBEDGES: mean number of dependents of lexical verbs.
  - VBEDGE*: distribution of lexical verbs by number of dependents.
  - NMOD:   nominal modifiers (Biber & Gray, 2010).
  - NMODDEPTH: mean maximum depth of nominal-modifier structures.
  - NCOMPRATE: proportion of nouns and proper nouns in nominal compounds.
  - NCOMPSIZE: mean size of nominal compound components.
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
NSENT            100.000
NWORDS          1143.000
WORDLEN            5.582
LEN               11.430
LENVAR             0.641
MDD                2.136
NDD                0.835
ADJD               0.464
LEFT               0.644
DEPTH              3.470
AVGMDD             5.790
MAXMDD            33.000
SUBORD             0.298
SUBDEPTH           1.059
SUBPRE             0.054
SUBPOST            0.946
CLS                2.080
CLL                9.218
NMOD               0.095
NMODDEPTH          1.181
NCOMPRATE          0.344
NCOMPSIZE          2.278
PRED               1.240
VBROOT             0.410
PREDROOT           0.510
VBEDGES            1.870
SBJPRE             1.000
SBJPOST            0.000
OBJPRE             0.000
OBJPOST            1.000
LXD                0.649
MTLD              50.048
LOWF               0.200
HIGHF              0.626
ZIPF               4.413
WFCOV              0.892

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

When `--lang` is omitted, or wordfreq does not support the requested language,
LOWF, HIGHF, ZIPF, and WFCOV are omitted. Other metrics, including MTLD, remain
available. Unsupported wordfreq languages produce a warning. `--lang` accepts
the language codes understood by Stanza and wordfreq, including more specific
BCP 47 codes where supported.

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
NMOD, LXD, subject order, and object order are pooled over their corresponding
word or relation populations. CLL is the number of words divided by the number
of verbal or copular predicates.

In the full output, only the base dependency labels are reported; i.e.,
subtypes such as `nsubj:pass` are collapsed into the base relation `nsubj`.
Punctuation is removed before analysis, so `PUNCT` and `punct` columns are not
reported.

NMOD is the proportion of words attached with a base `nmod` relation.
NMODDEPTH averages the maximum number of recursive `nmod` edges in each
maximal nominal-modifier structure; relation subtypes such as `nmod:poss` are
included. Nominal coordination connects branches without increasing depth.
NCOMPRATE is the proportion of `NOUN` and `PROPN` tokens participating in a
nominal `compound` component, and NCOMPSIZE is the mean number of tokens in
those maximal components. Other parts of speech and non-nominal compounds are
excluded. Metrics whose structures are absent have depth or size 0.

Subordinate clauses are identified by the base relations `acl`, `advcl`,
`ccomp`, `csubj`, and `xcomp`. Coordinated clause heads inherit the status and
depth of the clause they are coordinated with without adding an embedding
level. SUBORD is pooled over clauses, SUBPRE and SUBPOST are pooled over
subordinate clauses, and SUBDEPTH averages maximal top-level subordinate-chain
depths. A document without subordinate clauses has SUBDEPTH 0.

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
