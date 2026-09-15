# udstyle

`udstyle` describes the linguistic style and complexity of texts using
[Universal Dependencies](https://universaldependencies.org/) (UD).
It can be used to explore groups of texts in terms of their grammatical
characteristics such as sentence length, syntactic structure, clause use, or
vocabulary.

The input can be UD-annotated `.conllu` files, or plain-text files that are
parsed with [Stanza](https://stanfordnlp.github.io/stanza/) when `--parse` is
specified. Specify a language with `--lang` to report word-frequency metrics.
A language is required when `--parse` is specified.

```
Usage: python3 udstyle.py [OPTIONS] FILE...
  --lang=LANG           language code for Stanza and wordfreq
  --parse               parse texts with Stanza
  --chunksize=N         divide input into chunks of about N words
  --lowfreq=ZIPF        low-frequency threshold (default: 3.0 large/3.5 small)
  --highfreq=ZIPF       high-frequency Zipf threshold (default: 4.0)
  --output=FILENAME     write result to a tab-separated file.
  --persentence         report per sentence results, not mean per document
```

## Understanding the metrics

No single metric is an overall score of language ability or complexity. The
metrics describe related but distinct properties, and are usually most useful
as a profile. A higher value is not inherently better or worse. For group
comparisons, inspect several measures from the same family and interpret them
in the context of the language, task, and population.

Except where noted, counts exclude punctuation. Proportions range from 0 to 1:
for example, `SUBORD = 0.30` means that 30% of the clauses identified by the
tool are subordinate.

### Amount of language and sentence length

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `NSENT` | Number of sentences. | Describes the amount of material. It is mainly a context or control variable, not a complexity measure. |
| `NWORDS` | Number of syntactic words. | Another measure of sample size. Very different word counts can make diversity and rare-event metrics less comparable. |
| `LEN` | Mean sentence length in words. | Higher values indicate longer sentences, which may reflect elaboration or syntactic complexity, but can also be affected by segmentation and run-on speech. |
| `LENVAR` | Variation in sentence length, calculated as the sample standard deviation divided by the mean. | Higher values indicate a less uniform mix of short and long sentences. It requires at least two sentences and is reported as missing otherwise. |
| `WORDLEN` | Mean word length in characters. | Higher values indicate longer word forms on average. This can reflect vocabulary choice and morphology, and should mainly be compared within the same language and transcription conventions. |

`NSENT` and `NWORDS` are totals per document or chunk. `LEN`, `DEPTH`,
`AVGMDD`, `CLS`, and `PRED` are first calculated for each sentence and then
averaged for document output.

### Dependency structure

A dependency links a word (the dependent) to the word that governs it (its
head). Distance measures count intervening token positions; tree depth counts
links in the grammatical hierarchy. They therefore capture different aspects
of syntactic structure.

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `MDD` | Mean dependency distance within a sentence, averaged across sentences ([Gibson, 1998](https://doi.org/10.1016/S0010-0277(98)00034-1)). | Higher values mean that grammatically related words tend to occur farther apart. This is often treated as greater syntactic integration cost, but it also rises with sentence length. |
| `NDD` | Dependency distance normalized for sentence length and root position ([Lei & Jockers, 2018](https://doi.org/10.1080/09296174.2018.1504615)). | Useful alongside `MDD` when groups differ in sentence length. It is a transformed distance measure, so interpret relative differences rather than as a number of words. |
| `ADJD` | Proportion of eligible dependencies whose words are adjacent. | Higher values indicate more local links. It often moves opposite to `MDD`, though the two are not exact inverses. |
| `LEFT` | Proportion of eligible dependents that occur before their heads. | Describes word-order direction rather than complexity. Interpret within a language; expected values differ greatly across languages and constructions. |
| `DEPTH` | Maximum token-to-root path length in each sentence, averaged across sentences. | Higher values indicate more hierarchical embedding. A sentence can have high depth without long linear dependencies, so compare it with `MDD`. |
| `AVGMDD` | Maximum eligible dependency distance in each sentence, averaged across sentences. | Captures the typical sentence's longest link and is more sensitive to occasional long dependencies than `MDD`. |
| `MAXMDD` | Largest eligible dependency distance anywhere in the document. | Captures an extreme case. It is sensitive to document length, parsing errors, and a single unusual sentence, so it is best treated as descriptive. |

The distance and direction metrics exclude `root`, `punct`, `fixed`, `flat`,
and `conj` relations. This keeps artificial-root links, punctuation, fixed
expressions, names, and coordination from dominating the result.

### Clauses, subordination, and predicates

These measures distinguish how many predications a text contains, how long
they are, and whether clauses are embedded. They are complementary: more
subordination does not necessarily imply longer clauses.

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `CLS` | A simple clause count per sentence, operationalized as one plus the number of `VERB` tokens. | Higher values suggest more clauses per sentence. Because this is a deliberately simple proxy, compare it with `PRED` and `SUBORD`, especially for copular or verbless clauses. |
| `PRED` | Number of lexical-verbal or copular predicate heads per sentence. | Higher values indicate more predications per sentence. Copular predicates such as *is tired* count even when the root is not tagged as a verb. |
| `CLL` | Number of words divided by the number of lexical-verbal or copular predicates. | Higher values mean more words per predicate, suggesting greater elaboration within clauses; lower values can result from many short clauses. |
| `SUBORD` | Proportion of identified clauses that are subordinate. | Higher values indicate greater reliance on embedded or dependent clauses. |
| `SUBDEPTH` | Mean maximum depth of subordinate-clause chains. | `0` means no subordinate chain, `1` means one level, and larger values indicate clauses embedded inside subordinate clauses. |
| `SUBPRE` / `SUBPOST` | Proportions of subordinate clause heads before/after their governors. | Describe placement rather than complexity. Where all relevant clauses can be ordered, the pair sums to 1. |
| `VBROOT` | Proportion of sentence roots tagged `VERB`. | Describes how often the main root is a lexical verb. It may be lower in languages or constructions where copular predicates have an adjective or noun as root. |
| `PREDROOT` | Proportion of roots that are lexical verbs or heads of copular predicates. | A broader predicate-root measure than `VBROOT`; the gap between them indicates how much copular analyses matter. |

Subordinate clauses are identified by UD relations `acl`, `advcl`, `ccomp`,
`csubj`, and `xcomp`. Coordinated clause heads inherit the status and depth of
the clause they coordinate with, without adding another embedding level.

### Argument order and verbal elaboration

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `SBJPRE` / `SBJPOST` | Proportions of nominal subjects before/after their heads. | Describe observed subject order. For example, `SBJPRE = 0.80` means 80% of `nsubj` dependents precede their heads. |
| `OBJPRE` / `OBJPOST` | Proportions of objects before/after their heads. | Describe observed object order. Interpret within a language and genre, not as a universal complexity scale. |
| `VBEDGES` | Mean number of dependents attached to lexical verbs, excluding `aux`, `punct`, and `cop`. | Higher values mean that verbs support more directly attached material on average. This can reflect argument or modifier elaboration. |
| `VBEDGE0`, `VBEDGE1`, ... | Proportion of lexical verbs with exactly 0, 1, ... included dependents. | Shows the distribution hidden by the mean. The reported `VBEDGE*` proportions sum to 1. |

The subject and object pairs may be missing when a text contains no matching
relation. They are pooled over all matching relations in a document, rather
than giving every sentence equal weight.

### Complexity inside noun phrases

These metrics target phrasal complexity, which can vary independently of
clausal complexity ([Biber & Gray, 2010](https://doi.org/10.1016/j.jeap.2010.01.001)).

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `NMOD` | Proportion of words attached with the UD relation `nmod`, including subtypes such as `nmod:poss`. | Higher values indicate more nominal modification relative to text length. |
| `NMODDEPTH` | Mean maximum number of recursive `nmod` links in each maximal modifier structure. | `0` means no such structure; higher values indicate more deeply nested nominal modification. Branching adds breadth but not necessarily depth. |
| `NCOMPRATE` | Proportion of `NOUN` and `PROPN` tokens participating in nominal `compound` structures. | Higher values indicate that more nominal vocabulary occurs in compounds. |
| `NCOMPSIZE` | Mean number of tokens in a maximal nominal compound. | `0` means no nominal compounds; higher values indicate longer compound structures. |

### Vocabulary use

These metrics use lowercased lemmas of nouns, verbs, adjectives, and adverbs;
`LXD` additionally counts proper nouns as content words. Lemmas reduce the
effect of inflection, so forms such as *talk*, *talks*, and *talked* normally
contribute as the same vocabulary item when the annotation is correct.

| Metric | What it measures | How to interpret it |
| --- | --- | --- |
| `LXD` | Proportion of words that are content words (`NOUN`, `PROPN`, `VERB`, `ADJ`, or `ADV`). | Higher values indicate denser lexical content; lower values indicate a larger share of function words. This is composition, not vocabulary rarity. |
| `MTLD` | Measure of Textual Lexical Diversity over content-word lemmas, using a type-token-ratio threshold of 0.72. | Higher values indicate that repetition accumulates more slowly and vocabulary is more diverse. It is less directly driven by text length than raw type-token ratio, but still requires adequate and comparable samples. |
| `LOWF` | Proportion of content-word lemmas below the selected word-frequency threshold. Unknown words are included as low-frequency. | Higher values indicate more rare or unrecognized vocabulary. Read it together with `WFCOV`, because names, transcription variants, and recognition failures can raise it. |
| `HIGHF` | Proportion of content-word lemmas at or above the high-frequency threshold. | Higher values indicate greater reliance on common vocabulary. `LOWF` and `HIGHF` need not sum to 1 when their thresholds leave a middle band. |
| `ZIPF` | Mean Zipf frequency of content-word lemmas known to `wordfreq`. | Higher values indicate more commonly used vocabulary on average. Unknown lemmas are excluded, so low coverage can bias this value. |
| `WFCOV` | Proportion of content-word lemmas with a nonzero `wordfreq` score. | Values near 1 mean frequency estimates cover most of the sample. Low values warn that `LOWF` and especially `ZIPF` need cautious interpretation. |

`MTLD` is reported only for texts with at least 50 content words. `LOWF`,
`HIGHF`, `ZIPF`, and `WFCOV` require a supported `--lang`; they are omitted
otherwise.

### Tag and morphological distributions

Tab-separated output also contains proportions for UD part-of-speech tags,
base dependency relations, and observed verb-feature values. These columns can
show *which* constructions underlie a group difference; for example, whether
one group uses more pronouns, auxiliaries, past tense, or passive subjects.
Dependency subtypes such as `nsubj:pass` are collapsed to `nsubj` in the base
relation columns.

For a feature such as `Tense`, columns such as `Tense=Past` give the
distribution among tokens annotated for that feature. `TenseCov` gives the
proportion of possible hosts that actually carry a `Tense` annotation. Always
inspect coverage before comparing feature distributions, because UD annotation
practices differ between languages and treebanks.

## Example output

```console
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

## Comparing groups of transcripts

A practical analysis usually treats one file as one observation (for example,
one participant's transcript), and writes all results to a table:

```console
$ python3 udstyle.py --lang=en --parse --output=metrics.tsv transcripts/*.txt
```

Before testing for a group difference:

1. Keep elicitation tasks, transcription rules, language, and parser settings
   the same across groups. Differences in interviewer prompts, sentence
   boundaries, filled pauses, or partial words can change the metrics.
2. Compare `NWORDS` and `NSENT`. If transcript lengths vary substantially,
   consider `--chunksize=N` to analyze similarly sized samples. Chunks stay
   within files and never split a sentence.
3. Examine related metrics together. For example, interpret `MDD` with `LEN`
   and `NDD`; `MTLD` with `NWORDS`; `LOWF` and `ZIPF` with `WFCOV`; and
   `SUBORD` with `SUBDEPTH`, `CLS`, and `PRED`.
4. Plot participant-level values and report uncertainty and effect sizes, not
   only group means or *p*-values. If each participant supplies several chunks
   or sessions, account for those repeated observations in the statistical
   model.
5. Inspect unexpected values in the source and parsed output. Automatic
   parsers are trained mostly on edited language and may analyze disfluencies,
   fragments, unusual word order, or clinical speech less reliably.

These measurements are descriptive features, not clinical assessments. A
group difference does not by itself identify its cause and should not be
interpreted as impairment without independent validation.


## Calculation details

`MTLD`, `LOWF`, `HIGHF`, `ZIPF`, `WFCOV`, and `LENVAR` are calculated over a
whole document or chunk. `LOWF` uses the threshold given with `--lowfreq`: by
default, 3.0 for languages with a large [wordfreq](https://github.com/rspeer/wordfreq)
list and 3.5 for those with only a small list. `HIGHF` uses `--highfreq` (4.0
by default). A wordfreq score of zero is treated as unknown.

When `--lang` is omitted, or wordfreq does not support the requested language,
LOWF, HIGHF, ZIPF, and WFCOV are omitted. Other metrics, including MTLD, remain
available. Unsupported wordfreq languages produce a warning. `--lang` accepts
the language codes understood by Stanza and wordfreq, including more specific
BCP 47 codes where supported.

For document output, `ADJD` and `LEFT` are pooled over eligible dependency
links. `SUBORD`, `NMOD`, `LXD`, subject order, and object order are likewise
pooled over the clauses, words, or relations they describe. `CLL` is total
words divided by total predicates. Pooling prevents a very short sentence from
receiving the same weight as a long sentence. `MDD` and `NDD`, in contrast,
are sentence means averaged across the document. `MAXMDD` is the maximum
rather than an average.

Punctuation is removed before analysis, so `PUNCT` and `punct` columns are not
reported. When a required structure is absent, a metric is either missing
(when a proportion has no denominator) or zero for the explicitly defined
depth and size measures `SUBDEPTH`, `NMODDEPTH`, and `NCOMPSIZE`.

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
