"""Compute complexity metrics from Universal Dependencies.

Usage: python3 udstyle.py <FILENAMES>
Example:
$ python3 udstyle.py UD_Dutch-LassySmall/*.conllu
                                  LEN    MDD    NDD    ADJ   LEFT    MOD
nl_lassysmall-ud-dev.conllu    14.182  2.461  0.926  0.500  0.459  0.052
nl_lassysmall-ud-test.conllu   11.434  2.192  0.807  0.547  0.412  0.074
nl_lassysmall-ud-train.conllu  11.027  2.172  0.775  0.564  0.391  0.072
"""
import os
import sys
from math import log, sqrt
import pandas
# TODO: optionally use StanfordNLP to parse texts for user

# Constants for field numbers:
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
# https://universaldependencies.org/format.html
# ID: Word index, integer starting at 1 for each new sentence; may be a range
#     for multiword tokens; may be a decimal number for empty nodes (decimal
#     numbers can be lower than 1 but must be greater than 0).
# FORM: Word form or punctuation symbol.
# LEMMA: Lemma or stem of word form.
# UPOS: Universal part-of-speech tag.
# XPOS: Language-specific part-of-speech tag; underscore if not available.
# FEATS: List of morphological features from the universal feature inventory or
#        from a defined language-specific extension; underscore if not
#        available.
# HEAD: Head of the current word, which is either a value of ID or zero (0).
# DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a
#         defined language-specific subtype of one.
# DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
# MISC: Any other annotation.


def conllureader(filename, excludepunct=False):
	"""Load corpus. Returns list of lists of lists:
	sentences[sentno][tokenno][fieldno]"""
	result = []
	sent = []
	with open(filename, encoding='utf8') as inp:
		for line in inp:
			if line == '\n':
				result.append(renumber(sent))
				sent = []
			elif line.startswith('#'):  # ignore all comments
				pass
			else:
				fields = line[:-1].split('\t')
				if '.' in fields[ID]:  # skip empty nodes
					continue
				elif excludepunct and fields[UPOS] == 'PUNCT':
					continue
				elif '-' in fields[ID]:  # multiword tokens
					fields[ID] = int(fields[ID][:fields[ID].index('-')])
				else:  # normal tokens
					fields[ID] = int(fields[ID])
				fields[HEAD] = int(fields[HEAD])
				sent.append(fields)
	return result


def renumber(sent):
	"""Fix non-contiguous IDs because of multiword tokens or removed tokens"""
	mapping = {line[ID]: n for n, line in enumerate(sent, 1)}
	mapping[0] = 0
	for line in sent:
		line[ID] = mapping[line[ID]]
		line[HEAD] = mapping[line[HEAD]]
	return sent


def mean(iterable):
	"""Arithmetic mean."""
	seq = list(iterable)  # accept generators
	return sum(seq) / len(seq)


def analyze(filename):
	"""Return a dict {featname: vector, ...} describing UD file.
	Each feature vector has a value for each sentence."""
	result = {}
	sentences = conllureader(filename, excludepunct=True)
	result['LEN'] = [len(sent) for sent in sentences]
	# Ignore certain relations, following Chen and Gerdes (2017, p. 57)
	# http://www.aclweb.org/anthology/W17-6508
	exclude = ('fixed', 'flat', 'conj', 'punct')
	# Gibson (1998) http://dx.doi.org/10.1016/S0010-0277(98)00034-1
	# Liu (2008)
	# http://cogsci.snu.ac.kr/jcs/index.php/issues/?mod=document&category1=Volume+9&uid=76
	# mean dependency distance
	result['MDD'] = [
			mean(abs(line[ID] - line[HEAD]) for line in sent
				if line[DEPREL] not in exclude)
			for sent in sentences]
	# Lei & Jockers (2018): https://doi.org/10.1080/09296174.2018.1504615
	# normalized dependency distance
	result['NDD'] = [
		abs(log(mdd
				/ sqrt(
					([line[DEPREL] for line in sent].index('root') + 1)
					* len(sent))))
		for mdd, sent in zip(result['MDD'], sentences)]
	# proportion of adjacent dependencies
	# https://doi.org/10.1016/j.langsci.2016.09.006
	result['ADJ'] = [mean(abs(line[ID] - line[HEAD]) == 1 for line in sent)
			for sent in sentences]
	# dependency direction: proportion of left dependents
	# http://www.aclweb.org/anthology/W17-6508
	result['LEFT'] = [mean(line[ID] < line[HEAD] for line in sent)
			for sent in sentences]
	# nominal modifiers;
	# attempt to measure phrasal complexity (as opposed to clausal complexity).
	# see e.g. https://doi.org/10.1016/j.jeap.2010.01.001
	result['MOD'] = [mean(line[DEPREL] == 'nmod' for line in sent)
			for sent in sentences]
	return result


def compare(filenames):
	"""Collect statistics for multiple files.
	Returns a dataframe with one row per filename, with the mean score
	for each metric in the colmuns."""
	# This only reports the mean for each feature, you might want to look at
	# standard deviation and other aspects of the distribution.
	# This gives a macro average over the per-sentence scores.
	# TODO: should offer micro average as well.
	return pandas.DataFrame({
			os.path.basename(filename):
				pandas.DataFrame(analyze(filename)).mean()
			for filename in filenames}).T


if __name__ == '__main__':
	print(compare(sys.argv[1:]).round(3))
	# to get a .tsv file:
	# print(compare(sys.argv[1:]).to_csv(sep='\t'))
