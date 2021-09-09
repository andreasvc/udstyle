"""Compute complexity metrics from Universal Dependencies.

Usage: python3 udstyle.py [OPTIONS] FILE...
  --parse=LANG          parse texts with Stanza; provide 2 letter language code
  --output=FILENAME     write result to a tab-separated file.
Reported metrics:
  - LEN:  mean sentence length in words (excluding punctuation).
  - MDD:  mean dependency distance (Gibson, 1998).
  - NDD:  normalized dependency distance (Lei & Jockers, 2018).
  - ADJ:  proportion of adjacent dependencies.
  - LEFT: dependency direction: proportion of left dependents.
  - MOD:  nominal modifiers (Biber & Gray, 2010).
Example:
$ python3 udstyle.py UD_Dutch-LassySmall/*.conllu
                                  LEN    MDD    NDD    ADJ   LEFT    MOD
nl_lassysmall-ud-dev.conllu    14.182  2.461  0.926  0.500  0.459  0.052
nl_lassysmall-ud-test.conllu   11.434  2.192  0.807  0.547  0.412  0.074
nl_lassysmall-ud-train.conllu  11.027  2.172  0.775  0.564  0.391  0.072
"""
import os
import sys
import getopt
import subprocess
from math import log, sqrt
from contextlib import contextmanager
import pandas as pd
# TODO: extract POS and syntactic n-grams frequencies
# TODO: POS surprisal, requires training e.g. n-gram model on corpus

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


def which(program, exception=True):
	"""Return first match for program in search path.

	:param exception: By default, ValueError is raised when program not found.
		Pass False to return None in this case.
	"""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	if exception:
		raise ValueError('%r not found in path; please install it.' % program)


@contextmanager
def genericdecompressor(cmd, filename, encoding='utf8'):
	"""Run command line decompressor on file and return file object.

	:param cmd: executable in path with gzip-like command line interface;
		e.g., ``gzip, zstd, lz4, bzip2, lzop``
	:param filename: the file to decompress.
	:param encoding: if None, mode is binary; otherwise, text.
	:raises ValueError: if command returns an error.
	:returns: a file-like object that must be used in a with-statement;
		supports .read() and iteration, but not seeking."""
	with subprocess.Popen(
			[which(cmd), '--decompress', '--stdout', '--quiet', filename],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			encoding=encoding) as proc:
		# FIXME: should use select to avoid deadlocks due to OS pipe buffers
		# filling up and blocking the child process.
		yield proc.stdout
		retcode = proc.wait()
		if retcode:  # FIXME: retcode 2 means warning. allow warnings?
			raise ValueError('non-zero exit code %s from compressor %s:\n%r'
					% (retcode, cmd, proc.stderr.read()))


def openread(filename, encoding='utf8'):
	"""Open stdin/file for reading; decompress gz/lz4/zst files on-the-fly.

	:param encoding: if None, mode is binary; otherwise, text."""
	mode = 'rb' if encoding is None else 'rt'
	if filename == '-':  # TODO: decompress stdin on-the-fly
		return open(sys.stdin.fileno(), mode=mode, encoding=encoding)
	if not isinstance(filename, int):
		if filename.endswith('.gz'):
			return genericdecompressor('gzip', filename, encoding)
		elif filename.endswith('.zst'):
			return genericdecompressor('zstd', filename, encoding)
		elif filename.endswith('.lz4'):
			return genericdecompressor('lz4', filename, encoding)
	return open(filename, mode=mode, encoding=encoding)


def parsefiles(filenames, lang):
	"""Parse plain text files with Stanza if a corresponding
	.conllu file does not exist already."""
	nlp = None
	newfilenames = []
	for filename in filenames:
		conllu = '%s.conllu' % os.path.splitext(filename)[0]
		newfilenames.append(conllu)
		if (not os.path.exists(conllu)
				or os.stat(conllu).st_mtime < os.stat(filename).st_mtime):
			if nlp is None:
				import stanza
				from stanza.utils.conll import CoNLL
				processors = 'tokenize,mwt,pos,lemma,depparse'
				try:
					nlp = stanza.Pipeline(lang, processors=processors)
				except FileNotFoundError:
					stanza.download(lang)
					nlp = stanza.Pipeline(lang, processors=processors)
			with open(filename) as inp:
				doc = nlp(inp.read())
			# TODO: preserve paragraph breaks
			CoNLL.write_doc2conll(doc, conllu)
	return newfilenames


def conllureader(filename, excludepunct=False):
	"""Load corpus. Returns list of lists of lists:
	sentences[sentno][tokenno][fieldno]"""
	result = []
	sent = []
	with openread(filename) as inp:
		for line in inp:
			if line == '\n':
				if sent:
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
	if not result:
		raise ValueError('no sentences; not a valid .conllu file?')
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


def analyze(filename, excludepunct=True):
	"""Return a dict {featname: vector, ...} describing UD file.
	Each feature vector has a value for each sentence."""
	sentences = conllureader(filename, excludepunct=excludepunct)
	result = {}
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


def compare(filenames, parse=None, excludepunct=True):
	"""Collect statistics for multiple files.
	Returns a dataframe with one row per filename, with the mean score
	for each metric in the colmuns."""
	# This only reports the mean for each feature, you might want to look at
	# standard deviation and other aspects of the distribution.
	# This gives a macro average over the per-sentence scores.
	# TODO: should offer micro average as well.
	if parse:
		filenames = parsefiles(filenames, parse)
	return pd.DataFrame({
			os.path.basename(filename):
				pd.DataFrame(analyze(
					filename, excludepunct=excludepunct)).mean()
			for filename in filenames}).T


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(sys.argv[1:], '', ['export=', 'parse='])
		opts = dict(opts)
	except getopt.GetoptError:
		print(__doc__)
		return
	if not args:
		print(__doc__)
		return
	result = compare(args, opts.get('--parse'))
	if '--export' in opts:
		result.to_csv(opts.get('--export'), sep='\t')
	else:
		print(result.round(3))


if __name__ == '__main__':
	main()
