"""Compute complexity metrics from Universal Dependencies.

Usage: python3 udstyle.py [OPTIONS] FILE...
  --lang=LANG           two-letter language code (required)
  --parse               parse texts with Stanza
  --chunksize=N         divide input into chunks of about N words
  --lowfreq=ZIPF        low-frequency threshold (default: 3.0 large/3.5 small)
  --highfreq=ZIPF       high-frequency Zipf threshold (default: 4.0)
  --output=FILENAME     write result to a tab-separated file.
  --persentence         report per sentence results, not mean per document

Examples:
$ python3 udstyle.py --lang=en --parse README.md
$ python3 udstyle.py --lang=nl UD_Dutch-LassySmall/*.conllu

See README.md for explanation of the reported metrics.
"""
import getopt
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from math import log, sqrt

import pandas as pd
from wordfreq import available_languages, zipf_frequency

# TODO: extract POS and syntactic n-grams frequencies
# TODO: POS surprisal, requires training e.g. n-gram model on corpus

EMPTY = {'', '_'}
DISTEXCLUDE = {'fixed', 'flat', 'conj', 'punct', 'root'}
SKIPEDGE = {'aux', 'punct', 'cop'}
SUBORDRELATIONS = {'acl', 'advcl', 'ccomp', 'csubj', 'xcomp'}
CONTENTPOS = {'ADJ', 'ADV', 'NOUN', 'VERB'}
CONTENTPROPN = {'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB'}
NOMINALPOS = {'NOUN', 'PROPN'}
VERBPOS = {'VERB', 'AUX'}
MTLDMINWORDS, MTLDTHRESHOLD = 50, 0.72
LOWFREQ, LOWFREQSMALL, HIGHFREQ = 3.0, 3.5, 4.0
# Constants for UD field numbers:
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

# All UD 2.0 POS tags
POSTAGS = [
		'ADJ',  # adjective
		'ADP',  # adposition
		'ADV',  # adverb
		'AUX',  # auxiliary
		'CCONJ',  # coordinating conjunction
		'DET',  # determiner
		'INTJ',  # interjection
		'NOUN',  # noun
		'NUM',  # numeral
		'PART',  # particle
		'PRON',  # pronoun
		'PROPN',  # proper noun
		'SCONJ',  # subordinating conjunction
		'SYM',  # symbol
		'VERB',  # verb
		'X',  # other
		]
# Selected verb features that are reported
VERBFEATS = [
		'VerbForm',  # form of verb or deverbative (finite, infinitive, participle, gerund, ..)
		'Mood',  # expresses modality and subclassifies finite verb forms.
		'Tense',  # specifies the time when the action took / takes / will take place, in relation to a reference point
		'Aspect',  # specifies duration of the action in time, whether the action has been completed etc.
		'Voice',  # For Indo-European speakers, voice means mainly the active-passive distinction. In other languages, other shades of verb meaning are categorized as voice.
		]
# UD 2.0 base dependency tags, excluding punctuation
DEPTAGS = [
		'acl',  # clausal modifier of noun (adnominal clause)
		'advcl',  # adverbial clause modifier
		'advmod',  # adverbial modifier
		'amod',  # adjectival modifier
		'appos',  # appositional modifier
		'aux',  # auxiliary
		'case',  # case marking
		'cc',  # coordinating conjunction
		'ccomp',  # clausal complement
		'clf',  # classifier
		'compound',  # compound
		'conj',  # conjunct
		'cop',  # copula
		'csubj',  # clausal subject
		'dep',  # unspecified dependency
		'det',  # determiner
		'discourse',  # discourse element
		'dislocated',  # dislocated elements
		'expl',  # expletive
		'fixed',  # fixed multiword expression
		'flat',  # flat multiword expression
		'goeswith',  # goes with
		'iobj',  # indirect object
		'list',  # list
		'mark',  # marker
		'nmod',  # nominal modifier
		'nsubj',  # nominal subject
		'nummod',  # numeric modifier
		'obj',  # object
		'obl',  # oblique nominal
		'orphan',  # orphan
		'parataxis',  # parataxis
		'reparandum',  # overridden disfluency
		'root',  # root
		'vocative',  # vocative
		'xcomp',  # open clausal complement
		]


def which(program, exception=True):
	"""Return first match for program in search path.

	:param exception: By default, ValueError is raised when program not found.
		Pass False to return None in this case."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	if exception:
		raise ValueError(f'{program} not found in path; please install it.')


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
			raise ValueError(f'non-zero exit code {retcode} from '
					f'compressor {cmd}:\n{proc.stderr.read()}')


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
	"""Parse UTF-8 encoded plain text files with Stanza if a corresponding
	.conllu file does not exist already."""
	nlp = None
	newfilenames = []
	for filename in filenames:
		conllu = f'{os.path.splitext(filename)[0]}.conllu'
		newfilenames.append(conllu)
		if (not os.path.exists(conllu)
				or os.stat(conllu).st_mtime < os.stat(filename).st_mtime):
			if nlp is None:
				import stanza
				from stanza.utils.conll import CoNLL
				processors = 'tokenize,pos,lemma,depparse'
				try:
					nlp = stanza.Pipeline(lang, processors=processors)
				except FileNotFoundError:
					stanza.download(lang)
					nlp = stanza.Pipeline(lang, processors=processors)
			with open(filename, encoding='utf8') as inp:
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
					try:
						result.append(renumber(sent))
					except KeyError:
						pass
					sent = []
			elif line.startswith('#'):  # ignore all comments
				pass
			else:
				fields = line[:-1].split('\t')
				if ('.' in fields[ID]  # skip empty nodes
						or (excludepunct and fields[UPOS] == 'PUNCT')):
					continue
				elif '-' in fields[ID]:  # multiword tokens
					fields[ID] = int(fields[ID][:fields[ID].index('-')])
				else:  # normal tokens
					fields[ID] = int(fields[ID])
				try:
					fields[HEAD] = int(fields[HEAD])
				except ValueError:
					continue
				sent.append(fields)
	if not result:
		raise ValueError('no sentences; not a valid .conllu file?')
	return result


def chunksents(basename, sentences, chunksize):
	"""Group consecutive sentences into chunks close to `chunksize` words."""
	if chunksize <= 0:
		raise ValueError('chunksize must be a positive integer')
	chunks, chunk, nwords = [], [], 0
	for sent in sentences:
		sentlen = len(sent)
		if (chunk and nwords < chunksize < nwords + sentlen
				and chunksize - nwords < nwords + sentlen - chunksize):
			chunks.append((f'{basename}:{len(chunks)}', chunk))
			chunk, nwords = [], 0
		chunk.append(sent)
		nwords += sentlen
		if nwords >= chunksize:
			chunks.append((f'{basename}:{len(chunks)}', chunk))
			chunk, nwords = [], 0
	if chunk:
		chunks.append((f'{basename}:{len(chunks)}', chunk))
	return chunks


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


def defaultlowfreq(lang):
	"""Return the low-frequency threshold for wordfreq's best language list."""
	if lang not in available_languages('best'):
		raise ValueError(
				f'{lang!r} is not a canonical language code supported by wordfreq')
	return LOWFREQ if lang in available_languages('large') else LOWFREQSMALL


def getmtld(words, threshold=MTLDTHRESHOLD):
	"""Return the Measure of Textual Lexical Diversity for a token list."""
	# Adapted from LexicalRichness 0.5.1's MIT-licensed mtld method:
	# https://github.com/LSYS/LexicalRichness
	def measure(sequence):
		terms = set()
		segment_words = 0
		factors = 0
		for word in sequence:
			segment_words += 1
			terms.add(word)
			ttr = len(terms) / segment_words
			if ttr <= threshold:
				terms = set()
				segment_words = 0
				factors += 1
		if segment_words:
			factors += (1 - ttr) / (1 - threshold)
		if not factors:
			# An all-unique sequence is conventionally treated as one factor.
			factors = 1
		return len(words) / factors

	return (measure(words) + measure(reversed(words))) / 2


def dependencydepth(sentence):
	"""Return the maximum dependency-tree depth in a sentence."""
	heads = {line[ID]: line[HEAD] for line in sentence}
	maxdepth = 0
	for token in sentence:
		current = token[ID]
		visited = set()
		depth = 0
		while heads.get(current, 0) and current not in visited:
			visited.add(current)
			current = heads[current]
			depth += 1
		maxdepth = max(maxdepth, depth)
	return maxdepth


def baserel(token):
	"""Return the universal base relation of a UD token."""
	return token[DEPREL].split(':', 1)[0]


def dependencylinks(sentence):
	"""Return links included by all dependency-distance metrics."""
	return [token for token in sentence if baserel(token) not in DISTEXCLUDE]


def predicatecount(sentence):
	"""Count lexical-verbal and copular predicate heads."""
	return sum(
			token[UPOS] == 'VERB' or any(
					child[HEAD] == token[ID] and baserel(child) == 'cop'
					for child in sentence)
			for token in sentence)


def clauseanalysis(sentence):
	"""Return clause heads, subordinate depths, governors, and chain depths."""
	byid = {token[ID]: token for token in sentence}
	clauses = {token[ID] for token in sentence
			if (token[UPOS] == 'VERB'
				or baserel(token) in SUBORDRELATIONS
				or any(child[HEAD] == token[ID] and baserel(child) == 'cop'
					for child in sentence))}

	def parentclause(token):
		current = byid.get(token[HEAD])
		visited = set()
		while current is not None and current[ID] not in visited:
			if current[ID] in clauses:
				return current
			visited.add(current[ID])
			current = byid.get(current[HEAD])
		return None

	depths, governors = {}, {}
	def getdepth(tokenid, trail=None):
		if tokenid in depths:
			return depths[tokenid]
		trail = set() if trail is None else trail
		if tokenid in trail:
			return 0
		trail.add(tokenid)
		token = byid[tokenid]
		parent = parentclause(token)
		if baserel(token) == 'conj' and parent is not None:
			depth = getdepth(parent[ID], trail)
			governors[tokenid] = governors.get(parent[ID], parent[HEAD])
		elif baserel(token) in SUBORDRELATIONS:
			depth = (getdepth(parent[ID], trail) if parent is not None else 0) + 1
			governors[tokenid] = token[HEAD]
		else:
			depth = 0
		depths[tokenid] = depth
		trail.remove(tokenid)
		return depth

	for tokenid in clauses:
		getdepth(tokenid)

	chainroots = {}
	def getchainroot(tokenid, trail=None):
		if tokenid in chainroots:
			return chainroots[tokenid]
		trail = set() if trail is None else trail
		if tokenid in trail:
			return tokenid
		trail.add(tokenid)
		parent = parentclause(byid[tokenid])
		if parent is not None and depths.get(parent[ID], 0) > 0:
			root = getchainroot(parent[ID], trail)
		else:
			root = tokenid
		chainroots[tokenid] = root
		trail.remove(tokenid)
		return root

	chains = {}
	for tokenid, depth in depths.items():
		if depth:
			root = getchainroot(tokenid)
			chains[root] = max(chains.get(root, 0), depth)
	return clauses, depths, governors, list(chains.values())


def verbarities(sentence):
	"""Return arities of lexical verbs."""
	verbalheads = [line for line in sentence if line[UPOS] == 'VERB']
	return [sum(child[HEAD] == head[ID] and baserel(child) not in SKIPEDGE
			for child in sentence)
				for head in verbalheads]


def nominalmetrics(sentence):
	"""Return nmod depths and nominal-compound membership and sizes."""
	byid = {token[ID]: token for token in sentence}
	# nmod contributes one level; nominal coordination connects branches
	# without itself increasing their depth.
	edges = [(token[HEAD], token[ID], 1) for token in sentence
			if baserel(token) == 'nmod' and token[HEAD] in byid]
	edges.extend((token[HEAD], token[ID], 0) for token in sentence
			if (baserel(token) == 'conj' and token[UPOS] in NOMINALPOS
				and token[HEAD] in byid
				and byid[token[HEAD]][UPOS] in NOMINALPOS))
	nmodnodes = {node for head, node, weight in edges if weight
			for node in (head, node)}
	adjacency = {}
	for head, dependent, weight in edges:
		adjacency.setdefault(head, []).append((dependent, weight))
		adjacency.setdefault(dependent, [])

	def maxdepth(node, trail=None):
		trail = set() if trail is None else trail
		if node in trail:
			return 0
		trail.add(node)
		depth = max((weight + maxdepth(child, trail)
				for child, weight in adjacency.get(node, ())), default=0)
		trail.remove(node)
		return depth

	# Weak components give one maximal value for branching structures.
	undirected = {}
	for head, dependent, _ in edges:
		undirected.setdefault(head, set()).add(dependent)
		undirected.setdefault(dependent, set()).add(head)
	depths, seen = [], set()
	for start in nmodnodes:
		if start in seen:
			continue
		component, stack = set(), [start]
		while stack:
			node = stack.pop()
			if node not in component:
				component.add(node)
				stack.extend(undirected.get(node, ()))
		seen.update(component)
		dependents = {dependent for head, dependent, _ in edges
				if head in component and dependent in component}
		roots = component - dependents or component
		depths.append(max(maxdepth(root) for root in roots))

	# Nominal compounds are maximal undirected components of compound links.
	compoundgraph = {}
	for token in sentence:
		head = byid.get(token[HEAD])
		if (baserel(token) == 'compound' and head is not None
				and token[UPOS] in NOMINALPOS and head[UPOS] in NOMINALPOS):
			compoundgraph.setdefault(token[ID], set()).add(head[ID])
			compoundgraph.setdefault(head[ID], set()).add(token[ID])
	compounds, participants = [], set()
	for start in compoundgraph:
		if start in participants:
			continue
		component, stack = set(), [start]
		while stack:
			node = stack.pop()
			if node not in component:
				component.add(node)
				stack.extend(compoundgraph.get(node, ()))
		participants.update(component)
		compounds.append(len(component))
	return depths, participants, compounds


def documentmetrics(sentences, lang, lowfreq, highfreq):
	"""Return metrics defined over a whole document or chunk."""
	result = {}
	lengths = [len(sent) for sent in sentences]
	# The following metrics are adapted from
	# https://github.com/SemHuis/genai_news/blob/main/analysis/article_metrics.py
	result['LENVAR'] = None
	if len(lengths) >= 2:
		avglen = mean(lengths)
		variance = sum((length - avglen) ** 2 for length in lengths) / (
				len(lengths) - 1)
		result['LENVAR'] = sqrt(variance) / avglen if avglen else None
	lemmas = [line[LEMMA].lower() for sent in sentences for line in sent
			if line[UPOS] in CONTENTPOS and line[LEMMA] not in EMPTY]
	result['MTLD'] = getmtld(lemmas) if len(lemmas) >= MTLDMINWORDS else None
	frequencies = [zipf_frequency(lemma, lang) for lemma in lemmas]
	result['LOWF'] = (mean(frequency < lowfreq for frequency in frequencies)
			if frequencies else None)
	result['HIGHF'] = (mean(frequency >= highfreq for frequency in frequencies)
			if frequencies else None)
	known = [frequency for frequency in frequencies if frequency > 0]
	result['ZIPF'] = mean(known) if known else None
	result['WFCOV'] = (mean(frequency > 0 for frequency in frequencies)
			if frequencies else None)
	result['NSENT'] = len(sentences)
	tokens = [line for sent in sentences for line in sent]
	wordforms = [line[FORM] for line in tokens if line[UPOS] != 'PUNCT']
	result['NWORDS'] = len(wordforms)
	result['WORDLEN'] = mean(map(len, wordforms)) if wordforms else None
	return result


def sentencemetrics(sentences):
	"""Return dict of complexity metrics with results for each sentence."""
	result = {}
	result['LEN'] = [len(sent) for sent in sentences]
	# Gibson (1998) http://dx.doi.org/10.1016/S0010-0277(98)00034-1
	# Liu (2008) https://hdl.handle.net/10371/70907
	# mean dependency distance
	links = [dependencylinks(sent) for sent in sentences]
	result['MDD'] = [mean(abs(line[ID] - line[HEAD]) for line in sentlinks)
			if sentlinks else None for sentlinks in links]
	# Lei & Jockers (2018): https://doi.org/10.1080/09296174.2018.1504615
	# normalized dependency distance
	result['NDD'] = []
	for mdd, sent in zip(result['MDD'], sentences):
		roots = [line for line in sent if baserel(line) == 'root']
		result['NDD'].append(abs(log(mdd / sqrt(roots[0][ID] * len(sent))))
				if mdd is not None and roots else None)
	# proportion of adjacent dependencies
	# https://doi.org/10.1016/j.langsci.2016.09.006
	result['ADJD'] = [mean(abs(line[ID] - line[HEAD]) == 1
			for line in sentlinks) if sentlinks else None for sentlinks in links]
	# dependency direction: proportion of left dependents
	# http://www.aclweb.org/anthology/W17-6508
	result['LEFT'] = [mean(line[ID] < line[HEAD] for line in sentlinks)
			if sentlinks else None for sentlinks in links]
	# Maximum number of dependency edges between any token and the root.
	result['DEPTH'] = [dependencydepth(sent) for sent in sentences]
	# Maximum dependency spans use the same links as MDD and NDD.
	dependencydistances = [[abs(line[ID] - line[HEAD])
			for line in sentlinks] for sentlinks in links]
	result['AVGMDD'] = [max(distances, default=0)
			for distances in dependencydistances]
	# MAXMDD has the same sentence-level values as AVGMDD, but is aggregated
	# with max() instead of mean() at document level.
	result['MAXMDD'] = result['AVGMDD']
	# Metrics for subordinates, subject/object position, and verb edges
	result.update((metric, []) for metric
			in ['SUBORD', 'SUBDEPTH', 'SUBPRE', 'SUBPOST', 'SBJPRE', 'SBJPOST',
			'OBJPRE', 'OBJPOST', 'VBEDGES'])
	for sent in sentences:
		arities = verbarities(sent)
		result['VBEDGES'].append(mean(arities) if arities else None)
		for relation, before, after in (
				('nsubj', 'SBJPRE', 'SBJPOST'),
				('obj', 'OBJPRE', 'OBJPOST')):
			dependents = [line for line in sent if baserel(line) == relation]
			result[before].append(mean(line[ID] < line[HEAD]
					for line in dependents) if dependents else None)
			result[after].append(mean(line[ID] > line[HEAD]
					for line in dependents) if dependents else None)
		clauses, depths, governors, chains = clauseanalysis(sent)
		subordinates = [tokenid for tokenid in clauses if depths[tokenid] > 0]
		result['SUBORD'].append(
				len(subordinates) / len(clauses) if clauses else None)
		positions = [(tokenid, governors.get(tokenid))
				for tokenid in subordinates if governors.get(tokenid) in {
						token[ID] for token in sent}]
		result['SUBPRE'].append(mean(tokenid < governor
				for tokenid, governor in positions) if positions else None)
		result['SUBPOST'].append(mean(tokenid > governor
				for tokenid, governor in positions) if positions else None)
		result['SUBDEPTH'].append(mean(chains) if chains else 0)
	# number of clauses per sentence; https://doi.org/10.1007/s11145-007-9107-5
	result['CLS'] = [1 + sum(line[UPOS] == 'VERB' for line in sent)
			for sent in sentences]
	# avg clause len (words/predicate) https://aclanthology.org/2020.lrec-1.883
	predicates = [predicatecount(sent) for sent in sentences]
	result['CLL'] = [len(sent) / count if count else None
			for sent, count in zip(sentences, predicates)]
	result['PRED'] = predicates
	roots = [[line for line in sent if line[HEAD] == 0] for sent in sentences]
	result['VBROOT'] = [mean(line[UPOS] == 'VERB' for line in sentroots)
			for sentroots in roots]
	result['PREDROOT'] = [mean(
			line[UPOS] == 'VERB' or any(
					child[HEAD] == line[ID] and baserel(child) == 'cop'
					for child in sent)
			for line in sentroots)
			for sent, sentroots in zip(sentences, roots)]
	# nominal modifiers;
	# attempt to measure phrasal complexity (as opposed to clausal complexity).
	# see e.g. https://doi.org/10.1016/j.jeap.2010.01.001
	result['NMOD'] = [mean(baserel(line) == 'nmod' for line in sent)
			for sent in sentences]
	nominal = [nominalmetrics(sent) for sent in sentences]
	result['NMODDEPTH'] = [mean(depths) if depths else 0
			for depths, _, _ in nominal]
	result['NCOMPRATE'] = [len(participants) / nnouns if nnouns else None
			for sent, (_, participants, _) in zip(sentences, nominal)
			for nnouns in [sum(token[UPOS] in NOMINALPOS for token in sent)]]
	result['NCOMPSIZE'] = [mean(sizes) if sizes else 0
			for _, _, sizes in nominal]
	# lexical density: ratio of content words over total number of words
	# https://aclanthology.org/2020.lrec-1.883
	result['LXD'] = [sum(line[UPOS] in CONTENTPROPN for line in sent) / len(sent)
			for sent in sentences]
	return result


def countmorph(sentences, feature, postags):
	"""Return a feature value distribution and coverage."""
	counts = Counter()
	hosts = annotated = 0
	for sentence in sentences:
		for token in sentence:
			values = []
			if token[FEATS] not in EMPTY:
				for item in token[FEATS].split('|'):
					name, separator, value = item.partition('=')
					if separator and name == feature:
						values.extend(value.split(','))
			if token[UPOS] in postags or values:
				hosts += 1
			if values:
				annotated += 1
				counts.update(values)
	result = {f'{feature}={value}': count / annotated
			for value, count in counts.items()}
	result[f'{feature}Cov'] = (annotated / hosts) if hosts else None
	return result


def counttags(sentences):
	"""Count POS, base dependency, and selected morphology features."""
	numtokens = sum(len(sent) for sent in sentences)
	postags = Counter(line[UPOS] for sent in sentences for line in sent)
	deptags = Counter(baserel(line) for sent in sentences for line in sent)
	tags = {a: postags[a] / numtokens for a in POSTAGS}
	tags.update({a: deptags[a] / numtokens for a in DEPTAGS})
	for feat in VERBFEATS:
		tags.update(countmorph(sentences, feat, VERBPOS))
	return tags


def summarizedmetrics(sentences, lang, lowfreq, highfreq):
	"""Return one scalar score per metric for a document or chunk."""
	metrics = sentencemetrics(sentences)
	result = {}
	for name, values in metrics.items():
		values = [value for value in values if value is not None]
		result[name] = ((max(values) if name == 'MAXMDD' else mean(values))
				if values else None)
	# Pool proportions over the observations they describe instead of giving
	# every sentence equal weight.
	tokens = [line for sentence in sentences for line in sentence]
	links = [line for sentence in sentences for line in dependencylinks(sentence)]
	result['ADJD'] = (mean(abs(line[ID] - line[HEAD]) == 1 for line in links)
			if links else None)
	result['LEFT'] = (mean(line[ID] < line[HEAD] for line in links)
			if links else None)
	npredicates = sum(predicatecount(sentence) for sentence in sentences)
	result['CLL'] = len(tokens) / npredicates if npredicates else None
	analyses = [clauseanalysis(sentence) for sentence in sentences]
	nclauses = sum(len(clauses) for clauses, _, _, _ in analyses)
	nsubordinates = sum(sum(depth > 0 for depth in depths.values())
			for _, depths, _, _ in analyses)
	result['SUBORD'] = nsubordinates / nclauses if nclauses else None
	chains = [depth for _, _, _, sentencechains in analyses
			for depth in sentencechains]
	result['SUBDEPTH'] = mean(chains) if chains else 0
	positions = [(tokenid, governor)
			for clauses, depths, governors, _ in analyses
			for tokenid in clauses if depths[tokenid] > 0
			for governor in [governors.get(tokenid)] if governor is not None]
	result['SUBPRE'] = (mean(tokenid < governor
			for tokenid, governor in positions) if positions else None)
	result['SUBPOST'] = (mean(tokenid > governor
			for tokenid, governor in positions) if positions else None)
	for relation, before, after in (
			('nsubj', 'SBJPRE', 'SBJPOST'), ('obj', 'OBJPRE', 'OBJPOST')):
		dependents = [line for line in tokens if baserel(line) == relation]
		result[before] = (mean(line[ID] < line[HEAD] for line in dependents)
				if dependents else None)
		result[after] = (mean(line[ID] > line[HEAD] for line in dependents)
				if dependents else None)
	arities = [arity for sentence in sentences for arity in verbarities(sentence)]
	if arities:
		# Use the same verb-weighted population for the mean and distribution.
		result['VBEDGES'] = mean(arities)
		aritycounts = Counter(arities)
		result.update({f'VBEDGE{arity}': count / len(arities)
				for arity, count in sorted(aritycounts.items())})
	result['NMOD'] = (mean(baserel(line) == 'nmod' for line in tokens)
			if tokens else None)
	nominal = [nominalmetrics(sentence) for sentence in sentences]
	depths = [depth for sentdepths, _, _ in nominal for depth in sentdepths]
	result['NMODDEPTH'] = mean(depths) if depths else 0
	participants = sum(len(sentparticipants)
			for _, sentparticipants, _ in nominal)
	nnouns = sum(token[UPOS] in NOMINALPOS for token in tokens)
	result['NCOMPRATE'] = participants / nnouns if nnouns else None
	compounds = [size for _, _, sentsizes in nominal for size in sentsizes]
	result['NCOMPSIZE'] = mean(compounds) if compounds else 0
	result['LXD'] = (mean(line[UPOS] in CONTENTPROPN for line in tokens)
			if tokens else None)
	result.update(documentmetrics(sentences, lang, lowfreq, highfreq))
	result.update(counttags(sentences))
	return result


def analyze(filename, lang, lowfreq, highfreq,
		excludepunct=True, persentence=False, chunksize=None):
	"""Return a dataframe describing a UD file, optionally in chunks."""
	sentences = conllureader(filename, excludepunct=excludepunct)
	if persentence:
		result = sentencemetrics(sentences)
		result['sent'] = [' '.join(line[FORM] for line in sent)
				for sent in sentences]
		result['filename'] = filename
		return pd.DataFrame(result)
	basename = os.path.basename(filename)
	chunks = ([(basename, sentences)] if chunksize is None
			else chunksents(basename, sentences, chunksize))
	return pd.DataFrame({name: summarizedmetrics(chunk, lang, lowfreq, highfreq)
			for name, chunk in chunks}).T


def compare(filenames, lang, parse=False, excludepunct=True,
		persentence=False, chunksize=None, lowfreq=None,
		highfreq=HIGHFREQ):
	"""Collect statistics for multiple files."""
	if lowfreq is None:
		lowfreq = defaultlowfreq(lang)
	if parse:
		filenames = parsefiles(filenames, lang)
	if persentence and chunksize is not None:
		raise ValueError('--chunksize cannot be combined with --persentence')
	return pd.concat([
			analyze(filename, lang, lowfreq, highfreq,
				excludepunct=excludepunct, persentence=persentence,
				chunksize=chunksize)
			for filename in filenames], ignore_index=persentence)


def main():
	"""CLI."""
	try:
		opts, args = getopt.gnu_getopt(
				sys.argv[1:], '',
				['parse', 'persentence', 'output=', 'lang=', 'chunksize=',
				'lowfreq=', 'highfreq='])
		opts = dict(opts)
	except getopt.GetoptError:
		print(__doc__)
		return
	if not args:
		print(__doc__)
		return
	if '--lang' not in opts:
		print('--lang=LANG is required', file=sys.stderr)
		return
	try:
		chunksize = (int(opts['--chunksize'])
				if '--chunksize' in opts else None)
		if chunksize is not None and chunksize <= 0:
			raise ValueError
	except ValueError:
		print('--chunksize must be a positive integer', file=sys.stderr)
		return
	lowfreq = None
	if '--lowfreq' in opts:
		try:
			lowfreq = float(opts['--lowfreq'])
		except ValueError:
			print('--lowfreq must be a number', file=sys.stderr)
			return
	try:
		highfreq = float(opts.get('--highfreq', HIGHFREQ))
	except ValueError:
		print('--highfreq must be a number', file=sys.stderr)
		return
	if '--persentence' in opts and chunksize is not None:
		print('--chunksize cannot be combined with --persentence',
				file=sys.stderr)
		return
	try:
		result = compare(
				args, opts['--lang'], parse='--parse' in opts,
				persentence='--persentence' in opts,
				chunksize=chunksize, lowfreq=lowfreq, highfreq=highfreq)
	except ValueError as error:
		print(error, file=sys.stderr)
		return
	if '--persentence' in opts:
		if '--output' in opts:
			result.to_csv(opts.get('--output'), sep='\t')
		else:
			print(result)
	elif '--output' in opts:
		result.to_csv(opts.get('--output'), sep='\t')
	else:
		selection = ['NSENT', 'NWORDS', 'WORDLEN', 'LEN', 'LENVAR', 'MDD',
				'NDD', 'ADJD', 'LEFT', 'DEPTH', 'AVGMDD', 'MAXMDD',
				'SUBORD', 'SUBDEPTH', 'SUBPRE', 'SUBPOST', 'CLS', 'CLL',
				'NMOD', 'NMODDEPTH', 'NCOMPRATE', 'NCOMPSIZE', 'PRED',
				'VBROOT', 'PREDROOT', 'VBEDGES',
				'SBJPRE', 'SBJPOST', 'OBJPRE', 'OBJPOST',
				'LXD', 'MTLD', 'LOWF', 'HIGHF', 'ZIPF', 'WFCOV']
		print(result.loc[:, selection].round(3).T)  # skip tags


if __name__ == '__main__':
	main()
