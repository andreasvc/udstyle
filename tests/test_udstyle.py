import pytest

import udstyle


def sentence(length, upos='NOUN', lemma_prefix='lemma'):
	"""Return a minimal fake UD sentence."""
	result = []
	for n in range(1, length + 1):
		head = 0 if n == 1 else n - 1
		deprel = 'root' if n == 1 else 'dep'
		result.append([
			n, f'w{n}', f'{lemma_prefix}{n}', upos, '_', '_',
			head, deprel, '_', '_'])
	return result


@pytest.mark.parametrize(
	'lengths,size,expected',
	[
		([4, 6, 5], 10, [10, 5]),
		([8, 7, 5], 10, [8, 12]),
		([7, 5, 8], 10, [12, 8]),
		([15, 5], 10, [15, 5]),
	],
)
def test_chunksents(lengths, size, expected):
	sentences = [sentence(length) for length in lengths]
	chunks = udstyle.chunksents('', sentences, size)
	assert [sum(map(len, chunk)) for _, chunk in chunks] == expected


def test_chunksents_err():
	with pytest.raises(ValueError):
		udstyle.chunksents('', [sentence(5)], 0)


def test_low_frequency_threshold(monkeypatch):
	monkeypatch.setattr(
		udstyle, 'available_languages',
		lambda wordlist: {'large': 'path'} if wordlist == 'large'
		else {'large': 'path', 'small': 'path'})
	assert udstyle.defaultlowfreq('large') == 3.0
	assert udstyle.defaultlowfreq('small') == 3.5
	with pytest.raises(ValueError, match='canonical language code'):
		udstyle.defaultlowfreq('alias')


def test_analyze_returns_one_row_per_chunk(monkeypatch):
	monkeypatch.setattr(
		udstyle, 'conllureader',
		lambda filename, excludepunct: [sentence(4), sentence(6), sentence(5)])
	monkeypatch.setattr(
		udstyle, 'summarizedmetrics',
		lambda sentences, lang, lowfreq, highfreq: {
			'WORDS': sum(map(len, sentences))})
	result = udstyle.analyze('/tmp/sample.conllu', 'en', 3.0, 4.0,
			chunksize=10)
	assert list(result.index) == ['sample.conllu:0', 'sample.conllu:1']
	assert list(result['WORDS']) == [10, 5]


def test_analyze_without_chunks_returns_one_file_row(monkeypatch):
	monkeypatch.setattr(
		udstyle, 'conllureader',
		lambda filename, excludepunct: [sentence(4), sentence(6)])
	monkeypatch.setattr(
		udstyle, 'summarizedmetrics',
		lambda sentences, lang, lowfreq, highfreq: {'WORDS': 10})
	result = udstyle.analyze('/tmp/sample.conllu', 'en', 3.0, 4.0)
	assert list(result.index) == ['sample.conllu']
	assert result.loc['sample.conllu', 'WORDS'] == 10


def test_lenvar_uses_sample_standard_deviation():
	sentences = [sentence(5), sentence(15)]
	metrics = udstyle.documentmetrics(sentences, 'en', 3.0, 4.0)
	assert metrics['LENVAR'] == pytest.approx(2 ** 0.5 / 2)


def test_single_sentence_has_no_lenvar():
	sentences = [sentence(10)]
	metrics = udstyle.documentmetrics(sentences, 'en', 3.0, 4.0)
	assert metrics['LENVAR'] is None


def test_low_and_high_frequency_ratios(monkeypatch):
	values = {'common': 5.0, 'boundary': 4.0, 'rare': 2.0, 'unknown': 0.0}
	monkeypatch.setattr(
		udstyle, 'zipf_frequency',
		lambda lemma, lang: values[lemma])
	sentences = [[
		[1, 'common', 'common', 'NOUN', '_', '_', 0, 'root', '_', '_'],
		[2, 'boundary', 'boundary', 'ADV', '_', '_', 1, 'advmod', '_', '_'],
		[3, 'rare', 'rare', 'ADJ', '_', '_', 1, 'amod', '_', '_'],
		[4, 'unknown', 'unknown', 'VERB', '_', '_', 1, 'acl', '_', '_'],
	]]
	metrics = udstyle.documentmetrics(sentences, 'en', 3.0, 4.0)
	assert metrics['LOWF'] == pytest.approx(2 / 4)
	assert metrics['HIGHF'] == pytest.approx(2 / 4)
	assert metrics['ZIPF'] == pytest.approx((5 + 4 + 2) / 3)
	assert metrics['WFCOV'] == pytest.approx(3 / 4)


def test_unknown_frequencies_have_no_mean(monkeypatch):
	monkeypatch.setattr(udstyle, 'zipf_frequency', lambda lemma, lang: 0.0)
	metrics = udstyle.documentmetrics([sentence(1)], 'en', 3.0, 4.0)
	assert metrics['ZIPF'] is None
	assert metrics['WFCOV'] == 0


def test_dependency_depth():
	chain = sentence(4)
	star = sentence(4)
	for token in star[1:]:
		token[udstyle.HEAD] = 1
	assert udstyle.dependencydepth(chain) == 3
	assert udstyle.dependencydepth(star) == 1
	assert udstyle.sentencemetrics([chain, star])['DEPTH'] == [3, 1]


def test_phase_one_document_and_dependency_metrics():
	chain = sentence(4)
	star = sentence(3)
	for token in star[1:]:
		token[udstyle.HEAD] = 1
	metrics = udstyle.summarizedmetrics([chain, star], 'en', 3.0, 4.0)
	assert metrics['NSENT'] == 2
	assert metrics['NTOK'] == 7
	assert metrics['WORDLEN'] == pytest.approx(2)
	assert metrics['AVGMDD'] == pytest.approx((1 + 2) / 2)
	assert metrics['MAXMDD'] == 2


def test_word_length_always_excludes_punctuation():
	tokens = sentence(2)
	tokens[0][udstyle.FORM] = 'word'
	tokens[1][udstyle.FORM] = '...'
	tokens[1][udstyle.UPOS] = 'PUNCT'
	metrics = udstyle.documentmetrics([tokens], 'en', 3.0, 4.0)
	assert metrics['NTOK'] == 2
	assert metrics['WORDLEN'] == 4


def test_roots_verb_edges_and_argument_order():
	verbal = [
		[1, 'They', 'they', 'PRON', '_', '_', 2, 'nsubj', '_', '_'],
		[2, 'saw', 'see', 'VERB', '_', '_', 0, 'root', '_', '_'],
		[3, 'it', 'it', 'PRON', '_', '_', 2, 'obj', '_', '_'],
		[4, 'today', 'today', 'ADV', '_', '_', 2, 'advmod', '_', '_'],
	]
	copular = [
		[1, 'Ready', 'ready', 'ADJ', '_', '_', 0, 'root', '_', '_'],
		[2, 'is', 'be', 'AUX', '_', '_', 1, 'cop', '_', '_'],
		[3, 'she', 'she', 'PRON', '_', '_', 1, 'nsubj', '_', '_'],
	]
	metrics = udstyle.summarizedmetrics(
			[verbal, copular], 'en', 3.0, 4.0)
	assert metrics['PRED'] == 1
	assert metrics['VBROOT'] == pytest.approx(0.5)
	assert metrics['PREDROOT'] == 1
	assert metrics['VBEDGES'] == 3
	assert metrics['VBEDGE3'] == 1
	assert sum(value for name, value in metrics.items()
			if name.startswith('VBEDGE') and name != 'VBEDGES') == 1
	assert metrics['SBJPRE'] == pytest.approx(0.5)
	assert metrics['SBJPOST'] == pytest.approx(0.5)
	assert metrics['OBJPRE'] == 0
	assert metrics['OBJPOST'] == 1


def test_document_verb_edges_is_weighted_by_verbal_heads():
	oneverb = sentence(3, upos='VERB')
	twoverbs = sentence(2, upos='VERB')
	metrics = udstyle.summarizedmetrics(
			[oneverb, twoverbs], 'en', 3.0, 4.0)
	# Arities are 1, 1, 0 and 1, 0; averaging sentence means would give 7/12.
	assert metrics['VBEDGES'] == pytest.approx(3 / 5)
	assert metrics['VBEDGE0'] == pytest.approx(2 / 5)
	assert metrics['VBEDGE1'] == pytest.approx(3 / 5)


def test_example_data():
	result = udstyle.analyze(
			'tests/example.conllu', 'en', 3.0, 4.0,
			excludepunct=False).iloc[0]
	assert result['NSENT'] == 3
	assert result['NTOK'] == 25
	assert result['WORDLEN'] == pytest.approx(5.363636363636363)
	assert result['PRED'] == pytest.approx(5 / 3)
	assert result['VBEDGES'] == pytest.approx(3.25)
	assert result['VBEDGE2'] == pytest.approx(0.25)
	assert result['VBEDGE3'] == pytest.approx(0.25)
	assert result['VBEDGE4'] == pytest.approx(0.5)
	assert result['AVGMDD'] == 3
	assert result['MAXMDD'] == 4
	assert result['PREDROOT'] == 1
	assert result['SBJPRE'] == 1
	assert result['SBJPOST'] == 0
	assert result['OBJPRE'] == 0
	assert result['OBJPOST'] == 1


def test_morphological_feature_distributions_and_coverage():
	tokens = sentence(5, upos='VERB')
	tokens[0][udstyle.FEATS] = 'Mood=Ind|Tense=Pres|VerbForm=Fin'
	tokens[1][udstyle.FEATS] = 'Mood=Ind|VerbForm=Fin'
	tokens[2][udstyle.FEATS] = 'Mood=Imp|VerbForm=Fin'
	tokens[3][udstyle.FEATS] = 'VerbForm=Inf'
	tokens[4][udstyle.UPOS] = 'NOUN'
	tokens[4][udstyle.FEATS] = 'Number=Sing'
	tags = udstyle.counttags([tokens])
	assert tags['Mood=Ind'] == pytest.approx(2 / 3)
	assert tags['Mood=Imp'] == pytest.approx(1 / 3)
	assert tags['MoodCov'] == pytest.approx(3 / 4)
	assert tags['VerbForm=Fin'] == pytest.approx(3 / 4)
	assert tags['VerbForm=Inf'] == pytest.approx(1 / 4)
	assert tags['VerbFormCov'] == 1


def test_morphological_feature_on_nonverb_is_included():
	token = sentence(1, upos='ADJ')[0]
	token[udstyle.FEATS] = 'VerbForm=Part'
	result = udstyle.countmorph([[token]], 'VerbForm', {'VERB', 'AUX'})
	assert result == {'VerbFormCov': 1, 'VerbForm=Part': 1}


def test_mtld_requires_50_content_words():
	sentences = [sentence(49)]
	metrics = udstyle.documentmetrics(sentences, 'en', 3.0, 4.0)
	assert metrics['MTLD'] is None


def test_mtld_all_unique_words_form_one_factor():
	words = [f'word{n}' for n in range(50)]
	assert udstyle.getmtld(words) == 50


def test_mtld_repeated_word():
	assert udstyle.getmtld(['word'] * 50) == 2


def test_document_mtld_uses_content_word_lemmas(monkeypatch):
	received = []
	def fake_mtld(words, threshold=udstyle.MTLDTHRESHOLD):
		received.extend(words)
		return 42.5

	monkeypatch.setattr(udstyle, 'getmtld', fake_mtld)
	monkeypatch.setattr(udstyle, 'zipf_frequency', lambda lemma, lang: 5.0)
	sentences = [sentence(50, lemma_prefix='Lemma')]
	sentences[0][0][udstyle.LEMMA] = 'Lemma-1!'
	metrics = udstyle.documentmetrics(sentences, 'en', 3.0, 4.0)
	assert received == ['lemma-1!'] + [
			f'lemma{n}' for n in range(2, 51)]
	assert metrics['MTLD'] == 42.5
