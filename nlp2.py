import nltk
from nltk.corpus import brown
from nltk import CFG

# 訓練データ
tagged_sentences = brown.tagged_sents(categories="news")
size = int(len(tagged_sentences) * 0.9)
train_sentences = tagged_sentences[:size]

# 解析する文章をトークン化
sentence = input("解析したい文章を入力してください： ")
tokens = nltk.word_tokenize(sentence)

# 正規表現タガー
patterns = [
    (r"^([A-Z][a-z]+\.?)+$", "NNP"),
    (r"^-?[0-9]+(.[0-9]+)?$", "CD"),
    (r".*s$", "NNS"),
    (r".*ing$", "VBG"),
    (r".*ed$", "VBD"),
    (r".*es$", "VBZ"),
    (r".*ould$", "MD"),
    (r".*\'s$", "NN$"),
    (r".*ment$", "NN"),
    (r".*ful$", "JJ"),
    (r".*ly$", "RB"),
    (r".*", "NN")
]
regexp_tagger = nltk.RegexpTagger(patterns)

# ユニグラム・バイグラムタガー
unigram_tagger = nltk.UnigramTagger(train_sentences)
bigram_tagger = nltk.BigramTagger(train_sentences)

# 組み合わせタガー
combination_tagger_1 = nltk.UnigramTagger(train_sentences, backoff=regexp_tagger)
combination_tagger_2 = nltk.BigramTagger(train_sentences, backoff=combination_tagger_1)
combination_tagger_3 = nltk.TrigramTagger(train_sentences, backoff=combination_tagger_2)

# 簡易的な品詞に変更
def simplify_tag(tag):
    if tag.startswith('N') or tag == 'PPSS':
        return 'noun'
    elif tag.startswith('V') or tag.startswith('B'):
        return 'verb'
    elif tag.startswith('J'):
        return 'adj'
    elif tag.startswith('R'):
        return 'adv'
    elif tag == 'DT' or tag == 'AT' or tag == 'PP$':
        return 'det'
    elif tag in ['IN', 'TO']:
        return 'prep'
    else:
        return tag

# タグ付けを実行
print("【品詞のタグづけ結果】")
tagged_tokens = combination_tagger_3.tag(tokens)
simplified_tags = [(token, simplify_tag(tag)) for token, tag in tagged_tokens]
for token, tag in simplified_tags:
    print(token, tag)
print()

# 適格部分文字列表の作成
def init_wfst(tokens, grammar):
     numtokens = len(tokens)
     wfst = [[None for i in range(numtokens+1)] for j in range(numtokens+1)]
     for i in range(numtokens):
         productions = grammar.productions(rhs=tokens[i])
         wfst[i][i+1] = productions[0].lhs()
     return wfst

def complete_wfst(wfst, tokens, grammar, trace=False):
     index = dict((p.rhs(), p.lhs()) for p in grammar.productions())
     numtokens = len(tokens)
     for span in range(2, numtokens+1):
         for start in range(numtokens+1-span):
             end = start + span
             for mid in range(start+1, end):
                 nt1, nt2 = wfst[start][mid], wfst[mid][end]
                 if nt1 and nt2 and (nt1,nt2) in index:
                     wfst[start][end] = index[(nt1,nt2)]
                     if trace:
                         print("[{}] {} [{}] {} [{}] ==> [{}] {} [{}]".format(start, nt1, mid, nt2, end, start, index[(nt1,nt2)], end))
     return wfst

def display(wfst, tokens):
    num_tokens = len(wfst) - 1

    headers = ' '.join([("%-4d" % i) for i in range(1, num_tokens + 1)])
    print('WFST ' + headers)

    for i in range(num_tokens):
        row = "{:<4d}".format(i)
        for j in range(1, num_tokens + 1):
            cell = "{}".format(wfst[i][j] or '.')
            row += "{:<5s}".format(cell)
        print(row)

# grammarの定義を動的に
def create_grammar_from_data(data):
    nouns = set(token for token, pos in data if pos == 'noun')
    verbs = set(token for token, pos in data if pos == 'verb')
    adjectives = set(token for token, pos in data if pos == 'adj')
    adverbs = set(token for token, pos in data if pos == 'adv')
    prepositions = set(token for token, pos in data if pos == 'prep')
    determiners = set(token for token, pos in data if pos == 'det')

    grammar_str = """
        S -> NP VP | S PP
        NP -> noun | det noun | adj NP | NP PP
        VP -> verb NP | adv VP | verb NP NP | VP PP
        PP -> prep NP
        noun -> {}
        verb -> {}
        adj -> {}
        adv -> {}
        prep -> {}
        det -> {}
    """.format(" | ".join('"{}"'.format(token) for token in nouns),
               " | ".join('"{}"'.format(token) for token in verbs),
               " | ".join('"{}"'.format(token) for token in adjectives),
               " | ".join('"{}"'.format(token) for token in adverbs),
               " | ".join('"{}"'.format(token) for token in prepositions),
               " | ".join('"{}"'.format(token) for token in determiners))

    grammar = CFG.fromstring(grammar_str)
    return grammar


grammar = create_grammar_from_data(simplified_tags)

# インデックスの表示
print("【インデックス】")
tokens = sentence.split()
indexed_string = ' '.join(['[{}] {}'.format(i, token) for i, token in enumerate(tokens)]) + ' [{}]'.format(len(tokens))
print(indexed_string)
print()

# 適格部分文字列表
print("【適格部分文字列表】")
wfst0 = init_wfst(tokens, grammar)
wfst1 = complete_wfst(wfst0, tokens, grammar)
display(wfst1, tokens)
print()
wfst1 = complete_wfst(wfst0, tokens, grammar, trace=True)
print()

# チャート法構文解析による構文木の表示
print("【構文木】")
parser = nltk.ChartParser(grammar)
for tree in parser.parse(tokens):
  print(tree)
