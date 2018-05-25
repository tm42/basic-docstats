import math

from functools import partial
from collections import Counter, deque


class WordStatMachine(object):
    '''basic statistics on words in a collection of docs:
    affinity, tf-idf, frequency and share
    '''
    STOP_LIST_RU = set(
        # '''хорошо, лучше, никогда''' +
        '''и в во не что он на я с c со как а то все она так его но да
        ты к у же вы за бы по только ее мне было вот от меня еще нет
        о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до
        вас нибудь опять уж вам ведь там потом себя ничего ей может они
        тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего
        раз тоже себе под будет ж тогда кто этот того потому этого какой
        совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда
        зачем всех можно при наконец два об другой хоть после над больше
        тот через эти нас про всего них какая много разве три эту моя впрочем
        свою этой перед иногда чуть том нельзя такой им более
        всегда конечно всю между это ru'''.split()
    )
    STOP_LIST_EN = {
        'a', 'about', 'above', 'above', 'across', 'after', 'afterwards',
        'again', 'against', 'all', 'almost', 'alone', 'along', 'already',
        'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
        'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
        'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back',
        'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been',
        'before', 'beforehand', 'behind', 'being', 'below', 'beside',
        'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but',
        'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt',
        'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due',
        'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else',
        'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every',
        'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
        'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former',
        'formerly', 'forty', 'found', 'four', 'from', 'front', 'full',
        'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have',
        'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
        'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how',
        'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest',
        'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter',
        'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me',
        'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
        'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
        'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody',
        'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of',
        'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
        'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over',
        'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
        'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious',
        'several', 'she', 'should', 'show', 'side', 'since', 'sincere',
        'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something',
        'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
        'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
        'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
        'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
        'this', 'those', 'though', 'three', 'through', 'throughout', 'thru',
        'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve',
        'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very',
        'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when',
        'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
        'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
        'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
        'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'the'
    }
    STOP_LISTS = {
        'ru': STOP_LIST_RU,
        'en': STOP_LIST_EN,
    }

    def __init__(self):
        self._original_docs = deque()
        self._od_index_by_name = {}
        self.measured = {'freq': deque(), 'share': deque()}
        self.doc_counters = {}
        self.docs_n = 0
        self.all_words_n = 0
        self.doc_len = {}
        self.doc_word_counter = Counter([])
        self.all_words_counter = Counter([])

    @classmethod
    def _sanitize(cls, doc, stop_list=None, lang=None):
        '''drops words from the stop-list and non alpha-numeric words'''
        assert lang in {None} | cls.STOP_LISTS.keys(), 'wrong lang value'
        # TODO: more obvious logics required here
        stop_list = (
            stop_list if stop_list is not None else cls.STOP_LIST[lang]
            if lang is not None else cls.STOP_LIST_RU | cls.STOP_LIST_EN
        )
        if isinstance(stop_list, str):
            stop_list = set(stop_list.split())
        elif isinstance(stop_list, list):
            stop_list = set(stop_list)
        doc = [word for word in doc if word not in stop_list]
        return [word for word in doc if word.isalnum()]

    def add_document(self, doc, doc_name=None,
                     sanitize=True, stop_list=None, sep=', '):
        '''add a new doc to the collection'''
        if isinstance(doc, str):
            doc = doc.split(sep)[:-1]
        if sanitize:
            doc = self._sanitize(doc, stop_list)
        doc_orig = (
            doc_name if doc_name is not None else len(self._original_docs),
            doc[:]
        )
        self._original_docs.append(doc_orig)
        self._od_index_by_name[doc_orig[0]] = len(self._original_docs) - 1
        self.all_words_counter.update(doc)
        self.doc_word_counter.update(set(doc[:]))
        self.docs_n += 1
        doc_len = len(doc)
        self.all_words_n += doc_len
        self.doc_len[doc_name] = doc_len
        if doc_name not in self.doc_counters:
            self.doc_counters[doc_name] = Counter(doc)
        self.measured['freq'].append(
            (doc_name, self.doc_counters[doc_name].most_common())
        )
        self.measured['share'].append(
            (doc_name, list(
                map(
                    lambda x, doc_len=len(doc): (x[0], x[1] / doc_len),
                    self.measured['freq'][-1][1]
                )
            ))
        )

    def affinity(self, word, doc_counter, doc_len, mode=True):
        '''affinity calc'''
        local_word_share = doc_counter[word] / doc_len
        global_word_share = self.all_words_counter[word] / self.all_words_n
        affinity = local_word_share / global_word_share
        return affinity if mode else (affinity, doc_counter[word])

    def tf_idf(self, word, doc_counter, doc_len, mode=True):
        '''tf-idf calc, log2 is used'''
        tf = doc_counter[word] / doc_len
        idf = math.log2(self.docs_n / self.doc_word_counter[word])
        return tf * idf if mode else (tf * idf, doc_counter[word])

    def compute_metric(self, metric_name='aff', mode=True):
        '''compute chosen metric, where choices are ['aff', 'tf-idf']'''
        assert metric_name in {'aff', 'tf-idf'}, (
            'metric can be either "aff" or "tf-idf"'
        )
        self.measured[metric_name] = deque()
        if metric_name == 'tf-idf':
            metric = self.tf_idf
        elif metric_name == 'aff':
            metric = self.affinity
        for doc in self._original_docs:
            doc_name, doc_words = doc
            doc_counter = self.doc_counters[doc_name]
            doc_len = self.doc_len[doc_name]
            tool = partial(metric, doc_counter=doc_counter,
                           doc_len=doc_len, mode=mode)
            tmp_vector = [tool(word) for word in doc_words]
            tmp_doc = (doc_name, tmp_vector)
            self.measured[metric_name].append(tmp_doc)

    def most_characteristic_words(self, n=5, metric_name='aff', treshold=0):
        '''get top-n of words by a chosen metric for each doc'''
        assert metric_name in {'aff', 'tf-idf', 'share', 'freq'}, (
            '''metric must belong to {'aff', 'tf-idf', 'share', 'freq'}'''
        )
        top_lists = []
        for i,doc in enumerate(self._original_docs):
            n = min([len(doc[1]), n])
            metric_doc = self.measured[metric_name][i]
            if metric_name in ['share', 'freq']:
                top_lists.append((doc[0], metric_doc[1][:n]))
                continue
            tmp = [(word, metric_doc[1][x]) for x,word in enumerate(doc[1])]
            tmp = list(set(tmp))
            try:
                tmp = [word for word in tmp if word[1][1] >= treshold]
                tmp.sort(key=lambda x: x[1][0], reverse=True)
            except TypeError:
                tmp.sort(key=lambda x: x[1], reverse=True)
            top_lists.append((doc[0], tmp[:n]))

        return top_lists

    @staticmethod
    def _euclidean_distance(vector1, vector2):
        vectors = list(zip(vector1, vector2))
        distance = math.sqrt(
            sum([math.pow(vs[0] - vs[1], 2) for vs in vectors])
        )
        return distance

    @staticmethod
    def _cosine_distance(vector1, vector2):
        vectors = list(zip(vector1, vector2))
        x = sum([vs[0] * vs[1] for vs in vectors])
        s1v, s2v = (
            math.sqrt(
                sum(map(lambda x: math.pow(x, 2), s))
            ) for s in [vector1, vector2]
        )
        distance = x / (s1v * s2v)
        return distance

    def _get_base_doc(self, base_doc):
        if isinstance(base_doc, str):
            try:
                index = self._od_index_by_name[base_doc]
                words_vector_words = self._original_docs[index][1][:]
                words_vector_values = self.measured['tf-idf'][index][1][:]
            except KeyError:
                raise KeyError('No document with name {}'.format(base_doc))
        elif isinstance(base_doc, int):
            words_vector_words = self._original_docs[base_doc][1][:]
            words_vector_values = self.measured['tf-idf'][base_doc][1][:]
        elif isinstance(base_doc[0], tuple):
            words_vector_words, words_vector_values = map(
                list, zip(*base_doc)
            )
        else:
            raise TypeError(
                'words_vector must be either a list of tuples, '
                + 'an int or a string, '
            )

        if isinstance(words_vector_values[0], tuple):
            words_vector_values = [ft[0] for ft in words_vector_values]

        return words_vector_words, words_vector_values

    def _get_selection(self, indices, names, words_vector_words):
        '''returns set of documents by indices or names.
        if no names and no indices specified, all docs are returned
        '''
        if bool(indices or names) is False:
            print('both indices and names are empty, all docs will be used')
            selection = self.measured['tf-idf'].copy()
            originals = self._original_docs.copy()
        elif names:
            selection = [
                doc for doc in self.measured['tf-idf'] if doc[0] in names
            ].copy()
            originals = [
                doc for doc in self._original_docs if doc[0] in names
            ].copy()
        else:
            selection = [self.measured['tf-idf'][i] for i in indices].copy()
            originals = [self._original_docs[i] for i in indices].copy()

        if isinstance(selection[0][1][0], tuple):
            selection = [
                (doc[0], [ft[0] for ft in doc[1]]) for doc in selection
            ]
        selection_vectors = [
            (
                doc[0], [
                    doc[1][originals[i][1].index(word)]
                    if word in originals[i][1]
                    else 0
                    for word in words_vector_words
                ]
            ) for i,doc in enumerate(selection)
        ]

        return selection_vectors

    def compute_distances(self, base_doc=[], indices=[], names=[],
                          distance='cosine'):
        '''this method requires tf-idf to be computed.

        compute closeness to a given base_doc [(word0, tf-idf-value0), ...]
        if base_doc is an int or a string, it is used as an index / name
        to find base_doc among the collection of documents.

        indices or names of docs can be sent as well,
        if not — all docs are compared.
        '''

        assert 'tf-idf' in self.measured, (
            'tf-idf metric has not been computed yet'
        )
        assert distance in {'cosine', 'euclidean'}, (
            '''distance can be either {'cosine', 'euclidean'}'''
        )
        assert base_doc != [], 'base_doc is not defined'

        words_vector_words, words_vector_values = self._get_base_doc(base_doc)
        selection_vectors = self._get_selection(
            indices, names, words_vector_words
        )

        if distance == 'euclidean':
            distance_f = self._euclidean_distance
        elif distance == 'cosine':
            distance_f = self._cosine_distance

        tool = partial(distance_f, vector2=words_vector_values)
        res = [(doc[0], tool(doc[1])) for doc in selection_vectors]
        res.sort(key=lambda x: x[1], reverse=True)
        return res
