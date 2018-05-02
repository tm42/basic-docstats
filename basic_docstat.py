import math

from functools import partial
from collections import Counter, deque


class WordStatMachine(object):
    '''basic statistics on words in a collection of docs:
    affinity, tf-idf, frequency and share
    '''
    STOP_LIST = set(
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

    def __init__(self):
        self._original_docs = deque()
        self.measured = {'freq': deque(), 'share': deque()}
        self.doc_counters = {}
        self.docs_n = 0
        self.all_words_n = 0
        self.doc_len = {}
        self.doc_word_counter = Counter([])
        self.all_words_counter = Counter([])

    @classmethod
    def _sanitize(cls, doc, stop_list=None):
        '''drops words from the stop-list and non alpha-numeric words'''
        stop_list = stop_list if stop_list is not None else cls.STOP_LIST
        if isinstance(stop_list, str):
            stop_list = set(stop_list.split())
        elif isinstance(stop_list, list):
            stop_list = set(stop_list)
        doc = [word for word in doc if word not in stop_list]
        return [word for word in doc if word.isalnum()]

    def add_document(self, doc, doc_name=None, sanitize=True, stop_list=None):
        '''add a new doc to the collection'''
        if sanitize:
            doc = self._sanitize(doc, stop_list)
        doc_orig = (
            doc_name if doc_name is not None else len(self._original_docs),
            doc[:]
        )
        self._original_docs.append(doc_orig)
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

    def count_metric(self, metric_name='aff', mode=True):
        '''compute chosen metric, where choices are ['aff', 'tf-idf']'''
        self.measured[metric_name] = deque()
        if metric_name == 'tf-idf':
            metric = self.tf_idf
        elif metric_name == 'aff':
            metric = self.affinity
        else:
            raise ValueError('Unknown metric name — {}'.format(metric_name))
        for doc in self._original_docs:
            doc_name, doc_words = doc
            doc_counter = self.doc_counters[doc_name]
            doc_len = self.doc_len[doc_name]
            tool = partial(metric, doc_counter=doc_counter,
                           doc_len=doc_len, mode=mode)
            tmp_vector = [tool(word) for word in doc_words]
            tmp_doc = (doc_name, tmp_vector)
            self.measured[metric_name].append(tmp_doc)

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

    def get_top(self, n=5, metric_name='aff', treshold=0):
        '''get top-n of words by a chosen metric for each doc'''
        top_lists = []
        for i,doc in enumerate(self._original_docs):
            n = min([len(doc[1]), n])
            metric_doc = self.measured[metric_name][i]
            if metric_name in ['share', 'freq']:
                top_lists.append((doc[0], metric_doc[1][:n]))
                continue
            tmp = [(word, metric_doc[1][x]) for x,word in enumerate(doc[1])]
            tmp = list(set(tmp))
            # if isinstance(tmp[0][1], tuple):
            #     tmp = [word for word in tmp if word[1][1] >= treshold]

            #     def key_function(x):
            #         return x[1][0]
            # else:
            #     def key_function(x):
            #         return x[1]
            try:
                tmp = [word for word in tmp if word[1][1] >= treshold]
                tmp.sort(key=lambda x: x[1][0], reverse=True)
            except TypeError:
                tmp.sort(key=lambda x: x[1], reverse=True)
            top_lists.append((doc[0], tmp[:n]))
        return top_lists
