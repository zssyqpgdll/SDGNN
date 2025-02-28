import logging
import random
import string
from collections import Counter
from tqdm import tqdm
import numpy as np

from c2nl.objects import Code, Summary
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines


logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(lang_id,
                     type_index,
                     guid,
                     source,
                     source_tag,
                     target,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     dataset_name,
                     args,
                     uncase=False,
                     test_split=True,
                     load_adjacency=True):
    code_tokens = source.split()
    code_type = []
    code_node_type_index = [int(type_str) for type_str in type_index.split()]

    if len(code_tokens) != len(code_node_type_index):
        print(len(code_tokens))
        print(code_tokens)
        print(len(code_node_type_index))
        print(code_node_type_index)
        print(guid)
        assert len(code_tokens) == len(code_node_type_index)


    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            return None

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    code_node_type_index = code_node_type_index[:max_src_len]
    assert len(code_tokens) == len(code_node_type_index)

    if len(code_tokens) == 0:
        return None

    # 创建node type mask
    code_node_type_mask = np.zeros((args.num_node_type, len(code_node_type_index), 1))
    for node_index, node_type_index in enumerate(code_node_type_index):
        # node_type_index是从1开始的
        code_node_type_mask[node_type_index - 1, node_index, 0] = 1






    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP
    code = Code()
    code.text = source
    code.language = lang_id
    code.tokens = code_tokens
    code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
    if code_tag_type != 'subtoken':
        code.mask = [1 if ct == 'N' else 0 for ct in code_type]

    code.node_type_index = code_node_type_index
    code.node_type_mask = code_node_type_mask



    # TODO: 这里主要是因为加载6w多个adjacency，占用的空间实在太大了
    # 所以在构造模型所需的字典embedding时，可以选择不加载adjacency
    if load_adjacency:
        code.struc = np.load('{}/{}/adjacency/{}.npy'.format(args.data_dir, dataset_name, guid))


    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None

    example = dict()
    example['code'] = code
    example['summary'] = summary
    return example


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False, load_adjacency=True):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src']) as f:
        sources = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['src']))]

    if filenames['tgt'] is not None:
        with open(filenames['tgt']) as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(sources)

    if filenames['src_tag'] is not None:
        with open(filenames['src_tag']) as f:
            source_tags = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_tag']))]
    else:
        source_tags = [None] * len(sources)
    if filenames['guid'] is not None:
        with open(filenames['guid'], encoding='utf-8') as f:
            guids = [line.strip() for line in
                     tqdm(f, total=count_file_lines(filenames['guid']))]
    else:
        guids = [None] * len(sources)

    if filenames['type_index'] is not None:
        with open(filenames['type_index'], encoding='utf-8') as f:
            type_indexs = [line.strip() for line in
                     tqdm(f, total=count_file_lines(filenames['type_index']))]
    else:
        type_indexs = [None] * len(sources)

    print("{} {} {} {} {}".format(len(sources), len(source_tags), len(targets), len(guids), len(type_indexs)))
    assert len(sources) == len(source_tags) == len(targets) == len(guids) == len(type_indexs)


    examples = []
    for type_index, guid, src, src_tag, tgt in tqdm(zip(type_indexs, guids, sources, source_tags, targets),
                                        total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(lang_id=LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   type_index=type_index,
                                   guid=guid,
                                   source=src,
                                   source_tag=src_tag,
                                   target=tgt,
                                   max_src_len=args.max_src_len,
                                   max_tgt_len=args.max_tgt_len,
                                   code_tag_type=args.code_tag_type,
                                   dataset_name=dataset_name,
                                   args=args,
                                   uncase=args.uncase,
                                   test_split=test_split,
                                   load_adjacency=load_adjacency)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples



def load_data_version2(args, filenames, start_index=None, end_index=None, max_examples=-1, dataset_name='java',
              test_split=False, load_adjacency=True):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src']) as f:
        sources = [line.strip() for line in f.readlines()]
        sources=sources[start_index:end_index]


    if filenames['tgt'] is not None:
        with open(filenames['tgt']) as f:
            targets = [line.strip() for line in f.readlines()]
            targets = targets[start_index:end_index]
    else:
        targets = [None] * len(sources)


    if filenames['src_tag'] is not None:
        with open(filenames['src_tag']) as f:
            source_tags = [line.strip() for line in f.readlines()]
            source_tags = source_tags[start_index:end_index]
    else:
        source_tags = [None] * len(sources)


    if filenames['guid'] is not None:
        with open(filenames['guid'], encoding='utf-8') as f:
            guids = [line.strip() for line in f.readlines()]
            guids = guids[start_index:end_index]
    else:
        guids = [None] * len(sources)

    assert len(sources) == len(source_tags) == len(targets) == len(guids)

    examples = []
    for guid, src, src_tag, tgt in tqdm(zip(guids, sources, source_tags, targets),
                                        total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(lang_id=LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   guid=guid,
                                   source=src,
                                   source_tag=src_tag,
                                   target=tgt,
                                   max_src_len=args.max_src_len,
                                   max_tgt_len=args.max_tgt_len,
                                   code_tag_type=args.code_tag_type,
                                   dataset_name=dataset_name,
                                   args=args,
                                   uncase=args.uncase,
                                   test_split=test_split,
                                   load_adjacency=load_adjacency)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples



# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].tokens)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictioanry


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
