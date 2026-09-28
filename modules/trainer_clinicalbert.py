import json
import re
from collections import Counter
from transformers import AutoTokenizer, AutoConfig #Import Autoconfig for Tokenizer

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.pretrained_tokenizer_name = args.pretrained_tokenizer_name  # Add this line

        if self.dataset_name != 'iu_xray':
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.clean_report = self.clean_report_iu_xray
        self.ann = json.loads(open(self.ann_path, 'r').read())

        # Load the pre-trained BioBERT tokenizer and configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_name)
        self.config = AutoConfig.from_pretrained(self.pretrained_tokenizer_name) #Load config
        self.vocab_size = self.config.vocab_size #set Vocab size, so it can be correctly set by the code


        # You can still create your own vocabulary if needed, but it's not strictly necessary
        # self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        
        token2idx = {token: idx + 1 for idx, token in enumerate(vocab)}  # Start index from 1
        idx2token = {idx + 1: token for idx, token in enumerate(vocab)}
        
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('1. ', '') \
            .strip().lower().split('. ')
        
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '') \
                                         .replace('\\', '').replace("'", '').strip().lower())
        
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report_cleaned = ' . '.join(tokens) + ' .'
        
        return report_cleaned

    def get_token_by_id(self, id):
        return self.tokenizer.convert_ids_to_tokens(id)  # Use BioBERT's method

    def get_id_by_token(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)  # Use BioBERT's method

    def get_vocab_size(self):
        return self.tokenizer.vocab_size  # Use BioBERT's attribute

    def __call__(self, report):
        # Use the pre-trained BioBERT tokenizer
        ids = self.tokenizer.encode(report, add_special_tokens=True)
        return ids

    def decode(self, ids):
        # Use the pre-trained BioBERT tokenizer's decoding function
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]
