import json
import re
from collections import Counter

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        
        if self.dataset_name != 'iu_xray':
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        self.clean_report = self.clean_report_iu_xray
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

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
        
        sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\[\]{}]', '', t.replace('"', '').replace('/', '') \
                                         .replace('\\', '').replace("'", '').strip().lower())
        
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report_cleaned = ' . '.join(tokens) + ' .'
        
        return report_cleaned

    def get_token_by_id(self, id):
        return self.idx2token.get(id, '<unk>')  # Return <unk> if id not found

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.token2idx['<unk>'])  # Return <unk> id if token not found

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.get_id_by_token(token) for token in tokens]
        ids = [0] + ids + [0]  # Add padding tokens at the beginning and end
        return ids

    def decode(self, ids):
        txt = []
        for idx in ids:
            if idx > 0:  # Skip padding (0) and unknown tokens (if applicable)
                token = self.idx2token.get(idx)  
                if token:  
                    txt.append(token)  
        return ' '.join(txt).strip() 
      # Join tokens into a string and remove extra spaces

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]