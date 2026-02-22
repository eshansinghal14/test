class WeezlTokenizerTSV:
    def __init__(self, path):
        self.path = path
        self.pattern_to_code, self.code_to_pattern = self._load_dictionary(path)
        self._base_vocab = max(self.code_to_pattern.keys(), default=0) + 1
        self.pad_id = self._base_vocab
        self.unk_id = self._base_vocab + 1
        self.mask_id = self._base_vocab + 2

    def vocab_size(self):
        return self._base_vocab + 3

    def encode(self, text):
        chars = list(text)
        tokens = []
        i = 0
        while i < len(chars):
            current = ''
            best_code = None
            best_len = 0
            j = i
            while j < len(chars):
                current += chars[j]
                code = self.pattern_to_code.get(current)
                if code is not None:
                    best_code = code
                    best_len = j - i + 1
                    j += 1
                else:
                    break

            if best_code is not None:
                tokens.append(best_code)
                i += best_len
            else:
                tokens.append(self.unk_id)
                i += 1
        return tokens

    def decode(self, ids):
        out = []
        for tid in ids:
            if tid == self.pad_id:
                continue
            if tid == self.unk_id:
                out.append('�')
                continue
            if tid == self.mask_id:
                out.append('[MASK]')
                continue
            pat = self.code_to_pattern.get(int(tid))
            if pat is not None:
                out.append(pat)
            else:
                out.append('�')
        return ''.join(out)

    @staticmethod
    def _unescape(s):
        return (s.replace('\\\\', '\x00').replace('\\t', '\t')
                .replace('\\n', '\n').replace('\\r', '\r')
                .replace('\\0', '\0').replace('\x00', '\\'))

    @classmethod
    def _load_dictionary(cls, path):
        pattern_to_code = {}
        code_to_pattern = {0: ''}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t', 3)
                if len(parts) < 4:
                    continue
                code = int(parts[0])
                pattern = cls._unescape(parts[3])
                pattern_to_code[pattern] = code
                code_to_pattern[code] = pattern
        return pattern_to_code, code_to_pattern