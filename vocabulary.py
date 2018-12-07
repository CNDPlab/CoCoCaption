import pickle as pk



class Vocab:
    def __init__(self):
        self.data = pk.load(open('vocab.pkl', 'rb'))
        self.t2i = self.data['t2i']
        self.i2t = self.data['i2t']

    def convert(self, input, type):
        assert type in ['li2t', 'lt2i', 'i2t', 't2i']
        if type == 'li2t':
            return self.convert_li2t(input)
        elif type == 'lt2i':
            return self.convert_lt2i(input)
        elif type == 't2i':
            return self.convert_t2i(input)
        else:
            return self.convert_i2t(input)

    def convert_i2t(self, input):
        return self.i2t[input]

    def convert_t2i(self, input):
        try:
            return self.t2i[input]
        except:
            return self.t2i['<UNK>']

    def convert_li2t(self, input):
        return [self.convert_i2t(i) for i in input]

    def convert_lt2i(self, input):
        return [self.convert_t2i(i) for i in input]

