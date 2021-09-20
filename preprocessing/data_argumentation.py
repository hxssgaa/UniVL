import json
import pickle as pkl

from transformers import MarianMTModel, MarianTokenizer


def main():
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-roa')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-roa')
    tokenizer2 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-roa-en')
    model2 = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-roa-en')
    x = pkl.load(open('data/dstc10/dstc10_data.all.pickle', 'rb'))
    back_translated = []
    for k in x:
        for e in x[k]['transcript']:
            s = str(e)
            translated = model.generate(**tokenizer(s, return_tensors="pt", padding=True))
            translated_s = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
            back_translated = model2.generate(**tokenizer2(translated_s, return_tensors="pt", padding=True))
            back_translated_s = [tokenizer2.decode(t, skip_special_tokens=True) for t in back_translated][0]
            s = s.lower()
            back_translated_s = back_translated_s.lower()
            back_translated_s = back_translated_s.replace('?', ' ?').replace('  ', ' ')
            if s != back_translated_s:
                print(s, '--->', back_translated_s)
            else:
                print('done')
    print()


if __name__ == '__main__':
    main()
