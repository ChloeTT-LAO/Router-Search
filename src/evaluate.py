import re
import os
import json
import time
import jieba
import numpy as np
import openai
import argparse
from collections import Counter
from nltk.tokenize import word_tokenize

root = "./results/"

contractions = {
            'aint': "ain't",
            'arent': "aren't",
            'cant': "can't",
            'couldve': "could've",
            'couldnt': "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            'didnt': "didn't",
            'doesnt': "doesn't",
            'dont': "don't",
            'hadnt': "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hed': "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            'hes': "he's",
            'howd': "how'd",
            'howll': "how'll",
            'hows': "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            'Im': "I'm",
            'Ive': "I've",
            'isnt': "isn't",
            'itd': "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            'itll': "it'll",
            "let's": "let's",
            'maam': "ma'am",
            'mightnt': "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            'mightve': "might've",
            'mustnt': "mustn't",
            'mustve': "must've",
            'neednt': "needn't",
            'notve': "not've",
            'oclock': "o'clock",
            'oughtnt': "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            'shant': "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            'shouldve': "should've",
            'shouldnt': "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": 'somebodyd',
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            'somebodyll': "somebody'll",
            'somebodys': "somebody's",
            'someoned': "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            'someonell': "someone'll",
            'someones': "someone's",
            'somethingd': "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            'somethingll': "something'll",
            'thats': "that's",
            'thered': "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            'therere': "there're",
            'theres': "there's",
            'theyd': "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            'theyll': "they'll",
            'theyre': "they're",
            'theyve': "they've",
            'twas': "'twas",
            'wasnt': "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            'weve': "we've",
            'werent': "weren't",
            'whatll': "what'll",
            'whatre': "what're",
            'whats': "what's",
            'whatve': "what've",
            'whens': "when's",
            'whered': "where'd",
            'wheres': "where's",
            'whereve': "where've",
            'whod': "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            'wholl': "who'll",
            'whos': "who's",
            'whove': "who've",
            'whyll': "why'll",
            'whyre': "why're",
            'whys': "why's",
            'wont': "won't",
            'wouldve': "would've",
            'wouldnt': "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            'yall': "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            'youd': "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've",
        }

manualMap = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
        }

articles = ['an', 'the']

periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')

commaStrip = re.compile('(\d)(,)(\d)')

punct = [
            ';',
            r'/',
            '[',
            ']',
            '"',
            '{',
            '}',
            '(',
            ')',
            '=',
            '+',
            '\\',
            '_',
            '-',
            '>',
            '<',
            '@',
            '`',
            ',',
            '?',
            '!',
        ]


def process_string(s):
    s = str(s)
    words = []
    for word in ' '.join(jieba.cut(s)).split():
        if word not in '，、。 ,.《》':
            words.append(word)
    return words


def process_string_en(s):
    s = str(s).lower()
    words = []
    for word in word_tokenize(s):
        if word not in ',.?!:;\'"':
            words.append(word)
    return words


def compute_acc_single(gold_toks, pred_toks):
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0, 0, 0
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    # print(f"gold toks: {gold_toks}, pred toks: {pred_toks}, precision:{precision}, same: {num_same}")
    return precision, recall, f1


def compute_acc(a_golds, a_pred, lang):
    if lang == 'zh':
        if a_pred == '':
            return 0, 0, 0
        golds_toks = [process_string(a_gold) for a_gold in a_golds]
        pred_toks = process_string(a_pred)
    elif lang == 'en':
        if a_pred == '':
            return 0, 0, 0
        golds_toks = [process_string_en(a_gold) for a_gold in a_golds]
        pred_toks = process_string_en(a_pred)

    max_metrics = (0.0, 0.0, 0.0)  # (precision, recall, f1)
    for gold_toks in golds_toks:
        precision, recall, f1 = compute_acc_single(gold_toks, pred_toks)
        if f1 > max_metrics[2]:
            max_metrics = (precision, recall, f1)

    return max_metrics


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p
                in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess(word):
    word = str(word)
    word = word.replace('\n', ' ')
    word = word.replace('\t', ' ')
    word = word.strip()
    word = processPunctuation(word)
    word = processDigitArticle(word)
    return word


def evaluate(output, answer):
    resAns = output
    resAns = resAns.replace('\n', ' ')
    resAns = resAns.replace('\t', ' ')
    resAns = resAns.strip()
    resAns = processPunctuation(resAns)
    resAns = processDigitArticle(resAns)

    gtAnswers = answer
    gtAnswers = [preprocess(ans) for ans in gtAnswers]
    # print(resAns)
    # print(gtAnswers)
    return compute_acc(a_golds=gtAnswers, a_pred=resAns, lang='en')


def dataset_evaluate(dataset, method, model_name):
    rec_list = []
    with (open(os.path.join(root, model_name, f"{dataset}_{method}.jsonl"), 'r') as f):
        for line in f:
            data = json.loads(line.strip())
            if isinstance(data['llm_answer'], str):
                output = data['llm_answer']
            else:
                output = data['llm_answer']['answer']

            if dataset == 'infoseek':
                answer = data['answer_eval']
                if isinstance(answer[0], dict):
                    ranges = answer[0]['range']
                    if ranges[1] - ranges[0] >= 50:
                        step = 1
                    elif ranges[1] - ranges[0] >= 5:
                        step = 0.1
                    elif ranges[1] - ranges[0] >= 1:
                        step = 0.01
                    else:
                        step = 0.001
                    answer = [str(round(i, 3)) for i in np.arange(ranges[0], ranges[1]+step, step)]
                    answer.extend([i for i in range(int(ranges[0]), int(ranges[1]+1))])

            elif dataset in ['dyn_vqa', 'webqa']:
                answer = data['answer']
            elif dataset in ['openwikitqa', '2wikimultihopqa', 'tabfact']:
                answer = [data['answer']]
            
            if output == "":
                rec_list.append(0)
                continue

            _, rec, _ = evaluate(output, answer)

            rec_list.append(rec)
        print('F1-Recall: ', round(100 * float(sum(rec_list)) / len(rec_list), 2))
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--method", type=str, default="r1-router3")
    parser.add_argument("--model_name", type=str, default="r1-router")
    return parser.parse_args()


def main():
    args = parse_arguments()
    method = args.method
    dataset = args.dataset_name
    model_name = args.model_name
    if dataset == 'all':
        dataset_name = ['openwikitqa', '2wikimultihopqa', 'dyn_vqa', 'tabfact', 'webqa', 'infoseek']
    else:
        dataset_name = [dataset]

    for dataset in dataset_name:
        print(f"{dataset} {method} eval:")
        dataset_evaluate(dataset, method, model_name)
    

if __name__ == "__main__":
    main()