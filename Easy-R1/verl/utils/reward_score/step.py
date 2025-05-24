import re

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


def get_chat_response(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=4, n=1, patience=10, sleep_time=0):
    while patience > 0:
        patience -= 1
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    max_tokens=max_tokens,
                                                    temperature=temperature,
                                                    n=n)

            prediction = response['choices'][0]['message']['content'].strip()
            if prediction != "" and prediction != None:
                return prediction

        except Exception as e:
            print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""

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


def r1v_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def ret_accuracy_reward(answer: str, ground_truth: dict, model) -> float:
    # ground_truth = ground_truth.strip()
    try:
        answer = answer.lower()
        think_pattern = r"<think>(.*?)</think>"
        querty_pattern = r"<sub-question>(.*?)</sub-question>"
        retriever_pattern = r"<ret>(.*?)</ret>"
        # if len(answer) >= 1000:
        #     raise ValueError

        thought = re.search(think_pattern, answer, re.DOTALL).group(1).strip()
        query = [re.search(querty_pattern, answer, re.DOTALL).group(1).strip()]
        retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1).strip()

        gt_query = [ground_truth['query']]
        gt_retriever = ground_truth['retriever']
        retriever_list = [item.strip() for item in retriever.split(',')]
        # if query[0] == 'none' and retriever != 'none':
        #     raise ValueError
        # if query[0] != 'none' and retriever == 'none':
        #     raise ValueError
        # if query[0] == 'none' and retriever == 'none':
        #     if gt_query == 'none':
        #         return 1.0
        #     else:
        #         return 0.0
        for ret in retriever_list:
            if ret == 'none' and len(retriever_list) > 1:
                raise ValueError
            if not ret in ['text retriever', 'text image retriever', 'table retriever', 'none']:
                raise ValueError
        
        embeddings_query = model.encode(query, 
                            batch_size=12, 
                            max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
        embeddings_gt_query = model.encode(gt_query)['dense_vecs']
        similarity = (embeddings_query @ embeddings_gt_query.T)[0][0]

        set1, set2 = set(retriever_list), set(gt_retriever)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        # iou = float(len(intersection) / len(union))
        # iou = iou * iou
        ret_reward = 0
        if len(retriever_list) == 1:
            if retriever_list[0] == gt_retriever[0]:
                ret_reward = 1
        print(f"[INFO] verl.utils.reward_score.step.ret query: {query}, gt query: {gt_query}")
        print(f"[INFO] verl.utils.reward_score.step.ret think: {thought}")
        print(f"[INFO] verl.utils.reward_score.step.ret retriever: {retriever_list}, gt retriever: {gt_retriever}")
        print(f"[INFO] verl.utils.reward_score.step.ret similarity: {similarity}, ret_reward: {ret_reward}")
        return 0.5 * similarity + 0.5 * ret_reward
        
    except Exception as e:
        print(f"[ERROR] verl.utils.reward_score.step.ret {e}, output: {answer}")
        return 0.0


def ans_accuracy_reward(answer: str, ground_truth: str) -> float:
    try:
        answer = answer.lower()
        answer_pattern = r"<answer>(.*?)</answer>"
        think_pattern = r"<think>(.*?)</think>"
        # if len(answer) >= 1000:
        #     raise ValueError
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        thought = re.search(think_pattern, answer, re.DOTALL).group(1).strip()
        answer = re.search(answer_pattern, answer, re.DOTALL).group(1).strip()
        recall = evaluate(answer, ground_truth)[1]
        print(f"[INFO] verl.utils.reward_score.step.ans recall: {recall}")
        print(f"[INFO] verl.utils.reward_score.step.ans ans: {answer}")
        print(f"[INFO] verl.utils.reward_score.step.ans think: {thought}")
        print(f"[INFO] verl.utils.reward_score.step.ans ground truth: {ground_truth}")
        if len(ground_truth) == 1:
            if len(ground_truth[0].split(" ")) <= 5:
                if recall == 1:
                    return 1.0
            else:
                return recall
        else:
            if recall == 1:
                return 1.0
        return 0.0
    except Exception as e:
        print(f"[ERROR] verl.utils.reward_score.step.ans {e}, output: {answer}")
        return 0.0


def step_compute_score(predict_str: str, ground_truth: str, step_type: str, model) -> float:
    if step_type == "ret":
        reward = ret_accuracy_reward(predict_str, ground_truth, model)
    elif step_type == "ans":
        reward = ans_accuracy_reward(predict_str, ground_truth)

    return reward
