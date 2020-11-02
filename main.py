# -*- coding: utf-8 -*-
"""
Created on Mon Mar 2 00:09:07 2020

@author: EmanW10
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from adverserial import AdverserialNLP

squad_dataset_filepath = "train-v2.0.json"
# If true, Squad dataset file placed at "squad_dataset_filepath" will be used for adversarial attack.
# Otherwise the provided example will be used for adversarial attack.
use_squad_dataset = True

def read_squad_dataset(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj

def extract_questions(dataset):
    """
    Takes JSON data of squad dataset and converts it  into a format acceptable by AdverserialNLP
    """
    questions = {}
    for a in dataset['data']:
        for p in a['paragraphs']:
            for q in p['qas']:
                text = p['context']
                question = q['question']
                answers = [x['text'] for x in q['answers']]
                questions[q['id']] = {
                    'context': text,
                    'question': question,
                    'answers': answers
                }
            # break
        # break
    return questions

def evaluate(model, questions):
    """
    Takes a dictionary mapping question_ids to question info (question_text, answers, context)
    """
    f1s = []
    for q_id, q in questions.items():
        score, answer = model.infer(q)
        print("Prediction:", "'" + answer + "'", " " * ((20 - len(answer)) % 20), "Answer:", q['answers'])
        f1s.append(model.get_max_f1(answer, q['answers']))
    print("\nAverage F1-Score:", sum(f1s)/len(f1s))

if use_squad_dataset:
    dataset = read_squad_dataset("train-v2.0.json")
    questions = extract_questions(dataset)
    to_remove = list(questions.keys())[20:]
    for key in to_remove:
        del questions[key]
else:
    context = r"""The history of New York begins around 10,000 B.C. when the first people arrived. By 1100 A.D. two main cultures had become dominant as the Iroquoian and Algonquian developed.
    European discovery of New York was led by the Italian Giovanni da Verrazzano in 1524 followed by the first land claim in 1609 by the Dutch.
    As part of New Netherland, the colony was important in the fur trade and eventually became an agricultural resource thanks to the patroon system.
    In 1626 the Dutch bought the island of Manhattan from American Indians.[1] In 1664, England renamed the colony New York, after the Duke of York (later James II & VII.)
    New York City gained prominence in the 18th century as a major trading port in the Thirteen Colonies.

    New York played a pivotal role during the American Revolution and subsequent war.
    The Stamp Act Congress in 1765 brought together representatives from across the Thirteen Colonies to form a unified response to British policies.
    The Sons of Liberty were active in New York City to challenge British authority.
    After a major loss at the Battle of Long Island, the Continental Army suffered a series of additional defeats that forced a retreat from the New York City area, leaving the strategic port and harbor to the British army and navy as their North American base of operations for the rest of the war. The Battle of Saratoga was the turning point of the war in favor of the Americans, convincing France to formally ally with them. New York's constitution was adopted in 1777, and strongly influenced the United States Constitution. New York City was the national capital at various times between 1785 and 1790, where the Bill of Rights was drafted.
    Albany became the permanent state capital in 1797.
    In 1787, New York became the eleventh state to ratify the United States Constitution."""

    questions = {
        'id1': {
            'context': context, 
            'question': "Who was responsible for the founding of New York?", 
            'answers': ["giovanni da verrazzano"]
        },
        'id2': {
            'context': context, 
            'question': "When did New York help push forward the Consitution?", 
            'answers': ["1777"]
        },
        'id3': {
            'context': context, 
            'question': "Who was the city named after?",
            'answers': ["duke of york"]
        },
        'id4': {
            'context': context, 
            'question': "Which tribes were situated in New York early on?", 
            'answers': ['Iroquoian and Algonquian']
        }
    }

tokenizer = AutoTokenizer.from_pretrained('twmkn9/albert-base-v2-squad2')
model = AutoModelForQuestionAnswering \
    .from_pretrained('twmkn9/albert-base-v2-squad2')

adverserial_obj = AdverserialNLP(model, tokenizer)

print("\nEvaluating data before adverserial attack:")
evaluate(adverserial_obj, questions)
print("Evaluation Complete\n")

questions_adverserial = adverserial_obj.addany_search(questions)

print("\nEvaluating data after adverserial attack:")
evaluate(adverserial_obj, questions_adverserial)
print("Evaluation Complete\n")
