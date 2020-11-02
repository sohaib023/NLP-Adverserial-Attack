import torch
import random
import string

import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
from nltk import FreqDist
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

NUM_EPOCHS = 3
NUM_SAMPLE = 10
NUM_ADDITIONS = 10
NUM_SEARCHES_PER_MEGA_EPOCH = [1, 2, 4]
PRINT_RESULTS_PER_ITERATION = True

class AdverserialNLP:

    def __init__(self, model, tokenizer):
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))        
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        
        stop = set(stopwords.words('english')) | set(string.punctuation)
        frequency_list = FreqDist(i.lower() for i in brown.words() if i.lower() not in stop)
        self.vocab = list(map(lambda x: x[0], frequency_list.most_common()[:1000]))

        """
        self.state is a dictionary with question IDs as key and a tuple of 3 variables as the value:
        -   list containing adversarial words {w1, …., wd} (that can be joined to make a sentence) that achieved minimum score.
        -   the minimum score achieved during the adversarial search (from the words stored above)
        -   prediction of the model when this minimum score was achieved.
        At any given point in time this variable contains the results and current state of the adversarial search.
        """
        self.state = None

    def generate_permutations(self, start_probs, end_probs, answer_range):
        """
        Generates all possible permutations of answers as tuples containing (start_index, end_index, probability).
        Used by average_f1_score function.
        """
        permutations = []
        start, end = answer_range
        for i, p_start in enumerate(start_probs[start: end]):
            for j, p_end in enumerate(end_probs[start: end]):
                if p_end < 0 and p_start < 0:
                    p_end = -p_end
                if i <= j:
                    permutations.append((start + i, start + j + 1, p_start * p_end))
        permutations.sort(key=lambda x: x[2], reverse=True)
        return permutations[:min(len(permutations), 20)]

    def f1_score(self, pred, gt):
        """
        Computes word level f1-score of a predicted phrase against a ground-truth phrase.
        """
        pred_tokens = [token.lower().strip().translate(str.maketrans('', '', string.punctuation)) for token in pred.split()]
        gt_tokens = [token.lower().strip().translate(str.maketrans('', '', string.punctuation)) for token in gt.split()]

        same = [word for word in pred_tokens if word in gt_tokens]

        if len(same) == 0:
            return 0

        precision = len(same) / len(pred_tokens)
        recall = len(same) / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    def get_max_f1(self, pred, answers):
        """
        Compute F1-score of a prediction against all answers and return the f1-score of best match.
        """
        return max([self.f1_score(pred, answer) for answer in answers]) if len(answers) > 0 else 0

    def average_f1_score(self, start_probs, end_probs, question, input_ids, answer_range):
        """
        Computes the average f1-score as explained in README.docx
        """
        permutations = self.generate_permutations(start_probs, end_probs, answer_range)
        f1 = 0.
        total_prob = 0.
        for (start, end, prob) in permutations:
            phrase = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[start:end]))

            this_f1 = self.get_max_f1(phrase, question['answers'])
        
            f1 += prob * this_f1
            total_prob +=  prob
        f1 /= total_prob
        return f1

    def init_state(self, questions, num_searches):
        """
        initializes the self.state variable.
        For each question id present in "questions" variable, generate k=num_searches sequences of words.
        The sequence of words are then passed to score_candidates to compute the scores for the given sequences 
        along-with their predictions, which are then stored in self.state variable.
        """
        candidates = {}
        for q_id, q in questions.items():
            candidates[q_id] = [[random.sample(
                                                self.vocab + [w.lower() for w in word_tokenize(q['question'])], 
                                                NUM_ADDITIONS
                                            )]
                                                for i in range(num_searches)]

        scores, preds = self.score_candidates(questions, candidates)

        self.state = {k: [(c[0], s[0], p[0]) 
                    for c, s, p in zip(candidates[k], scores[k], preds[k])]
                    for k in candidates}

    def reinit_state(self, questions, num_searches): 
        """
        reinitializes the self.state variable.
        Very similar to "init_state" function. Difference is that instead of generating k=num_searches sequences of words, 
        it generates k=(num_searches - "number of existing sequences") sequences of words and appends them to pre-existing data.
        """      
        candidates = {}
        for k in self.state:
            min_words, min_score, min_pred = min(self.state[k], key=lambda x: x[1])
            min_f1 = self.get_max_f1(min_pred, questions[k]['answers'])
            if min_f1 == 0: continue

            num_new = num_searches - len(self.state[k])
            
            candidates[k] = [[random.sample(
                                                self.vocab + [w.lower() for w in word_tokenize(questions[k]['question'])],
                                                NUM_ADDITIONS
                                            )]
                                                for i in range(num_new)]

        scores, preds = self.score_candidates(questions, candidates)
        
        for k in candidates:
            for c, s, p in zip(candidates[k], scores[k], preds[k]):
                self.state[k].append((c[0], s[0], p[0]))
        
    def generate_candidates(self, questions, i):
        """
        For each question, it takes the sequence of words stored inside "self.state" variable and replaces it's i(th) 
        index word with a set of k candidate words to generate k different sentences.
        """
        candidates = {}
        for q_id in self.state.keys():
            min_words, min_score, min_pred = min(self.state[q_id], key=lambda x: x[1])
            min_f1 = self.get_max_f1(min_pred, questions[q_id]['answers'])
            if min_f1 == 0: continue
            candidates[q_id] = []
    
            q = questions[q_id]
            for words, score, pred in self.state[q_id]:
                swap_words = random.sample(self.vocab, NUM_SAMPLE) + [w.lower() for w in word_tokenize(q['question'])]
                q_candidates = []
                for new_word in swap_words:
                    new_words = list(words)
                    new_words[i] = new_word
                    q_candidates.append(new_words)
                candidates[q_id].append(q_candidates)
        return candidates

    def score_candidates(self, questions, candidates):
        """
        Takes the candidate sentences generated using "generate_candidates", adds them to the context of each question,
        and returns the corresponding expected F1-scores and predictions for each candidate sentence.
        """
        scores = {}
        preds = {}
        for q_id, q in questions.items():
            q_scores = []
            q_preds = []
            if q_id not in candidates:
                continue
            for q_candidates in candidates[q_id]:
                q_scores.append([])
                q_preds.append([])
                for candidate  in q_candidates:
                    modified_q = q.copy()
                    modified_q['context'] += " " + " ".join(candidate)
                    
                    if not(modified_q['context'].endswith('.')):
                        modified_q['context'] += '.'
                    
                    score, pred = self.infer(modified_q)

                    q_scores[-1].append(score)
                    q_preds[-1].append(pred)
            scores[q_id] = q_scores
            preds[q_id] = q_preds

        return scores, preds

    def infer(self, question):
        """
        Takes as input a dictionary containing following named values:
            - question
            - context
            - answers
        Returns the predicted answer of the model as a string, and the average expected f1-score of the model.
        """
        inputs = self.tokenizer.encode_plus(question['question'], question['context'], add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

        answer_range = (input_ids.index(3) + 1, len(input_ids) - 1)

        with torch.no_grad():
            answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start_scores = answer_start_scores.cpu()
        answer_end_scores = answer_end_scores.cpu()

        score = self.average_f1_score(answer_start_scores.numpy()[0], answer_end_scores.numpy()[0], question, input_ids, answer_range).item()

        answer_start = answer_range[0] + torch.argmax(answer_start_scores[:, answer_range[0]:answer_range[1]]) # Get the most likely beginning of answer with the argmax of the score
        answer_end = answer_start + torch.argmax(answer_end_scores[:, answer_start:answer_range[1]]) + 1 # Get the most likely end of answer with the argmax of the score
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return score, answer

    def addany_search(self, questions):
        """
        This is the main function that performs adversarial attack. 

        args:
            questions: dict {keys=question ids, value=dict containing {question, answers, context}}
        returns:
            questions in the format of the input, with context modified as per the adversarial attack.
        """
        self.init_state(questions, NUM_SEARCHES_PER_MEGA_EPOCH[0])
        done = False
        for mega_epoch, cur_num_searches in enumerate(NUM_SEARCHES_PER_MEGA_EPOCH):
            # If it is not the first epoch, re-initialize the state to initialize the required number of new sequences.
            if mega_epoch > 0:
                self.reinit_state(questions, cur_num_searches)
            for epoch in range(NUM_EPOCHS):
                print('Mega-Epoch {0}, Epoch {1}:'.format(mega_epoch, epoch))

                indices = list(range(NUM_ADDITIONS))
                random.shuffle(indices)
                for t, idx_swap in enumerate(indices):
                    scores = [min(self.state[k], key=lambda x: x[1])[1] for k in self.state]
                    print('\tIteration {0} (modifying index {1})     Average Expected Score: {2}'.format(t, idx_swap, sum(scores)/ len(scores)))
    
                    if PRINT_RESULTS_PER_ITERATION:
                        for q_id in self.state.keys():
                            min_words, min_score, min_pred = min(self.state[q_id], key=lambda x: x[1])
                            print("\t\t", q_id, ":", "Prediction:", "'" + min_pred + "'", " " * ((20 - len(min_pred)) % 21), "Answer:", str(questions[q_id]['answers']))
 
                            # print("\tF1:", self.get_max_f1(min_pred, questions[q_id]['answers']))
                            # print("\tScore:", min_score)
                            # print("\tNum Particles:", len(self.state[q_id]))
                    
                    candidates = self.generate_candidates(questions, idx_swap)

                    # If no candidates are generated, it means search for all questions has stopped, and so we can break from the loop
                    if len(list(candidates.keys())) == 0:
                        done = True
                        print("All questions recieved an F1-score of 0. Stopping Adverserial Attack...")
                        break

                    scores, preds = self.score_candidates(questions, candidates)
                    
                    new_state = {}
                    
                    for q_id in self.state:
                        # If q_id is not in candidates, it means q_id has an f1_score of 0 and does not need to be processed
                        # further, hence keep its current state as the new state.
                        if q_id not in candidates:
                            new_state[q_id] = self.state[q_id]
                            continue
                        new_state[q_id] = []
                        
                        # Iteration over num_searches
                        for search_num, (cur_cands, cur_scores, cur_preds) in enumerate(zip(candidates[q_id], scores[q_id], preds[q_id])): 
                            min_ind, min_score = min(enumerate(cur_scores), key=lambda x: x[1])
                            if min_score < self.state[q_id][search_num][1]:
                                new_state[q_id].append((cur_cands[min_ind], min_score,
                                                   cur_preds[min_ind]))
                            else:
                                new_state[q_id].append(self.state[q_id][search_num])
                    self.state = new_state
                if done:
                    break
            if done:
                break
        questions2 = {}
        for q_id, q in questions.items():
            words, _, _ = min(self.state[q_id], key=lambda x: x[1])
            context = q['context'] + " " + " ".join(words)
            if not(context.endswith('.')):
                context += '.'
            questions2[q_id] = {'context': context, 'answers': q['answers'], 'question':q['question']}
        return questions2