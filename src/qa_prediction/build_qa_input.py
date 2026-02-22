import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

import openai
import json



from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math
from typing import Dict, List, Tuple, Optional
import types
from accelerate import Accelerator
import os
import json
import pickle
import numpy as np
import torch
import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import openai
import re


import networkx as nx
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_triples(reasoning_paths):
    triples = []
    for path in reasoning_paths.split("\n"):
        if "->" in path:
            parts = [p.strip() for p in path.split("->")]
            for i in range(len(parts) - 2):
                subject = parts[i]
                relation = parts[i + 1]
                obj = parts[i + 2]
                if subject and relation and obj:  # 确保所有部分都不为空
                    triples.append((subject, relation, obj))
    return triples
def build_relation_graph(triples):
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_edge(s, r)
        G.add_edge(r, o)
    return G


def run_personalized_pagerank(graph, personalized_nodes, alpha=0.85):
    if len(graph.nodes()) == 0:
        return {}

    personalization = {node: 0 for node in graph.nodes()}

    if not any(node in personalization for node in personalized_nodes):
        for node in personalization:
            personalization[node] = 1.0 / len(personalization)
    else:
        found_nodes = [node for node in personalized_nodes if node in personalization]
        if found_nodes:
            weight = 1.0 / len(found_nodes)
            for node in found_nodes:
                personalization[node] = weight

    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except:
        return {node: 1.0 / len(graph.nodes()) for node in graph.nodes()}



try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.eval() 
except Exception as e:
    print(f"Warning: Failed to load language model: {e}")
    model = None
    tokenizer = None

def evaluate_path_with_lm(question, path):
    if model is None or tokenizer is None:
        return 1.0 
        
    try:
        with torch.no_grad():  
            input_text = f"Question: {question} Path: {path}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            relevance_score = outputs.logits.softmax(dim=1)[0, 1].item()
        return relevance_score
    except Exception as e:
        print(f"Warning: LM evaluation failed: {e}")
        return 1.0 

def filter_reasoning_path(question, reasoning_paths, ppr_threshold=0.1, lm_threshold=0.5):
    if not reasoning_paths or not question:
        return reasoning_paths
    
    # 
    original_paths = [path.strip() for path in reasoning_paths.split("\n") if path.strip()]
    
    # 
    triples = extract_triples(reasoning_paths)
    if not triples:
        return reasoning_paths
    
    graph = build_relation_graph(triples)
    
    personalized_nodes = set()
    for word in question.split():
        if word in graph.nodes:
            personalized_nodes.add(word)
    
    ppr_scores = run_personalized_pagerank(graph, personalized_nodes)
    
    if not ppr_scores:
        return reasoning_paths
    
    path_scores = []
    path_endpoints = {}
    for path in original_paths:
        parts = [p.strip() for p in path.split("->")]
        ppr_score = sum(ppr_scores.get(part, 0) for part in parts) / len(parts)
        
        lm_score = evaluate_path_with_lm(question, path)
        
        if ppr_score >= ppr_threshold and lm_score >= lm_threshold:
            combined_score = (ppr_score + lm_score) / 2  
            start, end = parts[0], parts[-1]
            path_scores.append((combined_score, path))
            path_endpoints[path] = (start, end)
        
        # if lm_score >= lm_threshold:
        #     combined_score = lm_score # 平均得分
        #     start, end = parts[0], parts[-1]
        #     path_scores.append((combined_score, path))
        #     path_endpoints[path] = (start, end)
    # 
    path_scores.sort(reverse=True)
    
    # 
    used_endpoints = set()
    filtered_paths = []
    for _, path in path_scores:
        endpoints = path_endpoints[path]
        if endpoints not in used_endpoints:
            filtered_paths.append(path)
            used_endpoints.add(endpoints)
    
    if not filtered_paths:
        return reasoning_paths
    
    return "\n".join(filtered_paths)






def path2text(context):
    openai.api_key = "sk-proj-"  # replace with your actual API key

    prompt = f"""
    Here are some reasoning paths separated by \n.
    {context}
    
    Please convert each reasoning path separated by \n into individual natural language sentences, without summarizing them into one sentence.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    generated_sentences = response['choices'][0]['message']['content']
    return generated_sentences

def save_sentences_to_json(sentences, index):
    filename = f"/egr/research-dselab/guokai1/workspace/G-Retriever/dataset/expla_graphs/graph_summary/gpt3.5/expla_graph_text_{index}.json"

    data = {
        "generated_sentences": sentences
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Sentences saved to {filename}")
import json
import os

def save_sentences_to_json_index(sentence, index):
    
    filename = f"--"

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump([], f)

    with open(filename, "r") as f:
        data = json.load(f)

    while len(data) <= index:
        data.append(None)

   
    data[index] = sentence

   
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Sentence saved to the index {index} in {filename}")

def read_sentence_from_index(filename, index):
  
    with open(filename, "r") as f:
        data = json.load(f)

   
    if index < len(data) and data[index] is not None:
        return data[index]
    else:
        raise IndexError(f"No data found at index {index}")


class PromptBuilder(object):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    # @staticmethod
    # def add_args(parser):
    #     parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
    #                         default='rmanluo/RoG')
    #     parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
    #     parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, prompt_path, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
   
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    # SAQ_RULE_INSTRUCTION = """Please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""

    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""       
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict,idx):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        

        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                # print('lists_of_paths',lists_of_paths)

                # noise_paths = [
                #     'Elon Musk -> people.person.founded -> SpaceX', 
                #     'The Mona Lisa -> visual_art.artwork.displayed_at -> The Louvre', 
                #     'Black Hole -> physics.theory.predicted_by -> Albert Einstein', 
                #     'Amazon River -> geography.river.mouth -> Atlantic Ocean', 
                #     'Python -> programming_language.created_by -> Guido van Rossum', 
                #     'Eiffel Tower -> location.location_containedby -> Paris', 
                #     'Coca-Cola -> business.product.manufacturer -> The Coca-Cola Company', 
                #     'Mount Everest -> geography.mountain.elevation -> 8848 meters', 
                #     'Leonardo da Vinci -> people.person.known_for -> Mona Lisa', 
                #     'Bitcoin -> finance.currency.creator -> Satoshi Nakamoto', 
                #     'Tesla Model S -> automotive.model.manufacturer -> Tesla', 
                #     'Shakespeare -> book.author.written_works -> Hamlet', 
                #     'Jurassic Park -> film.film.director -> Steven Spielberg', 
                #     'Facebook -> business.company.founded_by -> Mark Zuckerberg', 
                #     'Great Wall of China -> location.location_length -> 21,196 km', 
                #     'The Beatles -> music.artist.genre -> Rock', 
                #     'Pluto -> astronomy.celestial_body.classification -> Dwarf planet', 
                #     'Harvard University -> education.university.founded_year -> 1636', 
                #     'Japan -> location.country.official_language -> Japanese', 
                #     'Olympic Games -> sports.sports_event.founded_year -> 1896', 
                #     'New York City -> location.city.population -> 8.5 million', 
                #     'Cristiano Ronaldo -> sports.athlete.plays_for -> Al-Nassr', 
                #     'Genghis Khan -> people.person.founded -> Mongol Empire', 
                #     'Machu Picchu -> location.historic_place.discovered_by -> Hiram Bingham', 
                #     'Google -> business.company.headquarters -> Mountain View, California', 
                #     'Netflix -> business.company.industry -> Streaming Services', 
                #     'The Moon -> astronomy.celestial_body.orbits -> Earth', 
                #     'Pacific Ocean -> geography.body_of_water.area -> 165.25 million km²', 
                #     'Albert Einstein -> people.person.theories -> Theory of Relativity', 
                #     'Wright Brothers -> transportation.aircraft.inventors -> Airplane'
                # ]

                
                # lists_of_paths = noise_paths + lists_of_paths
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
            
        input = self.QUESTION.format(question = question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule:
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            print('idx', idx)
 
            
            input = self.GRAPH_CONTEXT.format(context = context) + input   
        input = self.prompt_template.format(instruction = instruction, input = input)
        # print('input---------', input)
            
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        original_path_count = len(list_of_paths)
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            print('too long------------')
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    final_path_count = len(new_list_of_paths)
                    paths_removed = original_path_count - final_path_count
                    print('too long------------')
                    print(f'Removed {paths_removed} paths: Original paths = {original_path_count}, New paths = {final_path_count}')
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)


    def inference_single_path(self, path_hidden, question_hidden, classifier, scaler):
        """
        Inference function for a single path, returns probability instead of binary prediction
        """
        # Prepare features
        path_embedding = path_hidden.mean(dim=1).squeeze(0).cpu().detach().numpy()
        question_embedding = question_hidden.mean(dim=1).squeeze(0).cpu().detach().numpy()
        features = np.concatenate([path_embedding, question_embedding])
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get probability directly
        probability = classifier.predict_proba(features_scaled)[0][1]
        return probability, probability  # 返回相同的值两次，第一个作为score，第二个作为probability


    def load_path_classifier(self, model_path='cwq_path_classifier_model.pkl'):
        """Load the path classifier model and scaler"""
        # if self.cached_classifier is None or self.cached_scaler is None:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.cached_classifier = model_data['classifier']
        self.cached_scaler = model_data['scaler']
        return self.cached_classifier, self.cached_scaler
    def predict_path_relevance(self,path_hidden, question_hidden):
        # Load saved model
        classifier, scaler = load_model()
        
        # Get prediction
        prediction, probability = inference_single_path(
            path_hidden, 
            question_hidden, 
            classifier, 
            scaler
        )
        
        return prediction, probability
    