
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
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
import time
from functools import partial
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
# from openai import OpenAI
openai.api_key = "sk-proj-"  # replace with your OpenAI API key
with open('entities_names.json') as f:
    entities_names = json.load(f)

class PageRankPathFilter:
    def __init__(self, alpha: float = 0.85, top_k: int = 30):
        """
        Initialize PageRank path filter
        Args:
            alpha: Damping factor for PageRank (default: 0.85)
            top_k: Number of top paths to keep (default: 30)
        """
        self.alpha = alpha
        self.top_k = top_k
        
    def _parse_path(self, path: str) -> List[Tuple[str, str, str]]:
        """Parse path string into list of (entity, relation, entity) triples"""
        steps = [s.strip() for s in path.split('->')]
        triples = []
        for i in range(0, len(steps)-1, 2):
            head = steps[i]
            relation = steps[i+1] 
            tail = steps[i+2]
            triples.append((head, relation, tail))
        return triples

    def _extract_question_entities(self, question: str) -> Set[str]:
        """Extract potential entities from question"""
        if not question:
            return set()
            
        words = question.lower().strip().split()
        skip_words = {'what', 'who', 'where', 'when', 'how', 'why', 'is', 'are', 'was', 'were', 
                     'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could', 'would', 'will',
                     'the', 'a', 'an', 'in', 'on', 'at', 'by', 'to', 'for', 'of', 'with', 'and', '[/inst]'}
        
        entities = set()
        for word in words:
            if word not in skip_words and len(word) > 1:
                entities.add(word)
                
  
        return entities

    def _build_subgraph(self, paths: List[str], question: str = None) -> nx.DiGraph:
        """Build directed graph from paths and question"""
        G = nx.DiGraph()
        
        # Add paths to graph
        for path in paths:
            triples = self._parse_path(path)
            for head, rel, tail in triples:
                # Add edges with base weight
                G.add_edge(head, tail, weight=1.0, relation=rel)
        
        # If question is provided, add question connections
        if question:
            question_entities = self._extract_question_entities(question)
            
            # Add question node
            question_node = "QUESTION_NODE"
            G.add_node(question_node, type='question')
            
            # Match entities with word variations and add weighted edges
            for entity in G.nodes():
                entity_lower = entity.lower()
                if any(qent in entity_lower or entity_lower in qent for qent in question_entities):
                    # Give higher weight to edges related to the question
                    G.add_edge(question_node, entity, weight=2.0, relation='question_entity')
                    G.add_edge(entity, question_node, weight=2.0, relation='entity_question')
                    
                    # Increase the weight of edges connected to question entities
                    for neighbor in G.neighbors(entity):
                        if G.has_edge(entity, neighbor):
                            G[entity][neighbor]['weight'] = 1.5
                    
        return G

    def _pagerank(self, G: nx.DiGraph) -> dict:
        """Calculate PageRank scores with weighted edges"""
        if len(G.nodes()) == 0:
            return {}

        # Add self-loops with weight
        for node in G.nodes():
            G.add_edge(node, node, weight=0.3)
        
        try:
            # Calculate standard PageRank with edge weights
            pr_scores = nx.pagerank(G, 
                                  alpha=self.alpha,
                                  max_iter=1000,
                                  weight='weight')  # Use edge weights
                                   
           
            max_score = max(pr_scores.values())
            if max_score > 0:
                pr_scores = {k: v/max_score for k, v in pr_scores.items()}
            
                
        except:
            print("Warning: PageRank calculation failed, using degree centrality")
            pr_scores = nx.degree_centrality(G)
            
        return pr_scores

    def _score_path(self, path: str, pr_scores: dict) -> float:
        """Score a path based on PageRank scores of its entities"""
        triples = self._parse_path(path)
        entities = [triple[0] for triple in triples] + [triples[-1][2]]
        path_scores = [pr_scores.get(entity, 0.0) for entity in entities]
        return sum(path_scores) / len(path_scores) if path_scores else 0.0

    def filter_paths(self, paths: List[str], question: str = None) -> Tuple[List[str], List[float]]:
        """Filter paths using PageRank scores and select top k paths"""
        if not paths:
            return [], []
            
        # Build subgraph with question information
        G = self._build_subgraph(paths, question)
        
        # Calculate PageRank scores
        pr_scores = self._pagerank(G)
        
        # Score all paths
        path_scores = []
        for path in paths:
            score = self._score_path(path, pr_scores)
            path_scores.append((path, score))
            
        # Sort paths by score and take top k
        all_paths = {path: score for path, score in path_scores}
        sorted_paths = dict(sorted(all_paths.items(), key=lambda x: x[1], reverse=True))

        path_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(self.top_k, len(paths))
        selected_paths = path_scores[:top_k]
        
        # Split into separate lists
        filtered_paths = []
        scores = []
        for path, score in selected_paths:
            filtered_paths.append(path)
            scores.append(score)
            print(f"Keeping path (score={score:.4f}): {path}")
        
        for path, score in path_scores[top_k:]:
            print(f"Removing path (score={score:.4f}): {path}")
                
        return filtered_paths, scores, sorted_paths
class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
                            default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
        parser.add_argument('--model_name1', type=str, help="HUGGING FACE MODEL or model path",
                            default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100

    def load_model(self, **kwargs):
        """Load the core model."""
        model = AutoModelForCausalLM.from_pretrained(self.args.model_path, **kwargs)
        return model

    def tokenize(self, text):
        """Tokenize the input text and return the number of tokens."""
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        """Prepare the model and tokenizer for inference."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            use_auth_token=True,
            use_fast=False,
            # trust_remote_code=True # for qwen
        )

        print("Loading model...")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "7,5,1,2,4,6,0,3"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "6,5,0,3,4,7,1,2"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1,6"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=self.DTYPE.get(self.args.dtype, None),
            device_map="auto",
            # trust_remote_code=True,
            **model_kwargs
        )

        print("Creating generator...")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )


        print("Model and generator initialized.")


    def clean_prompt(self,prompt):
        
        prompt = re.sub(r"\[INST\]\s*<<SYS>>\s*<</SYS>>\s*", "", prompt, flags=re.DOTALL)

     
        prompt = re.sub(r"\s*\[/INST\]\s*", "", prompt, flags=re.DOTALL)
        prompt = prompt.strip()
        modified_input = re.sub(
        r"Based on the reasoning paths, please answer the given question\. Please keep the answer as simple as possible and return all the possible answers as a list\.",
        r"Based on the reasoning paths, please answer the given question. Output only the answer(s), with each answer on a separate line. Do NOT include any explanation, formatting, or additional text. o NOT repeat the same answer multiple times.",
        prompt
        )
        messages = [
        {"role": "system", "content": "You are an AI assistant that strictly follows instructions. You must return only the correct answers, without any explanation or additional text."},
        {"role": "user", "content": modified_input}
        ]

        return messages


 

    @torch.inference_mode()
    def generate_sentence_logits(self, llm_input,alpha):
        """
        Generate a sentence from the input and retrieve the last hidden state representation.
        """

        # print('llm_input',llm_input)
        # generated_output = self.generator(
        #     llm_input,
        #     return_full_text=False,
        #     max_new_tokens=self.args.max_new_tokens
        # )
        # print('text', generated_output[0]['generated_text'])

        reasoning_scores = 0
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        last_layer_hidden_state = torch.zeros(batch_size, seq_length, hidden_dim)

        middle2_hidden_state = last_layer_hidden_state[:, -1, :]

        original_combine_answer = []
        combine_answer = []
        # new_list = generated_output[0]['generated_text'].split('\n')
        # combine_answer.append(new_list)
        # original_combine_answer.append(generated_output[0]['generated_text'])
        #---------------------
        # llm_answer, llm_generated_text, level1 = self._generate_with_confidence(llm_input,0.2)
        # combine_answer.append(llm_answer)
        # original_combine_answer.append(llm_generated_text)

        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        path_start_idx = llm_input.find(path_start_str)
        path_end_idx = llm_input.find(question_str)

        nopath_input = (
            llm_input[:path_start_idx] +
            path_start_str +
            "\n" +
            llm_input[path_end_idx:]
        )
        answer, generated_text, level = self._generate_with_confidence(nopath_input, 0)
        if level == "low":
            print('low')
            combine_answer.append(answer)
            original_combine_answer.append(generated_text)

        flattened1 = [item for sublist in combine_answer for item in sublist]
        deduped = list(dict.fromkeys(flattened1))
        combine_answer = [self.fix_spacing(item) for item in deduped]
        corrected_answers = self.correct_spacing(combine_answer, original_combine_answer)
        flattened = []
        for answer in corrected_answers:
            items = [item.strip() for item in answer.split('\n')]
            flattened.extend(items)
        flattened = list(set(flattened))

        final_answer = "\n".join(flattened)
        print('final_answer', final_answer)
        return final_answer, middle2_hidden_state, level

   
    @torch.inference_mode()               
    def generate_sentence_original(self, llm_input, alpha):
        reasoning_scores = 0
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        last_layer_hidden_state = torch.zeros(batch_size, seq_length, hidden_dim)

        middle2_hidden_state = last_layer_hidden_state[:, -1, :]
        inputs = self.tokenizer(llm_input, return_tensors="pt").to(self.model.device)

        # Generate text using the model directly
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True, 
                # do_sample=False,
                # temperature=1
            )
        generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text,middle2_hidden_state,0




    @torch.inference_mode()
    def generate_sentence_original_gpt(self, llm_input, alpha):
        reasoning_scores = 0
        batch_size = 1
        seq_length = 10
        hidden_dim = 768
        
        last_layer_hidden_state = torch.zeros(batch_size, seq_length, hidden_dim)
        middle2_hidden_state = last_layer_hidden_state[:, -1, :]
        
        # print('input-------------', llm_input)
        
        try:
            import openai
            import re
            
            client = openai.OpenAI(
                api_key="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"  # 替换为您的 OpenAI API 密钥
            )   
            
            formatted_prompt = f"{llm_input}\n\nPlease provide your answer in a simple format without any bullet points, numbers, or symbols. If multiple answers exist, separate them with line breaks only."
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.args.max_new_tokens,
                temperature=0
            )
            
            generated_text = response.choices[0].message.content
            
            # 
            generated_text = self.clean_format(generated_text)
            
        except Exception as e:
            print(f"API fail: {e}")
            generated_text = "Error: Unable to generate response"
        
        return generated_text, middle2_hidden_state, 0

    def clean_format(self, text):
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'^\*\s*', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'^•\s*', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\n\s*\n', '\n', text)
        
        text = text.strip()
        
        return text
    
    def calculate_answer_confidence(self, all_tokens, all_probs):
        """
        Calculate confidence scores for complete movie titles by handling token groups
        """
        answers = []
        current_title = []
        current_probs = []
        
        # Handle tokens sequentially
        for token, prob in zip(all_tokens, all_probs):
            # Check if token is a newline - indicates end of title
            if token == '\n':
                if current_title:
                    # Join the current title tokens and calculate average confidence
                    title = ''.join(current_title).strip()
                    if title:  # Only add non-empty titles
                        avg_confidence = sum(current_probs) / len(current_probs)
                        answers.append({
                            'answer': title,
                            'confidence': avg_confidence
                        })
                    current_title = []
                    current_probs = []
            else:
                current_title.append(token)
                current_probs.append(prob)
        
        # Handle any remaining title at the end
        if current_title:
            title = ''.join(current_title).strip()
            if title:
                avg_confidence = sum(current_probs) / len(current_probs)
                answers.append({
                    'answer': title,
                    'confidence': avg_confidence
                })
        
        return answers

    def fix_spacing(self, text):
        """
        Fix spacing issues in movie titles
        """
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Handle special cases for numbers (e.g. "400 Blows")
        text = re.sub(r'(\d+)\s+([A-Za-z])', r'\1\2', text)
        
        return text

    def correct_spacing(self, processed_answers, original_answers):
        """
        Correct spacing using original answers as reference
        """
        # Split original answers by newline and clean
        flat_original = []
        for answer in original_answers:
            titles = [title.strip() for title in answer.split('\n') if title.strip()]
            flat_original.extend(titles)
        
        def find_original_match(processed_str):
            # Remove all spaces and punctuation for comparison
            no_space_processed = re.sub(r'[^\w\s]', '', ''.join(processed_str.split()))
            
            # Find matching original by comparing without spaces and punctuation
            for original in flat_original:
                if re.sub(r'[^\w\s]', '', ''.join(original.split())) == no_space_processed:
                    return original
            return processed_str
        
        # Apply correction to each processed answer
        corrected_answers = [find_original_match(answer) for answer in processed_answers]
        return corrected_answers
    def extract_tail_entity(self, path: str) -> str:
        """提取路径的尾部实体"""
        parts = path.split(' -> ')
        return parts[-1].strip() if parts else ""


    def pick_relevant_paths_with_llama(self, paths, question):
        """
        Modified version that uses Llama 2 7B Chat HF for path selection
        """
        start_time = time.time()
        
        # Initialize model and tokenizer

        # Create numbered paths dictionary
        numbered_paths = {i+1: path for i, path in enumerate(paths)}
        
        # Create numbered paths text
        numbered_paths_text = "\n".join(f"{i}. {path}" for i, path in numbered_paths.items())
        
        # Construct prompt
    #     prompt = f"""<s>[INST] Only numbers as a comma-separated list! You are a reasoning assistant. Analyze these numbered reasoning paths and return only the numbers of useful paths. Do not include any answers. 

    # Question:
    # {question}

    # Numbered Reasoning Paths:
    # {numbered_paths_text}

    # Only numbers as a comma-separated list! Return only the numbers of useful paths as a comma-separated list, with no explanation or additional text. Only numbers as a comma-separated list![/INST]"""
        prompt = f"""<s>[INST] You are a reasoning assistant. Analyze these numbered reasoning paths and return only the numbers of useful paths.

        Question:
        {question}

        Numbered Reasoning Paths:
        {numbered_paths_text}

        For example, given the following reasoning paths:
        1. JaMarcus Russell -> people.person.place_of_birth -> Mobile
        2. JaMarcus Russell -> people.person.nationality -> United States of America
        3. JaMarcus Russell -> place_of_birth -> Mobile

        And the question:
        where is jamarcus russell from?

        The correct answer would be: 1, 3

        Return only the numbers of useful paths as a comma-separated list, with no explanation or additional text. [/INST]"""

        # # Tokenize and generate response
        # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # with torch.no_grad():
        #     outputs = model.generate(
        #         inputs.input_ids,
        #         max_new_tokens=50,
        #         temperature=0.2,
        #         do_sample=False,
        #         pad_token_id=tokenizer.eos_token_id
        #     )
        outputs = self.generator(
            prompt,
            return_full_text=False,
            max_new_tokens=self.args.max_new_tokens
        )
        # Decode response
        # number_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # # Extract just the generated response (after the prompt)
        # number_response = number_response[len(prompt):]
        number_response = outputs[0]['generated_text']

        # If response is empty, return empty list
        if not number_response or number_response.isspace():
            return []

        try:
            # Extract numbers from response
            selected_numbers = [int(num.strip()) for num in number_response.split(',') if num.strip()]
            selected_paths = [numbered_paths[num] for num in selected_numbers if num in numbered_paths]
            return selected_paths
        except:
            return []

        finally:
            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def pick_relevant_paths_with_gpt4o(self, paths, question):
        """
        Modified version that uses numbered paths for efficiency
        """
        start_time = time.time()
        
        # Create numbered paths dictionary
        numbered_paths = {i+1: path for i, path in enumerate(paths)}
        
        # Create numbered paths text
        numbered_paths_text = "\n".join(f"{i}. {path}" for i, path in numbered_paths.items())
        
        # Construct messages with numbered paths
        messages = [
            {"role": "system", "content": "You are a reasoning assistant. Your task is to analyze numbered reasoning paths and return only the numbers of useful paths."},
            {"role": "user", "content": f"Question:\n{question}\n\nNumbered Reasoning Paths:\n{numbered_paths_text}\n\nReturn only the numbers of useful paths as a comma-separated list, with no explanation or additional text."}
        ]

        client = openai.OpenAI(
            api_key="sk-proj"  # replace with your OpenAI API key
        )  
        # response = openai.ChatCompletion.create(
        response = client.chat.completions.create(
            model="gpt-4o",  # 
            messages=messages,
            temperature=0.2, #
            max_tokens=1000
        )

        # number_response = response["choices"][0]["message"]["content"]
        number_response = response.choices[0].message.content

        if not number_response or number_response.isspace():
            return []

        try:
            selected_numbers = [int(num.strip()) for num in number_response.split(',') if num.strip()]
            selected_paths = [numbered_paths[num] for num in selected_numbers if num in numbered_paths]
            return selected_paths
        except:
            return []

    def parse_input(self,text: str) -> tuple:
        """
        Parse input text to extract reasoning paths and the question.
        """
        clean_text = text.replace('<</SYS>>', '').replace('[INST]', '').replace('[/INST]', '')
        parts = clean_text.split('Question:')
        if len(parts) != 2:
            return [], ''
        
        paths_text = parts[0].strip()
        question = parts[1].strip()
        
        if 'Reasoning Paths:' in paths_text:
            paths_text = paths_text.split('Reasoning Paths:')[1].strip()
        paths = [line.strip() for line in paths_text.split('\n') if '->' in line]
        return paths, question
 
        
    def _generate_with_confidence(self, input_text, alpha):
        """
        Helper method to generate text and calculate confidence score
        """
        SEED = 42 
        import torch
        import random
        import numpy as np
 


        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate text using the model directly
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True, 
                # do_sample=False,
                # temperature=1
            )

        generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        batch_size = 32
        all_tokens = []
        all_probs = []
        


        # Process all tokens at once
        scores = outputs.scores  # All scores
        token_ids = generated_token_ids  # All token IDs

        eps = 1e-9
        temperature = 2.0  
        max_entropy = float('-inf')


        for step_scores, token_id in zip(scores, token_ids):
            step_scores = step_scores - step_scores.max()
            # print('step_scores',step_scores)
            
           
            step_scores = torch.where(
                torch.isinf(step_scores),
                torch.tensor(-100.0, dtype=step_scores.dtype), 
                step_scores
            )
            
            probs = F.softmax(step_scores / temperature, dim=-1)

            entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1).item()
            if entropy > max_entropy:
                    max_entropy = entropy
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            all_tokens.append(token_str)
            all_probs.append(probs.max().item())#max

            del probs

        # Final cleanup
        torch.cuda.empty_cache()
        
        # 9. Calculate answer confidences and filter by threshold
        answer_confidences = self.calculate_answer_confidence(all_tokens, all_probs)
        filtered_answers = []
        confidence_values = []
        # Filter answers with confidence > 0.85
        for answer in answer_confidences:
            # if alpha != 1:
            confidence_values.append(answer['confidence'])

            if answer['confidence'] >= alpha:
                filtered_answers.append(answer['answer'])
     
        if all(conf == 1 for conf in confidence_values):
            level = 'low'
        elif all(conf < 0.9 for conf in confidence_values):
            level = 'high'
        else:
            level = 'middle'
        
        
        # 10. Clean up all intermediate variables
        del outputs
        del all_tokens
        del all_probs
        del answer_confidences
        torch.cuda.empty_cache()
        
        # 11. Disable gradient checkpointing
        self.model.gradient_checkpointing_disable()    

        
        return filtered_answers, generated_text, level
    #############################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    @torch.inference_mode()
    def generate_sentence_attention(self, llm_input,alpha):
        original_forward = self.model.forward
        target_layer = self.model.config.num_hidden_layers // 2 + 2

        def custom_forward(*args, **kwargs):
            def custom_attention_hook(module, input, output):
                return (output[0], None, output[2]) 

            handles = []
            for i, layer in enumerate(self.model.model.layers):
                if i != target_layer:
                    handle = layer.self_attn.register_forward_hook(custom_attention_hook)
                    handles.append(handle)

            try:
                output = original_forward(*args, **kwargs)
                return output
            finally:
                for handle in handles:
                    handle.remove()

        try:
            self.model.forward = custom_forward

            torch.cuda.empty_cache()
            
            inputs = self.tokenizer(
                llm_input, 
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.model.device)

            self.model.gradient_checkpointing_enable()
            
            with torch.no_grad():
                self.model = self.model.half()
                output = self.model(**inputs, output_attentions=True)
                
                first_token_attention = output.attentions[target_layer]
                

        finally:
            self.model.forward = original_forward
            torch.cuda.empty_cache()

        #       ——----------------------------
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        hidden_states = torch.zeros(batch_size, seq_length, hidden_dim)

        
      
        input_text = llm_input
        reasoning_scores = {}

        start_idx = input_text.find("Reasoning Paths:")
        end_idx = input_text.find("Question:")
        #_______--------------------------
        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        path_start_idx = llm_input.find(path_start_str)
        path_end_idx = llm_input.find(question_str)
        
        paths_content = llm_input[path_start_idx + len(path_start_str):path_end_idx].strip()


        if len(paths_content) == 0:
            combine_answer_none = []
            original_combine_answer_none = []
            # If no paths found, just generate normally
            generated_output = self.generator(
                llm_input,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens
            )
            return generated_output[0]['generated_text'], hidden_states, 0

        question_end_idx = len(input_text)
        

        print('have path')
        question_text = input_text[end_idx:question_end_idx]
        question_start = len(self.tokenizer.encode(input_text[:end_idx], add_special_tokens=True))
        paths_text = input_text[start_idx:end_idx].strip()
        paths = [p.strip() for p in paths_text.split('\n')[1:] if p.strip()]
        
        try:
            for path in paths:
                if not path:
                    continue
                path_start = input_text.find(path)
                if path_start != -1:
                    token_start = len(self.tokenizer.encode(input_text[:path_start], add_special_tokens=True)) - 1
                    path_tokens = self.tokenizer.encode(path, add_special_tokens=False)
                    token_end = token_start + len(path_tokens) + 1
                    
                    path_attention = first_token_attention[0, :, -1, token_start:token_end]
                    
                    max_attention = path_attention.max()
                    mean_attention = path_attention.mean()
                    final_score = (max_attention + mean_attention) / 2
                    reasoning_scores[path] = final_score.item()
     
                    
        finally:
            del first_token_attention
            torch.cuda.empty_cache()
        
        sorted_paths = dict(sorted(reasoning_scores.items(), key=lambda x: x[1], reverse=True))



        top_n = min(50, len(sorted_paths))
        top_paths = list(sorted_paths.keys())[:top_n]


        prefix = input_text[:start_idx + len("Reasoning Paths:\n")]
        suffix = input_text[end_idx:]

        secondary_paths = "\n".join(top_paths)


        new_input = prefix + secondary_paths + "\n\n" + suffix

        new_generated_output = self.generator(
            new_input,
            return_full_text=False,
            max_new_tokens=self.args.max_new_tokens
        )


        return new_generated_output[0]['generated_text'], hidden_states, sorted_paths # type: ignore

    @torch.inference_mode()
    def generate_sentence_FI(self, llm_input,alpha):
   
        time_start=time.time()
        original_forward = self.model.forward
        target_layer = self.model.config.num_hidden_layers // 2 + 2
        
        def custom_forward(*args, **kwargs):
            def custom_attention_hook(module, input, output):
                return (output[0], None, output[2])  

            handles = []
            for i, layer in enumerate(self.model.model.layers):
                if i != target_layer:
                    handle = layer.self_attn.register_forward_hook(custom_attention_hook)
                    handles.append(handle)

            try:
                output = original_forward(*args, **kwargs)
                return output
            finally:
                for handle in handles:
                    handle.remove()

        try:
            self.model.forward = custom_forward

            torch.cuda.empty_cache()
            
            inputs = self.tokenizer(
                llm_input, 
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.model.device)

            self.model.gradient_checkpointing_enable()
            
            with torch.no_grad():
                self.model = self.model.half()
                output = self.model(**inputs, output_attentions=True)
                
                first_token_attention = output.attentions[target_layer]
                

        finally:
            self.model.forward = original_forward
            torch.cuda.empty_cache()

        #       ——----------------------------
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        hidden_states = torch.zeros(batch_size, seq_length, hidden_dim)

        

        input_text = llm_input
        reasoning_scores = {}

        start_idx = input_text.find("Reasoning Paths:")
        end_idx = input_text.find("Question:")
        #_______--------------------------
        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        path_start_idx = llm_input.find(path_start_str)
        path_end_idx = llm_input.find(question_str)
        
        paths_content = llm_input[path_start_idx + len(path_start_str):path_end_idx].strip()


        if len(paths_content) == 0:
            combine_answer_none = []
            original_combine_answer_none = []
   
            llm_answer, llm_generated_text, level_none = self._generate_with_confidence(llm_input,0.5)
            combine_answer_none.append(llm_answer)
            original_combine_answer_none.append(llm_generated_text)
            flattened1 = [item for sublist in combine_answer_none for item in sublist]
            deduped = list(dict.fromkeys(flattened1))
            combine_answer_none = [self.fix_spacing(item) for item in deduped]
            corrected_answers = self.correct_spacing(combine_answer_none, original_combine_answer_none)
            flattened_none = []
            for answer in corrected_answers:
                items = [item.strip() for item in answer.split('\n')]
                flattened_none.extend(items)
            flattened_none = list(set(flattened_none))

            final_answer_none = "\n".join(flattened_none)

            return final_answer_none, hidden_states, 0
        question_end_idx = len(input_text)
        

        
        question_text = input_text[end_idx:question_end_idx]
        question_start = len(self.tokenizer.encode(input_text[:end_idx], add_special_tokens=True))
        paths_text = input_text[start_idx:end_idx].strip()
        paths = [p.strip() for p in paths_text.split('\n')[1:] if p.strip()]
        
        try:
            for path in paths:
                if not path:
                    continue
                path_start = input_text.find(path)
                if path_start != -1:
                    token_start = len(self.tokenizer.encode(input_text[:path_start], add_special_tokens=True)) - 1
                    path_tokens = self.tokenizer.encode(path, add_special_tokens=False)
                    token_end = token_start + len(path_tokens) + 1
                    
                    path_attention = first_token_attention[0, :, -1, token_start:token_end]
                    
                    max_attention = path_attention.max()
                    mean_attention = path_attention.mean()
                    final_score = (max_attention + mean_attention) / 2
                    reasoning_scores[path] = final_score.item()
             
                    
        finally:
            del first_token_attention
            torch.cuda.empty_cache()
        time_end=time.time()
        sorted_paths = dict(sorted(reasoning_scores.items(), key=lambda x: x[1], reverse=True))


        top_n = min(40, len(sorted_paths)) 
        top_paths = list(sorted_paths.keys())[:top_n]
     
        answer = self.pick_relevant_paths_with_gpt4o(top_paths, question_text)
        if not answer:  
            gpt_path = []
        else:
            gpt_path = [line.lstrip("- ").strip() for line in answer]

        prefix = input_text[:start_idx + len("Reasoning Paths:\n")]
        suffix = input_text[end_idx:]

       
        primary_paths = "\n".join(gpt_path)
       
        secondary_paths = "\n".join(top_paths)
        new_paths_text = (
            "### High Priority Paths:\n" 
            f"{primary_paths}\n\n"
            "### Additional Paths:\n" 
            f"{secondary_paths}"
        )

        new_input = prefix + new_paths_text + "\n\n" + suffix

        original_combine_answer = []
        combine_answer = []

        llm_answer, llm_generated_text, level1 = self._generate_with_confidence(new_input,alpha)
        combine_answer.append(llm_answer)
        original_combine_answer.append(llm_generated_text)

        nopath_input = (
            llm_input[:path_start_idx] +
            path_start_str +
            "\n" +
            llm_input[path_end_idx:]
        )
        answer, generated_text, level = self._generate_with_confidence(nopath_input, 1)
        if level == "low":
            print('low')
            combine_answer.append(answer)
            original_combine_answer.append(generated_text)

        flattened1 = [item for sublist in combine_answer for item in sublist]
        deduped = list(dict.fromkeys(flattened1))
        combine_answer = [self.fix_spacing(item) for item in deduped]
        corrected_answers = self.correct_spacing(combine_answer, original_combine_answer)
        flattened = []
        for answer in corrected_answers:
            items = [item.strip() for item in answer.split('\n')]
            flattened.extend(items)
        flattened = list(set(flattened))

        
        final_answer = "\n".join(flattened)
        return final_answer, hidden_states, level

        # return new_generated_output[0]['generated_text'], hidden_states, 0 # type: ignore
    @torch.inference_mode()
    def inference_again(self, input, original_input):
        start_idx = original_input.find("Reasoning Paths:")
        end_idx = original_input.find("Question:")
        prefix = original_input[:start_idx + len("Reasoning Paths:\n")]
        suffix = original_input[end_idx:]
        input_path = "\n".join(input)
        new_paths_text = (
            f"{input_path}\n"
        )
        new_input = prefix + new_paths_text + "\n" + suffix
        original_forward = self.model.forward
        target_layer = self.model.config.num_hidden_layers // 2 + 2

        def custom_forward(*args, **kwargs):
            def custom_attention_hook(module, input, output):
                return (output[0], None, output[2])  

            handles = []
            for i, layer in enumerate(self.model.model.layers):
                if i != target_layer:
                    handle = layer.self_attn.register_forward_hook(custom_attention_hook)
                    handles.append(handle)

            try:
                output = original_forward(*args, **kwargs)
                return output
            finally:
                for handle in handles:
                    handle.remove()

        try:
            self.model.forward = custom_forward

            torch.cuda.empty_cache()
            
            inputs = self.tokenizer(
                new_input, 
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.model.device)

            self.model.gradient_checkpointing_enable()
            
            with torch.no_grad():
                self.model = self.model.half()
                output = self.model(**inputs, output_attentions=True)
                
                first_token_attention = output.attentions[target_layer]
                

        finally:
            self.model.forward = original_forward
            torch.cuda.empty_cache()

        #       ——----------------------------
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        hidden_states = torch.zeros(batch_size, seq_length, hidden_dim)

        
  
        input_text = original_input
        reasoning_scores = {}

        start_idx = input_text.find("Reasoning Paths:")
        end_idx = input_text.find("Question:")
        #_______--------------------------
        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        path_start_idx = original_input.find(path_start_str)
        path_end_idx = original_input.find(question_str)
        
        paths_content = original_input[path_start_idx + len(path_start_str):path_end_idx].strip()

  

        question_end_idx = len(input_text)
        question_text = input_text[end_idx:question_end_idx]
        question_start = len(self.tokenizer.encode(input_text[:end_idx], add_special_tokens=True))
        paths_text = input_text[start_idx:end_idx].strip()
        paths = [p.strip() for p in paths_text.split('\n')[1:] if p.strip()]
        
        try:
            for path in paths:
                if not path:
                    continue
                path_start = input_text.find(path)
                if path_start != -1:
                    token_start = len(self.tokenizer.encode(input_text[:path_start], add_special_tokens=True)) - 1
                    path_tokens = self.tokenizer.encode(path, add_special_tokens=False)
                    token_end = token_start + len(path_tokens) + 1
                    
                    path_attention = first_token_attention[0, :, -1, token_start:token_end]
                    
                    if path_attention.numel() == 0:
                        continue
                    max_attention = path_attention.max()
                    mean_attention = path_attention.mean()
                    final_score = (max_attention + mean_attention) / 2
                    reasoning_scores[path] = final_score.item()    
        finally:
            del first_token_attention
            torch.cuda.empty_cache()
        
        sorted_paths = dict(sorted(reasoning_scores.items(), key=lambda x: x[1], reverse=True))
        top_n = min(3, len(sorted_paths))
        top_paths = list(sorted_paths.keys())[:top_n]
        return top_paths




    def encode_text(self, sbert_model, text: str) -> torch.Tensor:
      

        with torch.no_grad():
            embeddings = sbert_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,  
                device=self.model.device
            )
            return embeddings

    def calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        计算两个嵌入向量之间的余弦相似度
        """
        return F.cosine_similarity(embedding1, embedding2, dim=1).item()


    @torch.inference_mode()
    def generate_sentence_with_similarity(self,llm_input, alpha):


        #       ——----------------------------
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

        hidden_states = torch.zeros(batch_size, seq_length, hidden_dim)

        
    
        input_text = llm_input
        reasoning_scores = {}

        start_idx = input_text.find("Reasoning Paths:")
        end_idx = input_text.find("Question:")
        #_______--------------------------

        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        path_start_idx = llm_input.find(path_start_str)
        path_end_idx = llm_input.find(question_str)
        
        paths_content = llm_input[path_start_idx + len(path_start_str):path_end_idx].strip()

        # if path_start_idx == -1 or path_end_idx == -1:

        if len(paths_content) == 0:
            combine_answer_none = []
            original_combine_answer_none = []
            # If no paths found, just generate normally
            generated_output = self.generator(
                llm_input,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens
            )
            return generated_output[0]['generated_text'], hidden_states, 0

        question_end_idx = len(input_text)
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        sbert_model.to(self.model.device)

        question_text = input_text[end_idx:question_end_idx]
        question_start = len(self.tokenizer.encode(input_text[:end_idx], add_special_tokens=True))
        paths_text = input_text[start_idx:end_idx].strip()
        paths = [p.strip() for p in paths_text.split('\n')[1:] if p.strip()]
        
        reasoning_scores = {}
        try:
            question_embedding = self.encode_text(sbert_model, question_text)
            
            for path in paths:
                if not path:
                    continue
                
                path_embedding = self.encode_text(sbert_model, path)
                
                similarity = F.cosine_similarity(question_embedding, path_embedding, dim=0)
                reasoning_scores[path] = similarity.item()
                    
        finally:
            torch.cuda.empty_cache()
        
        sorted_paths = dict(sorted(reasoning_scores.items(), key=lambda x: x[1], reverse=True))

        top_n = min(40, len(sorted_paths))
        top_paths = list(sorted_paths.keys())[:top_n]

        
        prefix = input_text[:start_idx + len("Reasoning Paths:\n")]
        suffix = input_text[end_idx:]
        new_paths_text = "\n".join(top_paths)
        new_input = prefix + new_paths_text + "\n\n" + suffix



        inputs = self.tokenizer(new_input, return_tensors="pt").to(self.model.device)

        # Generate text using the model directly
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                return_dict_in_generate=True,
                # output_scores=True,
                # output_hidden_states=True, 
                # do_sample=False,
                # temperature=1
            )
        generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text, hidden_states, sorted_paths # type: ignore

        # return new_generated_output[0]['generated_text'], hidden_states, 0 # type: ignore


##########0-------------pr-------------------------------

    @torch.inference_mode()
    def generate_sentence_with_page(self, llm_input, alpha,use_question: bool = True):
        """
        Generate response with PPR-based path filtering
        Args:
            llm_input: Input text
            use_question: Whether to use question for path filtering
        """
        reasoning_scores = 0
        batch_size = 1
        seq_length = 10
        hidden_dim = 768

       
        last_layer_hidden_state = torch.zeros(batch_size, seq_length, hidden_dim)

        middle2_hidden_state = last_layer_hidden_state[:, -1, :]
  
        ppr_filter = PageRankPathFilter(alpha=0.85, top_k=50)
        
        inputs = self.tokenizer(llm_input, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        path_start_str = "Reasoning Paths:"
        question_str = "\n\nQuestion:"
        
        path_start_idx = llm_input.find(path_start_str)
        path_end_idx = llm_input.find(question_str)
        
        paths_content = llm_input[path_start_idx + len(path_start_str):path_end_idx].strip()


        if len(paths_content) == 0:
            combine_answer_none = []
            original_combine_answer_none = []
     
            generated_output = self.generator(
                llm_input,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens
            )
            return generated_output[0]['generated_text'], hidden_states, 0
            
        # Extract question
        question_start_idx = path_end_idx + len(question_str)
        question_text = llm_input[question_start_idx:].strip() if use_question else None
        
        # Process paths
        paths_text = llm_input[path_start_idx + len(path_start_str) + 1:path_end_idx]
        paths = [p.strip() for p in paths_text.split('\n') if p.strip()]
        
        # Filter paths using PPR
        filtered_paths, path_scores, sorted_paths = ppr_filter.filter_paths(paths, question_text)
        
        # Print stats

        for path, score in zip(filtered_paths, path_scores):
            print(f"Path: {path}")
            print(f"Score: {score:.4f}\n")
        
        # Reconstruct input with filtered paths
        new_input = (
            llm_input[:path_start_idx] + 
            path_start_str + "\n" + 
            "\n".join(filtered_paths) + 
            llm_input[path_end_idx:]
        )
        
        inputs = self.tokenizer(new_input, return_tensors="pt").to(self.model.device)

        # Generate text using the model directly
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                return_dict_in_generate=True,
                # output_scores=True,
                # output_hidden_states=True, 
                # do_sample=False,
                # temperature=1
            )
        generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text, hidden_states, sorted_paths # type: ignore

