import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
import shutil

def get_output_file(path, force=True):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    print('qa_dataset',qa_dataset)
    return qa_dataset

def merge_rule_result_iter(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()

    qid = data["id"]
    predicted_paths = data["prediction"]
    ground_paths = data["ground_paths"]
    question_to_rule[qid] = {
        "predicted_paths": predicted_paths,
        "ground_paths": ground_paths,
    }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    print('qa_dataset',qa_dataset)
    return qa_dataset


def prediction(data, processed_list, input_builder, model,idx, method_type, alpha):
    question = data["question"]
    answer = data["answer"]
    print('answer', answer)
    print('Number of answers:', len(answer))
    id = data["id"]
    # if id in processed_list:
    #     return None
    if model is None:
        prediction,last_hidden_state, score = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    input = input_builder.process_input(data,idx)

    if method_type == "FI":
        prediction,last_hidden_state, score = model.generate_sentence_FI(input,alpha)

    if method_type == "original":
        prediction,last_hidden_state, score = model.generate_sentence_original(input,alpha)

    if method_type == "simi":
        prediction,last_hidden_state, score = model.generate_sentence_with_similarity(input,alpha)
    if method_type == "page":
        prediction,last_hidden_state, score = model.generate_sentence_with_page(input,alpha)

    # prediction,last_hidden_state, score = model.generate_sentence_with_similarity(input)
    print('score--', score)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
        "hidden_state": last_hidden_state,
        "score": score,
    }
    return result
def main(args, LLM):
    print('llm', LLM)
    method_type = args.method_type
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    # dataset = load_dataset(input_file, split='train')
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        print('rule_post\n', rule_postfix)
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        # dataset = merge_rule_result(train_data, rule_dataset, args.n, args.filter_empty)
        print('dataset', dataset)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # output_file = os.path.join(output_dir, f"predictions_simi40_web.jsonl")#########生成reasoning path
    output_file = os.path.join(output_dir, f"predictions_{args.method_type}_{args.alpha}.jsonl")#########生成reasoning path

    fout, processed_list = get_output_file(output_file, force=args.force)
    hidden_states = []
    scores = []
    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        count = 0


        for idx, data in enumerate(tqdm(dataset)):
            print('idx',idx)


            results = prediction(data, processed_list, input_builder, model, idx, args.method_type, args.alpha)
            if results["score"] != "low":
                continue
            # print('res', res)
            if results is not None:
                
                if isinstance(results, list):
                    for res in results:
                        if "hidden_state" in res:
                            hidden_states.append(res["hidden_state"].cpu())
                        if args.debug:
                            print(json.dumps(res))
                        res_to_write = {k: v for k, v in res.items() if k != "hidden_state"}
                        fout.write(json.dumps(res_to_write) + "\n")
                        fout.flush()
                else:
                    
                    print("all--path")
                    if "hidden_state" in results:
                        hidden_states.append(results["hidden_state"].cpu())
                    if "score" in results:
                        scores.append(results["score"])
                    if args.debug:
                        print(json.dumps(results))
                    results_to_write = {k: v for k, v in results.items() if k not in ["hidden_state"]}
                    fout.write(json.dumps(results_to_write) + "\n")
                    fout.flush()
            # count += 1  
            # if count >= 200:  
            #     break



    fout.close()
    backup_file_path = output_file + ".backup"

    # copy the output file to backup location
    shutil.copy(output_file, backup_file_path)
    eval_result(output_file, args.method_type, args.alpha)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--method_type", type=str, default="origianl")
    argparser.add_argument("--alpha", type=float, default=99)
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)


