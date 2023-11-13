import argparse
import random
import json

import requests
from tqdm import tqdm
import lm_eval.tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--url", type=str, default="http://localhost:5000")
    parser.add_argument("--secret", type=str, required=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    headers = {"Authorization": args.secret}

    response = requests.get(f"{args.url}/info", headers=headers)
    info = response.json()

    team_name = info["team_name"]
    model_size = info["model_size"]
    model_type = info["model_type"]

    task = lm_eval.tasks.get_task(args.task)()

    if task.has_test_docs():
        task_set = "test"  # Required for caching in the decontamination
        task_doc_func = task.test_docs
    elif task.has_validation_docs():
        task_set = "val"  # Required for caching in the decontamination
        task_doc_func = task.validation_docs

    final_results = {}

    task_docs = list(task_doc_func())

    # Cut down on the number of docs if we're testing
    if args.test:
        task_docs = task_docs[:20]

    rnd = random.Random()
    rnd.seed(args.seed)
    rnd.shuffle(task_docs)

    pbar = tqdm(total=len(task_docs))

    for doc in task_docs:
        pbar.update(1)

        ctx = task.fewshot_context(doc=doc, num_fewshot=args.num_fewshot, rnd=rnd, description="")
        reqs = task.construct_requests(doc, ctx)
        if isinstance(reqs, list):
            request_type = reqs[0].request_type
            data_json = {"args": [r.args for r in reqs]}
        elif isinstance(reqs, tuple):
            request_type = reqs[0].request_type
            data_json = {"args": [reqs[0].args]}
        else:
            request_type = reqs.request_type
            data_json = {"args": [reqs.args]}

        response = requests.post(f"{args.url}/{request_type}", json=data_json, headers=headers)
        results = response.json()

        if args.task == "lambada_vi":
            results = results[0]
        elif request_type == "greedy_until":
            results = [results[0]]
        elif request_type == "loglikelihood":
            results = [result[0] for result in results]

        results = task.process_results(doc, results)

        for key, value in results.items():
            if key not in final_results:
                final_results[key] = []
            final_results[key].append(value)

        postfix = {key: task.aggregation()[key](value) for key, value in final_results.items()}
        pbar.set_postfix(postfix)
    pbar.close()

    if args.output is None:
        args.output = f"{team_name}_{model_type}_{model_size}_{args.task}.json"

    # Aggregate results
    final_results = {key: task.aggregation()[key](value) for key, value in final_results.items()}

    # Write results to file
    with open(args.output, "w") as f:
        final_results["task"] = args.task
        final_results["team_name"] = team_name
        final_results["model_size"] = model_size
        final_results["model_type"] = model_type
        final_results["seed"] = args.seed
        final_results["num_fewshot"] = args.num_fewshot
        json.dump(final_results, f)