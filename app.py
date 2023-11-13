import argparse

from flask import Flask, request
import lm_eval.models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--secret", type=str, required=True)

    parser.add_argument("--team_name", type=str, required=True)
    parser.add_argument("--model_size", type=int, required=True, choices=[1, 3, 7, 13])
    parser.add_argument("--model_type", type=str, required=True, choices=["pretrained", "finetuned"])
    return parser.parse_args()

app = Flask(__name__)

@app.route("/info", methods=["GET"])
def info():
    return {
        "team_name": args.team_name,
        "model_size": args.model_size,
        "model_type": args.model_type,
    }

@app.route("/loglikelihood", methods=["POST"])
def loglikelihood():
    data = request.get_json()
    args = data.get("args", None)
    return lm.loglikelihood(args)

@app.route("/greedy_until", methods=["POST"])
def greedy_until():
    data = request.get_json()
    args = data.get("args", None)
    return lm.greedy_until(args)

@app.before_request
def check_secret():
    if request.headers.get("Authorization") != f"{args.secret}":
        return "Unauthorized", 401

if __name__ == "__main__":
    args = parse_args()

    lm = lm_eval.models.get_model("hf-causal")(
        pretrained=args.pretrained,
        batch_size=1,
        max_batch_size=1,
    )

    app.run(
        host=args.host,
        port=args.port
    )