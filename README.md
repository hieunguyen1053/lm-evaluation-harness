# Language Model Evaluation Harness

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/hieunguyen1053/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

## Host API for evaluation

To host an API for evaluation, run the following command. The API will be hosted at `http://localhost:5000` by default.

```bash
MODEL_PATH=vlsp-2023-vllm/hoa-7b # Replace with your model path or model name
TEAM_NAME=VLSP-team-0 # Replace with your team name
MODEL_SIZE=7 # Replace with your model size 1, 3, 7, 13 (Billion parameters)
MODEL_TYPE=pretrained # Replace with your model type: pretrained, finetuned
python app.py \
    --pretrained $MODEL_PATH \
    --device cuda:0 \
    --team_name $TEAM_NAME \
    --model_size $MODEL_SIZE \
    --model_type $MODEL_TYPE
```

To test the API, run the following command:

```bash
python evaluator.py \
    --task hellaswag_vi \
    --url http://localhost:5000 \
    --num_fewshot 10 \
    --test
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
