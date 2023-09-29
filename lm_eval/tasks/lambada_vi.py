from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity


class Lambada_vi(Task):
    VERSION = None
    DATASET_PATH = "vlsp-2023-vllm/lambada_vi"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["context"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["target_word"]

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy

    def process_results(self, doc, results):
        ll, is_greedy = results

        return {"ppl": ll, "acc": int(is_greedy)}

    def aggregation(self):
        return {"ppl": perplexity, "acc": mean}

    def higher_is_better(self):
        return {"ppl": False, "acc": True}