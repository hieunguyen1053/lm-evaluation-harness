from sacrebleu import sacrebleu
from lm_eval import metrics
from lm_eval.base import Task, rf

class Translation_vi(Task):
    VERSION = 0
    DATASET_PATH = "vlsp-2023-vllm/en-to-vi-formal-informal-tranlations"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["val"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        fewshot_samples_text = "Dịch từ tiếng Anh sang tiếng Việt\n"
        for d in doc["fewshot_samples"]:
            fewshot_samples_text += "en: " + d["en"] + "\n" + "vi: " + d["vi"] + "\n\n"
        text = fewshot_samples_text + "en: " + doc["en"].strip() + "\n" + "vi:"
        return text

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["en"].strip()

    def doc_to_target(self, doc):
        return doc["vi"].strip()

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        assert (
            num_fewshot == 0
        ), "This dataset is intended only for the zero-shot setting."
        return super().fewshot_context(
            doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
        )

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        ref_pred = (
            doc["vi"].strip(),
            list(map(str.strip, results))
        )
        print(ref_pred)

        return {
            "bleu": ref_pred,
            "chrf": ref_pred,
            # "ter": ref_pred,
        }

    def aggregation(self):
        return {
            "bleu": metrics.bleu,
            "chrf": metrics.chrf,
            # "ter": metrics.ter,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
            "chrf": True,
            # "ter": False,
        }

    def __str__(self):
        return f"{self.DATASET_PATH} en to vi Task"