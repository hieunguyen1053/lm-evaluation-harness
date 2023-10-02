from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class Arithmetic_vi(Task):
    VERSION = 1
    DATASET_PATH = "vlsp-2023-vllm/arithmetic_vi"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return NotImplemented

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return self.dataset["test"].filter(lambda x: x['meta'] == self.DATASET_NAME)

    def doc_to_text(self, doc):
        return "Question: " + doc["context"] + " Answer: "

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        return doc["completion"]

    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx, doc["completion"])
        return is_prediction

    def process_results(self, doc, results):
        (is_prediction,) = results
        return {"acc": is_prediction}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}


class Arithmetic2DPlus_vi(Arithmetic_vi):
    DATASET_NAME = "two_digit_addition"


class Arithmetic2DMinus_vi(Arithmetic_vi):
    DATASET_NAME = "two_digit_subtraction"


class Arithmetic3DPlus_vi(Arithmetic_vi):
    DATASET_NAME = "three_digit_addition"


class Arithmetic3DMinus_vi(Arithmetic_vi):
    DATASET_NAME = "three_digit_subtraction"


class Arithmetic4DPlus_vi(Arithmetic_vi):
    DATASET_NAME = "four_digit_addition"


class Arithmetic4DMinus_vi(Arithmetic_vi):
    DATASET_NAME = "four_digit_subtraction"


class Arithmetic5DPlus_vi(Arithmetic_vi):
    DATASET_NAME = "five_digit_addition"


class Arithmetic5DMinus_vi(Arithmetic_vi):
    DATASET_NAME = "five_digit_subtraction"


class Arithmetic2DMultiplication_vi(Arithmetic_vi):
    DATASET_NAME = "two_digit_multiplication"


class Arithmetic1DComposite_vi(Arithmetic_vi):
    DATASET_NAME = "sum_of_digits"
