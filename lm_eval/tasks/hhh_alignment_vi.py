from lm_eval.base import MultipleChoiceTask

class HHH_alignment_vi(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "vlsp-2023-vllm/hhh_alignment"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["input"],
            "choices": [target for target in doc["targets"]["choices"]],
            "gold": doc["targets"]["labels"].index(1),
        }
        return out_doc