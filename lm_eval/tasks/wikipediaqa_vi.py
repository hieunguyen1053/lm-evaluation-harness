from lm_eval.base import MultipleChoiceTask


class WikipediaQA_vi(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "vlsp-2023-vllm/wikipediaqa_vi"

    def __init__(self):
        super().__init__()
        import datasets
        self.dataset = datasets.load_from_disk("vlsp-2023-vllm/wikipediaqa_vi")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
