from lm_eval.base import MultipleChoiceTask


class Exams_vi(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "vlsp-2023-vllm/exams_vi"

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
        return self.dataset["test"].map(self._process_doc).filter(lambda x: x['metadata']['subject'] == self.SUBJECT_NAME)

    def _process_doc(self, doc):
        out_doc = {
            "id": doc["id"],
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


class Exams_lichsu_vi(Exams_vi):
    SUBJECT_NAME = "MÔN SỬ"


class Exams_dialy_vi(Exams_vi):
    SUBJECT_NAME = "MÔN ĐỊA"


class Exams_vatly_vi(Exams_vi):
    SUBJECT_NAME = "MÔN LÍ"


class Exams_sinhhoc_vi(Exams_vi):
    SUBJECT_NAME = "MÔN SINH"


class Exams_hoahoc_vi(Exams_vi):
    SUBJECT_NAME = "MÔN HOÁ"


class Exams_toan_vi(Exams_vi):
    SUBJECT_NAME = "MÔN TOÁN"


class Exams_van_vi(Exams_vi):
    SUBJECT_NAME = "MÔN VĂN"