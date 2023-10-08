from .hellaswag import HellaSwag

class HellaSwag_vi(HellaSwag):
    VERSION = 0
    DATASET_PATH = "vlsp-2023-vllm/hellaswag"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False