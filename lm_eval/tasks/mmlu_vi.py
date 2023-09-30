from .hendrycks_test import GeneralHendrycksTest

class MMLU_vi(GeneralHendrycksTest):
    DATASET_PATH = "vlsp-2023-vllm/mmlu"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(subject="")