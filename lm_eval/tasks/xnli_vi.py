from .xnli import XNLIBase


class XNLI_vi(XNLIBase):  # Vietnamese
    DATASET_PATH = "xnli"
    DATASET_NAME = "vi"

    QUESTION_WORD = "đúng không"
    ENTAILMENT_LABEL = "Vâng"
    NEUTRAL_LABEL = "Vì vậy"
    CONTRADICTION_LABEL = "Không"