import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerDiceTopK10Loss


class nnUNetTrainerDiceTopK10Loss_5000epochs(nnUNetTrainerDiceTopK10Loss):
    """Dice+TopK10 trained for 5000 epochs instead of 1000.

    BUGFIX (2026-06-21): the previous version set `num_epochs = 5000` as a CLASS
    attribute. nnUNetTrainer.__init__ assigns `self.num_epochs = 1000` as an
    INSTANCE attribute (nnUNetTrainer.py:158), which shadows the class attribute —
    so training silently ran 1000 epochs. R19 and R37 both ran 1000ep, NOT 5000ep,
    despite the trainer name. Fixed by overriding in __init__ AFTER super().

    The __init__ mirrors the parent's exact signature (no *args/**kwargs) so
    nnUNetTrainer's `my_init_kwargs` introspection still works."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000
