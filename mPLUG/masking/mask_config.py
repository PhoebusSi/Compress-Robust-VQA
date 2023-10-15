# class MaskConfigs:
#     """
#     Configs of mask training
#     """
#     zero_rate = 0.5
#     threshold = 1e-2
#     init_scale = 2e-2
#     mask_classifier = False
#     mask_biases = False
#     force_masking = 'bert'
#     controlled_init = 'magnitude_soft'  # "choices": ["magnitude", "uniform", "magnitude_and_uniform", "double_uniform", "magnitude_soft"]
#     structured_masking = None
#     structured_masking_types = None
#     name_of_masker = 'MaskedLinear1'
#     masking_scheduler_conf = 'lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1'
#     init_sparsity = None
#     final_sparsity_epoch = 1
#     masker_update_step = 100
#     train_classifier = True
#     global_prune = False
#     load_mask_from = None

class MaskConfigs:
    """
    Configs of mask training
    """
    def __init__(self) -> None:
        self.zero_rate = 0.5
        self.threshold = 1e-2
        self.init_scale = 2e-2
        self.mask_classifier = False
        self.mask_biases = False
        self.force_masking = 'bert'
        self.controlled_init = 'magnitude_soft'  # "choices": ["magnitude", "uniform", "magnitude_and_uniform", "double_uniform", "magnitude_soft"]
        self.structured_masking = None
        self.structured_masking_types = None
        self.name_of_masker = 'MaskedLinear1'
        self.masking_scheduler_conf = 'lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1'
        self.init_sparsity = None
        self.final_sparsity_epoch = 1
        self.masker_update_step = 100
        self.train_classifier = True
        self.global_prune = False
        self.load_mask_from = None