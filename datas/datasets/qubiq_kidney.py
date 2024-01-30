from .qubiq import QUBIQ


class QUBIQKidney(QUBIQ):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 qubiq_label="weighted",
                 qubiq_task=0,
                 fold=0,
                 num_cases=24,
                 num_raters=3,
                 in_channels=1,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 window_low=0,
                 window_high=255,
                 crop_size=[256, 256],
                 ignore_index=255,
                 reduce_zero_label=False,
                 image_prefix=None,
                 image_suffix=None,
                 label_prefix=None,
                 label_suffix=None,
                 **kwargs):
        super().__init__(train=train,
                         data_dir=data_dir,
                         qubiq_dataset="qubiq/kidney",
                         qubiq_label=qubiq_label,
                         qubiq_task=qubiq_task,
                         fold=fold,
                         num_cases=num_cases,
                         num_raters=num_raters,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         scale_min=scale_min,
                         scale_max=scale_max,
                         window_low=window_low,
                         window_high=window_high,
                         crop_size=crop_size,
                         ignore_index=ignore_index,
                         reduce_zero_label=reduce_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()
