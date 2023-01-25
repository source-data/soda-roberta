from transformers.trainer_callback import ProgressCallback


class MyProgressCallback(ProgressCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    Modified to exclude non-scalars from logs. Useful if logs include a matrix/tensor for embedding or other visualizations.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            logs = {k: v for k, v in logs.items() if not isinstance(v, list)}  # modification: remove non scalar from logs before output to console
            self.training_bar.write(str(logs))
