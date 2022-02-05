import os, re
from turtle import pd
from transformers.integrations import (
    TensorBoardCallback,
    rewrite_logs
)
from transformers.utils import logging
logger = logging.get_logger(__name__)


def rewrite_logs(d):
    """
    group into 'losses' eval/loss and train/loss into with the loss breakdown provided as supp_data
    """
    new_d = {}
    for k, v in d.items():
        if k == "loss":
            new_d["losses/main_loss"] = {"train": v}
        elif k == "eval_loss":
            new_d["losses/main_loss"] = {"eval": v}
        else:
            m = re.search(r"^(.*)_supp_data_(.*)", k)
            if m is not None:
                main_tag = f"losses/{m.group(1)}_supp"
                scalar_tag = m.group(2)
                if main_tag not in new_d:
                    new_d[main_tag] = {}
                new_d[main_tag][scalar_tag] = v
            else:
                main_tag = f"other_data/{k}"
                if main_tag not in new_d:
                    new_d[main_tag] = {}
                new_d[main_tag][k] = v
    return new_d


class MyTensorBoardCallback(TensorBoardCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for main_tag, scalar_dict in logs.items():
                self.tb_writer.add_scalars(main_tag, scalar_dict, state.global_step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
