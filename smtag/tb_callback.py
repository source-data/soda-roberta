import os, re
import numpy as np
import torch
from transformers.integrations import (
    TensorBoardCallback
)
from transformers.utils import logging
logger = logging.get_logger(__name__)


def rewrite_logs(d):
    """
    group into 'losses' eval/loss and train/loss into with the loss breakdown provided as supp_data
    """

    # example on training setp
    # {'loss': 21.4436, 'learning_rate': 4.9970059880239524e-05, 'epoch': 0.06}
    # example on evaluation step
    # {'eval_loss': 19.32028579711914, 'eval_supp_data_loss_diag': 0.23595364391803741, 'eval_supp_data_loss_off_diag': 0.9614391922950745, 'eval_supp_data_loss_twin_z': 1.1973928213119507, 
    # 'eval_supp_data_loss_z_1': 0.0, 'eval_supp_data_loss_z_2': 0.0, 'eval_supp_data_loss_lm_1': 8.891294479370117, 'eval_supp_data_loss_lm_2': 9.229777336120605, 'eval_runtime': 25.7975, 
    # 'eval_samples_per_second': 102.762, 'eval_steps_per_second': 2.171, 'epoch': 0.06}

    new_d = {}
    for k, v in d.items():
        if k == "loss":
            new_d["losses/main_loss"] = {"train": v}
        elif k == "eval_loss":
            new_d["losses/main_loss"] = {"eval": v}
        else:
            supp_data = re.search(r"^(.*)_supp_data_(.*)", k)  # could use nge lookahead (?!img)
            img = re.search(r"^(.*)_img_(.*)", k)  # for example as in  "eval_supp_data_img_correl"
            if supp_data is not None and img is None:
                main_tag = f"losses/{supp_data.group(1)}_supp"
                scalar_tag = supp_data.group(2)
                if main_tag not in new_d:
                    new_d[main_tag] = {}
                new_d[main_tag][scalar_tag] = v
            elif img is not None:
                main_tag = f"images/{img.group(1)}_{img.group(2)}"
                if main_tag not in new_d:
                    new_d[main_tag] = {}
                new_d[main_tag] = v
            else:
                main_tag = f"other_data/{k}"
                if main_tag not in new_d:
                    new_d[main_tag] = {}
                new_d[main_tag][k] = v
    return new_d


class MyTensorBoardCallback(TensorBoardCallback):

    """Display log and metrics. Modified to plot losses together and to plot supp_data items passed in the model output. Also looks for logs elements with  _img_ in keys to display as images."""

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
            for main_tag, val in logs.items():
                if main_tag.startswith("images"):
                    val = torch.tensor(val) # re-tensorify
                    if val.dim() < 3:
                        val = val.unsqueeze(0)
                    val = val - val.min()
                    val = val / val.max()
                    val = 1.0 - val  # invert: high correl black, low correl white
                    val = torch.cat([val] * 3, 0)  # format C x H x W
                    self.tb_writer.add_image("images", val, state.global_step)
                else:
                    # assume a scalar
                    self.tb_writer.add_scalars(main_tag, val, state.global_step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
