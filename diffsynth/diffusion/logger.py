import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


class EMAModelLogger:
    """
    A ModelLogger that supports the EMA model.
    """
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, use_ema=True):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.use_ema = use_ema


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            os.makedirs(self.output_path, exist_ok=True)
            # trainable part
            trainable_dict = unwrapped_model.export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            trainable_dict = self.state_dict_converter(trainable_dict)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(trainable_dict, path, safe_serialization=True)
            # EMA part - directly access ema_model to avoid ZeRO shard / precision issues
            if self.use_ema and hasattr(unwrapped_model, "ema_model"):
                ema_state = unwrapped_model.ema_model.averaged_model.state_dict()
                ema_dict = {}
                for k, v in ema_state.items():
                    clean_k = k
                    if self.remove_prefix_in_ckpt and clean_k.startswith(self.remove_prefix_in_ckpt):
                        clean_k = clean_k[len(self.remove_prefix_in_ckpt):]
                    ema_dict[clean_k] = v

                if ema_dict:
                    ema_file_name = file_name.replace(".safetensors", "_ema.safetensors")
                    ema_path = os.path.join(self.output_path, ema_file_name)
                    accelerator.save(ema_dict, ema_path, safe_serialization=True)
