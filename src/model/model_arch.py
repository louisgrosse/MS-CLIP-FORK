# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from src.model.factory import *
from transformers.modeling_outputs import BaseModelOutputWithPooling
import re


class ImageEncoder(nn.Module):
    def __init__(
            self, image_encoder) -> None:
        super().__init__()

        self.model = image_encoder

    def forward(self, images):
        if not isinstance(images, torch.Tensor):

            if len(images["pixel_values"].shape) == 5:
                images["pixel_values"] = images["pixel_values"].squeeze(dim=1)
                features = self.model.encode_image(images["pixel_values"])

        else:
            if len(images.shape) == 5:
                images = images.squeeze(dim=1)
            features = self.model.encode_image(images)

        if isinstance(features, BaseModelOutputWithPooling):
            features = features.pooler_output
        return features


class TextEncoder(nn.Module):
    def __init__(self, text_encoder) -> None:
        super().__init__()

        self.model = text_encoder

    def forward(self, input_ids):
        if not isinstance(input_ids, torch.Tensor):
            if len(input_ids["input_ids"].shape) == 3:
                input_ids["input_ids"] = input_ids["input_ids"].squeeze(dim=1)
                input_ids["attention_mask"] = input_ids["attention_mask"].squeeze(dim=1)
            output = self.model.encode_text(input_ids["input_ids"])
        else:
            if len(input_ids.shape) == 3:
                input_ids = input_ids.squeeze(dim=1)
            output = self.model.encode_text(input_ids)
        return output


class BaseModel(nn.Module):
    def __init__(self,
                 channels: int = 12,
                 base_model_str="ViT-B-16",
                 ckpt: str = "laion2b-s34b-b88K",
                 clone_weights: bool = True,
                 ):
        super().__init__()
        self.base_model_str = base_model_str
        self.ckpt = ckpt

        if "32" in base_model_str:
            self.stride = 32
        if "14" in base_model_str:
            self.stride = 14
        if "16" in base_model_str:
            self.stride = 16

        self.model, self.tokenizer = load_model(base_model=base_model_str, ckpt_path=ckpt, channels=channels,
                                                clone_weights=clone_weights)

        # Move model to device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = 1.0 / temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = self.logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            base_model_str="ViT-B-16",
            ckpt: str = "laion2b_s34b_b88K",
            channels: int = 10,
            warm_up=None,
            max_iter=None,
            initial_temperature: float = 0.07,
            weight_decay: float = 0.05,  #  penalizes larger weights
            head_lr: float = 3.76e-5,  #  projection lr
            clone_weights: bool = True,
            pacl_weight: float = 0.2,
            interval=None,
            frequency=None,
            trainable_modules: list = [],
            full_trainable: bool = False,
            restart: bool = False,
            t_0: int = None,
            t_multi: int = None,
            mean_init: bool = False,
            use_gc: bool = True,
            decay_factor: float = 0.1,
            patch_alignment: bool = False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_base_model = BaseModel(channels=channels, base_model_str=base_model_str, ckpt=ckpt,
                                         clone_weights=clone_weights)
        self.channels = channels
        self.tokenizer = self.clip_base_model.tokenizer
        self.warm_up = warm_up
        self.max_iter = max_iter
        self.pacl_weight = pacl_weight
        self.interval = interval
        self.frequency = frequency
        self.restart = restart
        self.t_0 = t_0
        self.t_multi = t_multi
        self.head_lr = head_lr
        self.automatic_optimization = (not use_gc)  #  (not self.params.use_gc) # needed when use_gc is on
        self.fp16 = use_gc,  #  (self.params.precision == 16)
        self.use_gc = use_gc
        self.decay_factor = decay_factor

        if clone_weights:
            trainable_modules.append("visual.conv1.weight")

        # freeze whole model before choosing which modules to unfreeze
        if not full_trainable:
            for param in self.clip_base_model.model.parameters():
                param.requires_grad = False

            # select module to unfreeze
            if trainable_modules is not None:
                for name, param in self.clip_base_model.model.named_parameters():
                    for module in trainable_modules:
                        if module == name:
                            param.requires_grad = True
                # attention
                if "attn" in trainable_modules:
                    pattern = re.compile(r"^visual\.transformer\.resblocks\.\d+\.attn")
                    for name, param in self.clip_base_model.model.named_parameters():
                        if pattern.search(name):
                            param.requires_grad = True

                if "visual" in trainable_modules:
                    # Updated pattern to match any module starting with "visual."
                    pattern = re.compile(r"^visual\..*")
                    for name, param in self.clip_base_model.model.named_parameters():
                        if pattern.search(name):
                            param.requires_grad = True
        else:
            for param in self.clip_base_model.model.parameters():
                param.requires_grad = True

        self.text_encoder = TextEncoder(
            text_encoder=self.clip_base_model.model,
        )

        self.image_encoder = ImageEncoder(
            image_encoder=self.clip_base_model.model,
        )

        self.temperature = nn.Parameter(
            torch.ones([]) * initial_temperature)  # nn.Parameter(torch.tensor(initial_temperature))
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    # abstract method
    def _compute_losses(self, image_embeddings_patch, image_embeddings_cls, text_embeddings):
        if self.patch_alignment:
            loss_fn = ClipLoss(self.temperature)
            loss_patch = loss_fn(image_embeddings_patch, text_embeddings)
            loss_cls = loss_fn(image_embeddings_cls, text_embeddings)
            return self.pacl_weight * loss_patch + (1 - self.pacl_weight) * loss_cls
        else:
            loss_fn = ClipLoss(self.temperature)
            loss_cls = loss_fn(image_embeddings_cls, text_embeddings)
            return loss_cls

    def compute_accuracy(self, image_embeddings, text_embeddings):

        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        predicted_texts_indices = logits.argmax(dim=-1)
        predicted_images_indices = logits.T.argmax(dim=-1)
        # Create ground truth indices
        ground_truth_indices = torch.arange(logits.shape[0], device=logits.device)

        # Compute the number of correct predictions
        correct_texts = (predicted_texts_indices == ground_truth_indices).float().mean().item()
        correct_images = (predicted_images_indices == ground_truth_indices).float().mean().item()
        # Calculate average accuracy
        accuracy = (correct_texts + correct_images) / 2.0

        return accuracy

    def accuracy(self, image_embeddings, text_embeddings, topk=(1,)):

        output = (text_embeddings @ image_embeddings.T) / self.temperature
        target = torch.arange(output.shape[0], device=output.device)

        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        pred_T = output.T.topk(max(topk), 1, True, True)[1].t()
        correct_T = pred_T.eq(target.view(1, -1).expand_as(pred_T))

        n = len(target)

        return [(correct[:k].reshape(-1).float().sum(0, keepdim=True) / n).item() for k in topk][0], \
        [(correct_T[:k].reshape(-1).float().sum(0, keepdim=True) / n).item() for k in topk][0]

    def inference_text(self, inputs):

        text_features = self.text_encoder(
            inputs
        )

        return F.normalize(text_features, dim=-1)

    def inference_vision(self, image):

        images = self.image_encoder(image)

        if isinstance(images, tuple):
            image_features = images[0]
            return F.normalize(image_features, dim=-1)


        else:
            return F.normalize(images, dim=-1)

    def forward(self, inputs):

        image_features = self.image_encoder(inputs[0])

        text_features = self.text_encoder(
            inputs[1]
        )

        if isinstance(image_features, tuple):
            image_embeddings = self.image_projection(image_features[1])  #224 224 b32 50 1 49
            text_embeddings = self.text_projection(text_features)

        else:
            image_embeddings = image_features
            text_embeddings = text_features

        return F.normalize(image_embeddings, dim=-1), F.normalize(text_embeddings, dim=-1)

    def configure_optimizers(self):
        parameters = [
            {"params": [param for name, param in self.clip_base_model.model.named_parameters() if
                        "visual.conv1.weight" in name], "lr": self.head_lr},
            {"params": [param for name, param in self.clip_base_model.model.named_parameters() if
                        "visual.conv1.weight" not in name], "lr": self.head_lr * self.decay_factor},

        ]

        optimizer = optim.SGD(parameters)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=self.t_0,
                                                                      T_mult=self.t_multi, )

        return {
            "optimizer": optimizer,
            'lr_scheduler': {
                'name': 'train/lr',  # put lr inside train group in tensorboard
                'scheduler': lr_scheduler,
                'interval': self.interval,
                'frequency': self.frequency,
            }
        }

    def validation_step(self, batch, *args, **kwargs):

        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings_patch=None, image_embeddings_cls=image_embeddings,
                                    text_embeddings=text_embeddings)
        acc = self.compute_accuracy(image_embeddings, text_embeddings)
        accu_t2i, accu_i2t = self.accuracy(image_embeddings, text_embeddings)
        self.val_loss = self.all_gather(loss).mean()
        self.val_acc = self.all_gather(acc).mean()
        self.val_accuracy_t2i = self.all_gather(accu_t2i).mean()
        self.val_accuracy_i2t = self.all_gather(accu_i2t).mean()
        self.log(name="val_loss", value=self.val_loss, prog_bar=True, sync_dist=True)
        self.log(name="val_acc_custom", value=self.val_acc, prog_bar=True, sync_dist=True)
        self.log(name="val_acc_i2t", value=self.val_accuracy_i2t, prog_bar=True, sync_dist=True)
        self.log(name="val_acc_t2i", value=self.val_accuracy_t2i, prog_bar=True, sync_dist=True)
        return loss

    #abstract method
    def processors(self):
        img_processor, tokenizer = self.clip_base_model.configure_processors()
        return img_processor, tokenizer
