import math
import torch
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape, txt_length):
        super().__init__()
        self.mode = cfg.get("CLIP_BACKBONE_TYPE", "convnext").lower()
        self.txt_length = txt_length

        # ---- CNN/ConvNeXt/ResNet path ----
        if self.mode in ("convnext", "resnet"):
            model_name = cfg.get("CLIP_CNN_MODEL_NAME", "convnext_base_w_320")
            pretrained = cfg.get("CLIP_CNN_PRETRAINED", "laion_aesthetic-s13B-b82K")
            self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.text_tokenizer = open_clip.get_tokenizer(model_name)

            # Dynamically support multi-scale channel settings for different CNN models
            model_name_lower = model_name.lower()
            if "convnext_" in model_name_lower:
                self.model_type = "convnext"
                if "_base" in model_name_lower:
                    channels = [128, 128, 256, 512, 1024]
                elif "_large" in model_name_lower:
                    channels = [192, 192, 384, 768, 1536]
                elif "_xxlarge" in model_name_lower:
                    channels = [384, 384, 768, 1536, 3072]
                else:
                    raise ValueError("Unknown convnext size in model_name")
                self.feature_names = ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
                self._out_feature_strides = {
                    "stem": 2, "res2": 4, "res3": 8, "res4": 16, "res5": 32, "clip_embedding": -1
                }
                self._out_feature_channels = {
                    "stem": channels[0], "res2": channels[1], "res3": channels[2],
                    "res4": channels[3], "res5": channels[4], "clip_embedding": self.dim_latent
                }
            elif "rn" in model_name_lower:
                self.model_type = "resnet"
                if model_name_lower.replace('-quickgelu', '') in ['rn50', 'rn101']:
                    channels = [64, 256, 512, 1024, 2048]
                elif model_name_lower == 'rn50x4':
                    channels = [80, 320, 640, 1280, 2560]
                elif model_name_lower == 'rn50x16':
                    channels = [96, 384, 768, 1536, 3072]
                elif model_name_lower == 'rn50x64':
                    channels = [128, 512, 1024, 2048, 4096]
                else:
                    raise ValueError("Unknown resnet size in model_name")
                self.feature_names = ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
                self._out_feature_strides = {
                    "stem": 2, "res2": 4, "res3": 8, "res4": 16, "res5": 32, "clip_embedding": -1
                }
                self._out_feature_channels = {
                    "stem": channels[0], "res2": channels[1], "res3": channels[2],
                    "res4": channels[3], "res5": channels[4], "clip_embedding": self.dim_latent
                }
            else:
                raise ValueError("Unsupported CNN backbone type")
        
        # ---- ViT (Huggingface) path ----
        elif self.mode == "vit":
            model_name = cfg.get("CLIP_VIT_MODEL_NAME", "qihoo360/fg-clip-base")
            self.clip_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            # ViT-base, use three intermediate layers
            self.feature_names = ["vit_l11", "vit_l6", "vit_l1"]
            self._out_feature_channels = {
                "vit_l11": 768, "vit_l6": 768, "vit_l1": 768
            }
            self._out_feature_strides = {
                "vit_l11": 4, "vit_l6": 8, "vit_l1": 16
            }
        else:
            raise ValueError(f"Unknown CLIP_BACKBONE_TYPE: {self.mode}")

        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        if self.mode in ("convnext", "resnet"):
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text).to(cast_dtype)
            x = x + self.clip_model.positional_embedding.to(cast_dtype)[:x.size(1)]
            x = x.permute(1, 0, 2)
            attn_mask = self.clip_model.attn_mask[:x.size(0), :x.size(0)]
            x = self.clip_model.transformer(x, attn_mask=attn_mask)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x)
            state = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
            return x, F.normalize(state, dim=-1) if normalize else state
        elif self.mode == "vit":
            outputs = self.clip_model.text_model(
                input_ids=text,
                return_dict=True,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            pad_mask = (text != 0)
            last_index = pad_mask.sum(1) - 1
            batch_indices = torch.arange(text.shape[0], device=text.device)
            text_state = last_hidden[batch_indices, last_index, :]
            text_state = self.clip_model.text_projection(text_state)
            if normalize:
                text_state = F.normalize(text_state, dim=-1)
            return last_hidden, text_state

    def tokenize_text(self, text):
        if self.mode in ("convnext", "resnet"):
            tokens = self.text_tokenizer(text)
            if tokens.shape[1] > self.txt_length:
                tokens = tokens[:, :self.txt_length]
            else:
                padding = torch.zeros((tokens.shape[0], self.txt_length - tokens.shape[1]), dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding], dim=1)
            return tokens
        elif self.mode == "vit":
            tokens = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.txt_length)['input_ids']
            if tokens.shape[1] > self.txt_length:
                tokens = tokens[:, :self.txt_length]
            else:
                padding = torch.zeros((tokens.shape[0], self.txt_length - tokens.shape[1]), dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding], dim=1)
            return tokens.cuda()

    def extract_features(self, x):
        if self.mode == "convnext":
            out = {}
            x = self.clip_model.visual.trunk.stem(x)
            out['stem'] = x.contiguous()
            for i in range(4):
                x = self.clip_model.visual.trunk.stages[i](x)
                out[f'res{i+2}'] = x.contiguous()
            x = self.clip_model.visual.trunk.norm_pre(x)
            out['clip_vis_dense'] = x.contiguous()
            return out
        elif self.mode == "resnet":
            out = {}
            x = self.clip_model.visual.act1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
            x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
            out['stem'] = x.contiguous()
            x = self.clip_model.visual.avgpool(x)
            x = self.clip_model.visual.layer1(x)
            out['res2'] = x.contiguous()
            x = self.clip_model.visual.layer2(x)
            out['res3'] = x.contiguous()
            x = self.clip_model.visual.layer3(x)
            out['res4'] = x.contiguous()
            x = self.clip_model.visual.layer4(x)
            out['res5'] = x.contiguous()
            out['clip_vis_dense'] = x
            return out
        elif self.mode == "vit":
            # Ensure the input image is 224x224
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                vision_outputs = self.clip_model.vision_model(
                    x,
                    output_hidden_states=True,
                    return_dict=True
                )
                h_l11 = vision_outputs.hidden_states[1]
                h_l6 = vision_outputs.hidden_states[6]
                h_l1 = vision_outputs.hidden_states[-1]

                def tokens2feature(h):
                    patch_tokens = h[:, 1:, :]
                    B, N, C = patch_tokens.shape
                    feat_hw = int(N ** 0.5)
                    return patch_tokens.permute(0, 2, 1).contiguous().view(B, C, feat_hw, feat_hw)

                feat_l11 = tokens2feature(h_l11)
                feat_l6 = tokens2feature(h_l6)
                feat_l1 = tokens2feature(h_l1)
            return {
                'vit_l11': feat_l11,
                'vit_l6': feat_l6,
                'vit_l1': feat_l1,
            }

    @property
    def dim_latent(self):
        if self.mode in ("convnext", "resnet"):
            return self.clip_model.text_projection.shape[-1]
        elif self.mode == "vit":
            return self.clip_model.text_projection.shape[-1]

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self.feature_names
        }

    @property
    def size_divisibility(self):
        return -1

    def forward(self, x):
        return self.extract_features(x)
