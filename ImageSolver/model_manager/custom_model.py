import os

import timm
import torch

class CustomModel():
    def __init__(self, base_model_name, pretrained=True):
        self.model = timm.create_model(base_model_name, pretrained=pretrained)
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def load_weights_api(self, name):
        self.model.load_state_dict(torch.load(os.path.join('api','models', name+'.pth'), weights_only=True))

    def load_weights(self, finetuned_model_name):
        # Load the weights of the finetuned model to self.model
        # finetuned_self.base_model_name refers to the name of the .pth file (only the name, not the whole path, without .pth)
        self.model.load_state_dict(torch.load(os.path.join('model_manager','finetuned_models', finetuned_model_name+'.pth'), weights_only=True))

    def save_model(self, custom_name):
        model_save_path = os.path.join('model_manager', 'finetuned_models')
        # torch.save(model, os.path.join(model_save_path, custom_name+'.pth'))
        torch.save(self.model.state_dict(), os.path.join(model_save_path, custom_name+'.pth'))

    def generate_embedding(self, image_tensor):
        features = self.model.forward_features(image_tensor)
        # if self.base_model_name == 'convit_base' or self.base_model_name == 'convit_small' or self.base_model_name == 'convit_tiny':
        #     embedding = torch.mean(features, dim=1).squeeze(0)
        # else:
        #     embedding = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling
        embedding = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling
        return embedding

    def freeze_layer(self):
        if self.base_model_name == 'efficientnet_b0' or self.base_model_name == 'efficientnet_b1' or self.base_model_name == 'efficientnet_b2':
            for param in self.model.parameters():
                param.requires_grad = False

            # Erste Convolution (Stem) auftauen
            for param in self.model.conv_stem.parameters():
                param.requires_grad = True

            # BatchNorm auch auftauen (optional)
            # for param in self.model.bn1.parameters():
            #     param.requires_grad = True

            for param in self.model.blocks[-2:].parameters():  # Nur letzte 2 Blöcke trainieren
                param.requires_grad = True

            for param in self.model.classifier.parameters():  # Head trainieren
                param.requires_grad = True
        
        elif self.base_model_name == 'convnext_base' or self.base_model_name == 'convnext_large' or self.base_model_name == 'convnext_small':
            # Nur die letzten Schichten trainierbar machen
            for param in self.model.parameters():
                param.requires_grad = False  # Erst alles einfrieren

            for param in self.model.stem.parameters():  # Optional: Frühere Schichten auftauen
                param.requires_grad = True

            for param in self.model.stages[-1:].parameters():
                param.requires_grad = True

            for param in self.model.head.parameters():  # Head trainieren
                param.requires_grad = True
        
        elif self.base_model_name == 'convformer_s36' or self.base_model_name == 'convformer_m36' or self.base_model_name == 'convformer_b36':
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.stem.parameters():  # Optional: Frühere Schichten auftauen
                param.requires_grad = True

            # for param in self.model.stages[-1:].parameters():
            #     param.requires_grad = True

            for param in self.model.head.parameters():  # Falls 'head' nicht existiert, versuche 'classifier'
                param.requires_grad = True
        
        elif self.base_model_name == 'convit_base' or self.base_model_name == 'convit_small' or self.base_model_name == 'convit_tiny':
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.patch_embed.parameters():  # Optional: Frühere Schichten auftauen
                param.requires_grad = True

            for param in self.model.blocks[-1:].parameters():  # Nur letzte 2 Blöcke trainieren
                param.requires_grad = True

            for param in self.model.head.parameters():  # Falls 'head' nicht existiert, versuche 'classifier'
                param.requires_grad = True

        elif self.base_model_name == 'resnet50':
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.conv1.parameters():  # Optional: Frühere Schichten auftauen
                param.requires_grad = True

            for param in self.model.bn1.parameters():
                param.requires_grad = True

            # for param in self.model.layer4.parameters():
            #     param.requires_grad = True

            for param in self.model.fc.parameters():  # Falls 'head' nicht existiert, versuche 'classifier'
                param.requires_grad = True
        
        else:
            print('##### ERROR: model not found #####')
            return