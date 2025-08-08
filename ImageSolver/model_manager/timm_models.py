from .custom_model import CustomModel

def load_custom_model(base_model_name):
    # Basis-Modell laden
    model = CustomModel(base_model_name, pretrained=True)
    return model