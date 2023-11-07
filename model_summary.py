import argparse
import torch
import torchvision.models as models
from torchsummary import summary

def fetch_pretrained_model(model_name):
    try:
        # Attempt to dynamically retrieve a pretrained model from torchvision.models
        model = getattr(models, model_name)(pretrained=True)
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name}")

    return model

def main():
    parser = argparse.ArgumentParser(description='Print pretrained model summary')
    parser.add_argument('modelname', type=str)
    args = parser.parse_args()
    model_name = args.modelname

    # Fetching the pretrained model using the specified model name
    model = fetch_pretrained_model(model_name)

    print(f"-------------- {model_name} Model Architecture Summary --------------")
    print(model)

    input_shape = (3, 224, 224)  # Input shape (number of channels, height, width)
    print("Model Summary:")
    print(summary(model, input_shape))

if __name__ == "__main__":
    # Execute the main function when the script is run
    main()
