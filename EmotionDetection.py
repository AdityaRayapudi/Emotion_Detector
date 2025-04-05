import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extraction
        self.feature_extractor = nn.Sequential(
            # 1 input image channel, 6 output channels, 5x5 square convolution
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # ANN 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16 * 9 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        

        logits = self.linear_relu_stack(x)
        return logits
    
class LabelMap():
    labels_map = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Suprised",
    }

    def __init__(self):
        pass

    def getLabel(label_num):
        return LabelMap.labels_map[label_num]