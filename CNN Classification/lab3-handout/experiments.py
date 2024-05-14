from torch import optim, nn, compile
from torchvision.models import EfficientNet_V2_L_Weights, ResNet101_Weights,efficientnet_v2_l, resnet101
from infrastructure import Experiment
from model import YourFirstNet, YourSecondNet

SELECTMODEL = 1
class YourFirstCNN(Experiment):
    
    def init_model(self, n_labels, **kwargs):
        if SELECTMODEL == 1:
            model = YourFirstNet(n_labels=5)
        if SELECTMODEL == 2:
            model = YourSecondNet(n_labels=5)
        if SELECTMODEL == 3:
            model = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Linear(in_features, 5)
        if SELECTMODEL == 4:
            model = resnet101(ResNet101_Weights.IMAGENET1K_V2)
            in_features = model.fc.in_features
            model.classifier = nn.Linear(in_features, 5)
        
        print(model)
        self.ckpt.model = compile(model)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters(), lr=0.001)
