import torch
from torchvision import datasets, transforms, models 
from PIL import Image

class ConnectModel:
    # load model given path
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    # process frame 
    def process(self, frame):
        # frame = Image.open(imagePath).convert('RGB')
        return self.transform(frame)
    
    def predict(self, frame):
        input_tensor = self.preprocess(frame)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return prediction