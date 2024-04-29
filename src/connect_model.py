import torch
from torchvision import datasets, transforms, models 
from PIL import Image
# from cnn_model import CNN

asl_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


class ConnectModel:
    # load model given path
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    # process frame 
    def process(self, frame):
        # frame = Image.open(imagePath).convert('RGB')
        return self.transform(frame)
    
    def predict(self, frame):
        input_tensor = self.process(frame)
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(prediction, dim=1)
            # _, predicted_index = torch.max(probabilities, 1)
            # predicted_label = asl_classes[predicted_index.item()]
            top_probs, top_idxs = torch.topk(probabilities, 3)
            top_probs = top_probs.numpy().flatten() *100
            top_idxs = top_idxs.numpy().flatten()
            top_classes = [asl_classes[idx] for idx in top_idxs]
        return list(zip(top_classes, top_probs))

        return predicted_label

# model = ConnectModel(r'output\new_model.pth')