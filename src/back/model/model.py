import torch
from pathlib import Path
from .processImg import grayscaler, normalize_image, resize_image
class ClassificationModel:
    def __init__(self,save_path: str):
        self.model=self.load_model(Path(save_path))

    def load_model(self, path:Path):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Running on the GPU')
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print('Running on the CPU')
        assert Path.exists(path), "No checkpoint found"
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        model = Classifier(num_classes=2, encoder=Encoder())
        model.load_state_dict(checkpoint["model_state"]) 
        model.eval()  
        return model
    def get_prediction(output_tensor: torch.Tensor) -> tuple[str, float]:

        class_labels = {
            0: 'Osteophytes',
            1: 'No finding'
        }
        
        probabilities = torch.softmax(output_tensor, dim=1)
        
        predicted_class = torch.argmax(output_tensor, dim=1).item()
        
        confidence = probabilities[0][predicted_class].item()
        
        prediction = class_labels[predicted_class]
        
        return prediction, confidence
    def classify(self, image):
        image = resize_image(image, 512)
        image = grayscaler(image)
        image = normalize_image(image)
        image_tensor = torch.from_numpy(b)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        prediction = self.model(image_tensor)
        probabilities = torch.softmax(prediction, dim=1)
        predicted_class,_ = self.get_prediction(probabilities)

        return 
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels:int=1, base_filters:int=64):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters

        self.pool = nn.MaxPool2d((2, 2))

        self.level1 = ConvolutionBlock(self.in_channels, self.base_filters,dropout_rate=0.2, kernel_size=3, activation=nn.ReLU())
        self.level2 = ConvolutionBlock(self.base_filters, self.base_filters * 2,dropout_rate=0.2, kernel_size=3, activation=nn.ReLU())
        self.level3 = ConvolutionBlock(self.base_filters * 2, self.base_filters * 4,dropout_rate=0.2, kernel_size=3, activation=nn.ReLU())
        
        self.bottleneck = ConvolutionBlock(self.base_filters * 4, self.base_filters * 8,dropout_rate=0.2, kernel_size=3, activation=nn.ReLU())
    def forward(self, x):

        skip_connections = []

        level1_features = self.level1(x)
        skip_connections.append(level1_features)
        level1_pooled = self.pool(level1_features)

        level2_features = self.level2(level1_pooled)
        skip_connections.append(level2_features)
        level2_pooled = self.pool(level2_features)

        level3_features = self.level3(level2_pooled)
        skip_connections.append(level3_features)
        level3_pooled = self.pool(level3_features)

        bottleneck_features = self.bottleneck(level3_pooled)
        return {
            'encoded_image': bottleneck_features,
            'skip_connections': skip_connections
        }
class Classifier(nn.Module):
    def __init__(self, num_classes, encoder):
        super().__init__()
        self.encoder = encoder
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define classifier with explicit num_classes output
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Will flatten 512x1x1 to 512
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # This should now correctly output num_classes
        )
        
    def forward(self, x):
        encoder_output = self.encoder(x)
        encoded_features = encoder_output['encoded_image']
        # Add debug prints to verify shapes
        print(f"encoded_features shape: {encoded_features.shape}")
        pooled_features = self.global_pool(encoded_features)
        print(f"pooled_features shape: {pooled_features.shape}")
        logits = self.classifier(pooled_features)
        print(f"logits shape: {logits.shape}")
        return logits
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate:float, kernel_size:int, activation):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Dropout2d(dropout_rate)
        )
    def forward(self, x):
        return self.block(x)