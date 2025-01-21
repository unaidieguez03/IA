from iaModelClasses import classification, encoder
import torch
from pathlib import Path
from processImg import grayscaler, normalize_image, resize_image
class ClassificationModel:
    def __init__(self,save_path: str):
        self.model=self.load_model(Path(save_path))

    def load_model(path:Path):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Running on the GPU')
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print('Running on the CPU')
        assert Path.exists(path), "No checkpoint found"
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        model = classification.Classifier(num_classes=2, encoder=encoder.Encoder())
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)  # Move model to the correct device
        model.eval()  # Set the model to evaluation mode
        return model
    def get_prediction(output_tensor: torch.Tensor) -> tuple[str, float]:
        """
        Convert model output tensor to medical classification prediction and confidence score.
        
        Args:
            output_tensor (torch.Tensor): Model output tensor of shape (1, 2)
            
        Returns:
            tuple[str, float]: Predicted condition and its probability
        """
        # Class mapping
        class_labels = {
            0: 'Osteophytes',
            1: 'No finding'
        }
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output_tensor, dim=1)
        
        # Get the predicted class (index of highest probability)
        predicted_class = torch.argmax(output_tensor, dim=1).item()
        
        # Get the confidence score (probability of predicted class)
        confidence = probabilities[0][predicted_class].item()
        
        # Get the actual condition name
        prediction = class_labels[predicted_class]
        
        return prediction, confidence
    def classify(self, image) -> torch.Tensor:
        image = resize_image(image, 512)
        image = grayscaler(image)
        image = normalize_image(image)
        image_tensor = torch.from_numpy(b)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        prediction = self.model(image_tensor)
        probabilities = torch.softmax(prediction, dim=1)
        predicted_class,_ = self.get_prediction(probabilities)

        return predicted_class