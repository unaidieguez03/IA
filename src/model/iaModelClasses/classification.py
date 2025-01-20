class Classifier(nn.Module):
    def __init__(self, num_classes, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)
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