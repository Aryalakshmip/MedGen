import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel
import os
import warnings
warnings.filterwarnings('ignore')

class Args:
    def __init__(self):
        # Model configuration
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        self.d_model = 512
        self.d_ff = 512
        self.d_vf = 2048
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        
        # Memory settings
        self.rm_num_slots = 3
        self.rm_num_heads = 8
        self.rm_d_model = 512
        
        # Generation settings
        self.max_seq_length = 60
        self.beam_size = 3
        self.sample_method = 'beam_search'
        self.sample_n = 1
        self.temperature = 1.0  # Add this line
        self.group_size = 1  # Add this line
        self.output_logsoftmax = 1  # Add this line
        self.decoding_constraint = 0  # Add this line
        self.block_trigrams = 1  # Add this line
        
        # Tokenizer settings
        self.ann_path = 'data/iu_xray/annotation.json'
        self.threshold = 3
        self.dataset_name = 'iu_xray'
        
        # Other settings
        self.bos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.use_bn = 0
        self.drop_prob_lm = 0.5

def load_image(image_path):
    """Load and preprocess X-ray image"""
    # Load image
    image = Image.open(image_path)
    
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        gray_img = image.convert('L')
        image = Image.merge('RGB', (gray_img, gray_img, gray_img))
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Debug image dimensions
    print(f"Image mode: {image.mode}")
    print(f"Image size: {image.size}")
    
    # Apply transformations and reshape for model input
    image_tensor = transform(image)
    print(f"Initial tensor shape: {image_tensor.shape}")  # Should be [3, 224, 224]
    
    # Create a tensor with shape [1, 2, 3, 224, 224] for IU X-ray format
    # Duplicate the image for both views since we only have one image
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
    image_tensor = image_tensor.repeat(1, 2, 1, 1, 1)  # Repeat for 2 views: [1, 2, 3, 224, 224]
    
    print(f"Final tensor shape: {image_tensor.shape}")  # Should be [1, 2, 3, 224, 224]
    
    return image_tensor

def generate_report(image_path, model, tokenizer, device):
    """Generate medical report from X-ray image"""
    try:
        model.eval()
        with torch.no_grad():
            # Load and process image
            image = load_image(image_path).to(device)
            print("Image loaded and processed successfully")
            print(f"Input tensor shape to model: {image.shape}")
            
            # Generate report using beam search
            output = model(image, mode='sample')  # Changed from 'beam_search' to 'sample'
            if output is None or 'predictions' not in output:
                print("Model output is invalid")
                return None
                
            predictions = output['predictions']
            print(f"Raw predictions shape: {predictions.shape}")
            
            # Decode predictions
            report = tokenizer.decode_batch(predictions.cpu().numpy())[0]
            print("Report generated successfully")
            
            return report
            
    except Exception as e:
        print(f"Error in generate_report: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def visualize_results(image_path, report):
    """Display X-ray image and generated report"""
    plt.figure(figsize=(12, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
    plt.axis('off')
    plt.title('Input X-ray Image')
    
    # Show report
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.5, report, wrap=True, fontsize=10)
    plt.axis('off')
    plt.title('Generated Medical Report')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Initialize settings
        args = Args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Print all args for debugging
        print("\nModel Configuration:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        
        # Initialize model and tokenizer
        tokenizer = Tokenizer(args)
        model = R2GenModel(args, tokenizer)
        
        # Load trained model
        model_path = "results/iu_xray/model_best.pth"
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Model loaded successfully")
            
            # Add this debugging code after loading the checkpoint
            print("\nCheckpoint keys:")
            print(checkpoint.keys())
            print("\nModel state dict keys:")
            print(model.state_dict().keys())
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
            
        model = model.to(device)
        model.eval()
        
        # Generate report for a sample image
        image_path = "data/iu_xray/images/CXR1000_IM-0003/0.png"
        
        # Verify paths
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return
        if not os.path.exists(image_path):
            print(f"Image not found at: {image_path}")
            return
            
        # Generate report
        print("\nGenerating report...")
        report = generate_report(image_path, model, tokenizer, device)
        
        if report:
            print("\nGenerated Medical Report:")
            print("-" * 50)
            print(report)
            print("-" * 50)
            
            visualize_results(image_path, report)
        else:
            print("Failed to generate report")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()