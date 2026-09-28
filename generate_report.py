import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel

def load_image(image_path):
    """Load and preprocess the image"""
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def generate_report(args, model, image, tokenizer, device):
    """Generate medical report from image"""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image, mode='sample')
        reports = tokenizer.decode_batch(output['predictions'].cpu().numpy())
    return reports[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to input image')
    parser.add_argument('--model_path', type=str, default='results/iu_xray/model_best.pth', 
                       help='path to trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model configuration
    model_args = {
        'visual_extractor': 'resnet101',
        'visual_extractor_pretrained': True,
        'd_model': 512,
        'd_ff': 512,
        'd_vf': 2048,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'logit_layers': 1,
        'bos_idx': 0,
        'eos_idx': 0,
        'pad_idx': 0,
        'use_bn': 0,
        'dataset_name': 'iu_xray',
    }
    args.__dict__.update(model_args)

    # Create tokenizer
    tokenizer = Tokenizer(args)

    # Load model
    device = torch.device(args.device)
    model = R2GenModel(args, tokenizer)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Load and process image
    image = load_image(args.image_path)
    
    # Generate report
    report = generate_report(args, model, image, tokenizer, device)
    
    print("\n=== Medical Report Generation ===")
    print(f"Input Image: {args.image_path}")
    print("\nGenerated Report:")
    print(report)

if __name__ == '__main__':
    main()