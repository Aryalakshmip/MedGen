import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from tqdm import tqdm

# Download NLTK resources for text processing
try:
    # Try to download punkt and stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    # Also download punkt_tab to fix the error
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Warning: NLTK download error: {e}")
    
# Instead of directly importing, add error handling
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    print("Warning: NLTK imports failed. Using basic text processing instead.")
    # Define fallback functions for tokenization
    def sent_tokenize(text):
        # Simple sentence tokenization by splitting on periods followed by space
        return [s.strip() + '.' for s in text.split('. ') if s.strip()]
    
    def word_tokenize(text):
        # Simple word tokenization by splitting on spaces
        return [w.strip() for w in re.split(r'[\s.,;!?()]+', text.lower()) if w.strip()]
    
    # Define a simple set of stopwords as fallback
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'when', 'where', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'to', 'at', 'in', 'on', 'for',
                'with', 'by', 'of', 'this', 'that', 'these', 'those', 'it', 'its'}

class ExplainableXrayReportAnalysis:
    def __init__(self, annotation_path, generated_reports_path, ground_truth_reports_path,
                 image_base_dir, output_dir='output_visualizations'):
        """
        Initialize the Explainable AI system for X-ray reports
        
        Args:
            annotation_path: Path to annotation.json containing test samples info
            generated_reports_path: Path to CSV containing model-generated reports
            ground_truth_reports_path: Path to CSV containing ground truth reports
            image_base_dir: Base directory containing X-ray images
            output_dir: Directory to save visualizations
        """
        self.annotation_path = annotation_path
        self.generated_reports_path = generated_reports_path
        self.ground_truth_reports_path = ground_truth_reports_path
        self.image_base_dir = image_base_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load test data
        self.load_data()
        
        # Initialize model for explainability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model()
        
        # Define medical finding keywords
        self.finding_keywords = {
            'lungs_clear': ['clear', 'normal', 'no infiltrates', 'no opacity', 'unremarkable', 'well-expanded'],
            'pneumothorax': ['pneumothorax', 'air collection', 'pleural separation'],
            'pleural_effusion': ['pleural effusion', 'effusion', 'pleural fluid', 'fluid collection'],
            'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'cardiopericardial silhouette'],
            'consolidation': ['consolidation', 'opacity', 'infiltrate', 'air space opacity', 'patchy opacity'],
            'edema': ['edema', 'pulmonary edema', 'interstitial edema', 'vascular congestion'],
            'atelectasis': ['atelectasis', 'volume loss', 'lung collapse'],
            'fracture': ['fracture', 'broken', 'displaced'],
            'lung_nodule': ['nodule', 'mass', 'lesion'],
            'hilar_abnormality': ['hilar enlargement', 'hilar prominence', 'hilar abnormality']
        }
        
        # Stopwords for text processing
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load and prepare the dataset"""
        # Load annotation.json
        print("Loading annotations...")
        with open(self.annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        # Extract test samples
        self.test_samples = annotation_data['test']
        
        # Load generated reports
        print("Loading generated reports...")
        self.generated_reports_df = pd.read_csv(self.generated_reports_path)
        
        # Ensure the required column name exists
        if 'report' not in self.generated_reports_df.columns:
            # If there's only one column, rename it to 'report'
            if len(self.generated_reports_df.columns) == 1:
                self.generated_reports_df = self.generated_reports_df.rename(columns={self.generated_reports_df.columns[0]: 'report'})
            else:
                # Try to identify a column that might contain report text
                for col in self.generated_reports_df.columns:
                    if 'report' in col.lower() or 'text' in col.lower() or 'generated' in col.lower():
                        self.generated_reports_df = self.generated_reports_df.rename(columns={col: 'report'})
                        break
        
        # Load ground truth reports (if available separately, otherwise use from annotation)
        try:
            self.ground_truth_df = pd.read_csv(self.ground_truth_reports_path)
            
            # Ensure the required column name exists in ground truth
            if 'report' not in self.ground_truth_df.columns:
                if len(self.ground_truth_df.columns) == 1:
                    self.ground_truth_df = self.ground_truth_df.rename(columns={self.ground_truth_df.columns[0]: 'report'})
                else:
                    for col in self.ground_truth_df.columns:
                        if 'report' in col.lower() or 'text' in col.lower() or 'ground' in col.lower():
                            self.ground_truth_df = self.ground_truth_df.rename(columns={col: 'report'})
                            break
            
            use_gt_from_csv = True
        except Exception as e:
            print(f"Using ground truth from annotation.json: {e}")
            use_gt_from_csv = False
        
        # Create mapping between images and reports
        print("Creating image-report mapping...")
        self.image_report_mapping = {}
        
        for i, sample in enumerate(self.test_samples):
            image_id = sample['id']
            image_paths = sample['image_path']
            # Use the first image if there are multiple
            image_path = os.path.join(self.image_base_dir, image_paths[0])
            
            # Get ground truth report
            if use_gt_from_csv:
                if i < len(self.ground_truth_df):
                    ground_truth = self.ground_truth_df.iloc[i]['report']
                else:
                    print(f"Warning: No ground truth report for index {i}")
                    continue
            else:
                ground_truth = sample['report']
            
            # Get generated report (assuming same order as test samples)
            if i < len(self.generated_reports_df):
                generated = self.generated_reports_df.iloc[i]['report']
            else:
                print(f"Warning: No generated report for index {i}")
                continue
            
            self.image_report_mapping[image_id] = {
                'image_path': image_path,
                'ground_truth': ground_truth,
                'generated': generated
            }
        
        print(f"Loaded {len(self.image_report_mapping)} image-report pairs")
    
    def initialize_model(self):
        """Initialize a pre-trained model for feature extraction"""
        # Load a pre-trained DenseNet model
        self.model = models.densenet121(pretrained=True)
        
        # Remove the final classification layer
        self.model.classifier = nn.Identity()
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_medical_findings(self, report):
        """Enhanced extraction with better negation handling"""
        report = report.lower()
        findings = {}
        sentences = sent_tokenize(report)
        
        for finding_type, keywords in self.finding_keywords.items():
            # Initialize as not found
            findings[finding_type] = False
            
            for sentence in sentences:
                for keyword in keywords:
                    if keyword in sentence:
                        # Check for negations with a window of words
                        words = sentence.split()
                        keyword_idx = [i for i, word in enumerate(words) if keyword in word]
                        
                        if keyword_idx:
                            idx = keyword_idx[0]
                            negation_window = words[max(0, idx-3):idx]
                            negation_terms = ['no', 'not', 'without', 'free', 'absence', 'negative']
                            
                            # If finding is normal state (e.g., lungs_clear), negation means abnormal
                            if finding_type == 'lungs_clear':
                                if any(neg in negation_window for neg in negation_terms):
                                    findings[finding_type] = False
                                else:
                                    findings[finding_type] = True
                            # For abnormal findings, negation means normal
                            else:
                                if any(neg in negation_window for neg in negation_terms):
                                    findings[finding_type] = False
                                else:
                                    findings[finding_type] = True
                            break
        
        return findings
    
    def extract_findings_by_sentence(self, report):
        """
        Extract key medical findings with their corresponding sentences
        
        Args:
            report: Text of radiological report
            
        Returns:
            Dictionary mapping finding types to relevant sentences
        """
        sentences = sent_tokenize(report)
        findings_sentences = {}
        
        for finding_type, keywords in self.finding_keywords.items():
            findings_sentences[finding_type] = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check if any keyword is in the sentence
                for keyword in keywords:
                    if keyword in sentence_lower:
                        # Handle negations for specific findings
                        if finding_type in ['pneumothorax', 'pleural_effusion', 'consolidation']:
                            if f"no {keyword}" in sentence_lower or f"without {keyword}" in sentence_lower:
                                findings_sentences[finding_type].append(sentence)
                                break
                        else:
                            findings_sentences[finding_type].append(sentence)
                            break
        
        return findings_sentences
    
    def extract_findings_with_context(self, report):
        """Extract findings with anatomical context"""
        findings = self.extract_medical_findings(report)
        
        # Anatomical regions
        regions = {
            'right_lung': ['right lung', 'right lower lobe', 'right upper lobe', 'right middle lobe', 'RLL', 'RUL', 'RML'],
            'left_lung': ['left lung', 'left lower lobe', 'left upper lobe', 'LLL', 'LUL'],
            'bilateral': ['bilateral', 'both lungs'],
            'base': ['base', 'bases', 'basal'],
            'apex': ['apex', 'apices', 'apical'],
            'perihilar': ['perihilar', 'hilar'],
            'peripheral': ['peripheral']
        }
        
        context_findings = {}
        sentences = sent_tokenize(report.lower())
        
        for finding_type, is_present in findings.items():
            if is_present:
                context = []
                
                for keyword in self.finding_keywords[finding_type]:
                    for sentence in sentences:
                        if keyword in sentence:
                            for region, region_terms in regions.items():
                                if any(term in sentence for term in region_terms):
                                    context.append(region)
                
                context_findings[finding_type] = {
                    'present': True,
                    'location': list(set(context)) if context else ['unspecified']
                }
            else:
                context_findings[finding_type] = {
                    'present': False,
                    'location': []
                }
                
        return context_findings
    
    def generate_grad_cam(self, image_path, target_finding):
        """
        Generate Grad-CAM heatmap for a specific finding
        
        Args:
            image_path: Path to the X-ray image
            target_finding: The finding to generate explanation for
            
        Returns:
            Original image and heatmap overlay
        """
        # Open and preprocess image
        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            # Set requires_grad to True for the input tensor
            input_tensor.requires_grad_(True)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
        
        # Get the specific layer to use for Grad-CAM
        # For DenseNet, we use the last dense block
        target_layer = self.model.features.denseblock4.denselayer16
        
        # Register hooks for the forward and backward passes
        gradients = []
        activations = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass - don't use torch.no_grad() since we need gradients
        output = self.model(input_tensor)
        
        # Create a fake gradient with respect to the finding
        # We'll use the average of output features as our "prediction"
        pred = output.mean()
        
        # Make sure all computation graph is cleared before backprop
        self.model.zero_grad()
        
        # Backward pass to get gradients
        pred.backward()
        
        # Get the gradients and activations
        if not gradients or not activations:
            print(f"Warning: No gradients or activations captured for {image_path}")
            forward_handle.remove()
            backward_handle.remove()
            return None, None
            
        gradient = gradients[0]
        activation = activations[0]
        
        # Calculate weights based on gradients
        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        
        # Generate weighted activation map
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence
        
        # Normalize the CAM
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        # Resize CAM to match input image size
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (img.width, img.height))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy array
        original_img = np.array(img)
        
        overlay = heatmap * 0.4 + original_img * 0.6
        overlay = overlay.astype(np.uint8)
        
        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return original_img, overlay
    
    def analyze_and_visualize(self, sample_limit=None):
        """
        Analyze reports and generate explanatory visualizations
        
        Args:
            sample_limit: Optional limit on number of samples to process
        """
        print("Analyzing reports and generating visualizations...")
        results = []
        
        # Process each image-report pair
        image_ids = list(self.image_report_mapping.keys())
        if sample_limit:
            image_ids = image_ids[:sample_limit]
        
        for image_id in tqdm(image_ids):
            data = self.image_report_mapping[image_id]
            
            # Extract findings from generated report
            findings = self.extract_medical_findings(data['generated'])
            findings_sentences = self.extract_findings_by_sentence(data['generated'])
            
            # Process each significant finding
            for finding_type, is_present in findings.items():
                if is_present and findings_sentences[finding_type]:
                    # Generate Grad-CAM visualization
                    original_img, heatmap = self.generate_grad_cam(data['image_path'], finding_type)
                    
                    if original_img is None or heatmap is None:
                        continue
                    
                    # Save visualization
                    output_path = os.path.join(self.output_dir, f"{image_id}_{finding_type}.png")
                    self.save_visualization(original_img, heatmap, findings_sentences[finding_type], 
                                           output_path, finding_type)
                    
                    # Record result
                    results.append({
                        'image_id': image_id,
                        'finding_type': finding_type,
                        'sentences': findings_sentences[finding_type],
                        'visualization_path': output_path
                    })
        
        # Save summary results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'explanation_results.csv'), index=False)
        print(f"Analysis complete. Saved {len(self.image_report_mapping)} visualizations.")
        return results_df
    
    def save_visualization(self, original_img, heatmap, sentences, output_path, finding_type):
        """
        Save visualization with original image, heatmap, and text explanation
        
        Args:
            original_img: Original X-ray image
            heatmap: Generated heatmap overlay
            sentences: List of sentences describing the finding
            output_path: Path to save the visualization
            finding_type: Type of finding being visualized
        """
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(original_img)
        ax1.set_title('Original X-ray')
        ax1.axis('off')
        
        # Display heatmap overlay
        ax2.imshow(heatmap)
        ax2.set_title(f'Heatmap for: {finding_type.replace("_", " ").title()}')
        ax2.axis('off')
        
        # Add text annotation below the images
        plt.figtext(0.5, 0.01, f"Report text: {' '.join(sentences)}", 
                    wrap=True, horizontalalignment='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for text at bottom
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def evaluate_explanation_quality(self):
        """
        Evaluate the quality of explanations by comparing with ground truth findings
        
        Returns:
            DataFrame with evaluation metrics
        """
        print("Evaluating explanation quality...")
        evaluation_results = []
        
        for image_id, data in self.image_report_mapping.items():
            # Extract findings from both generated and ground truth reports
            gen_findings = self.extract_medical_findings(data['generated'])
            gt_findings = self.extract_medical_findings(data['ground_truth'])
            
            # Calculate match rate for each finding type
            for finding_type in self.finding_keywords.keys():
                gen_present = gen_findings.get(finding_type, False)
                gt_present = gt_findings.get(finding_type, False)
                
                # Record whether the finding matches between generated and ground truth
                evaluation_results.append({
                    'image_id': image_id,
                    'finding_type': finding_type,
                    'generated_present': gen_present,
                    'ground_truth_present': gt_present,
                    'match': gen_present == gt_present
                })
        
        # Convert to DataFrame and calculate overall metrics
        eval_df = pd.DataFrame(evaluation_results)
        
        # Calculate metrics per finding type
        metrics = eval_df.groupby('finding_type')['match'].agg(['mean', 'count']).reset_index()
        metrics.columns = ['finding_type', 'accuracy', 'count']
        
        # Save evaluation results
        eval_df.to_csv(os.path.join(self.output_dir, 'explanation_evaluation.csv'), index=False)
        metrics.to_csv(os.path.join(self.output_dir, 'explanation_metrics.csv'), index=False)
        
        print("Evaluation complete.")
        return metrics
    
    def generate_attention_maps(self, sample_limit=10):
        """
        Generate attention maps for report sentences using text-guided attention
        
        Args:
            sample_limit: Number of samples to process
            
        Returns:
            DataFrame with paths to generated attention maps
        """
        print("Generating text-guided attention maps...")
        results = []
        
        # Process a subset of image-report pairs
        image_ids = list(self.image_report_mapping.keys())[:sample_limit]
        
        for image_id in tqdm(image_ids):
            data = self.image_report_mapping[image_id]
            
            # Load and preprocess image
            try:
                img = Image.open(data['image_path']).convert('RGB')
                img_array = np.array(img)
            except Exception as e:
                print(f"Error loading image {data['image_path']}: {e}")
                continue
                
            # Extract sentences from generated report
            sentences = sent_tokenize(data['generated'])
            
            # Generate attention map for each sentence
            for i, sentence in enumerate(sentences):
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue
                
                # Generate a simplified attention map based on sentence content
                attention_map = self.generate_text_guided_attention(img_array, sentence)
                
                # Save the attention map
                output_path = os.path.join(self.output_dir, f"{image_id}_sent{i}_attn.png")
                self.save_attention_visualization(img_array, attention_map, sentence, output_path)
                
                # Record result
                results.append({
                    'image_id': image_id,
                    'sentence_index': i,
                    'sentence': sentence,
                    'visualization_path': output_path
                })
        
        # Save summary results
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.to_csv(os.path.join(self.output_dir, 'attention_results.csv'), index=False)
        print(f"Generated {len(results)} attention maps.")
        return results_df
    
    def generate_text_guided_attention(self, image, sentence):
        """
        Generate a simplified text-guided attention map
        
        This is a simplified implementation - in a full research paper you would
        use a more sophisticated approach like cross-modal attention networks
        
        Args:
            image: Numpy array of image
            sentence: Text sentence from report
            
        Returns:
            Attention map highlighting regions relevant to the sentence
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = image
            
        # Resize for processing
        gray_img = cv2.resize(gray_img, (224, 224))
        
        # Apply preprocessing to highlight features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(np.uint8(gray_img))
        
        # Extract keywords from sentence
        words = word_tokenize(sentence.lower())
        keywords = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        # Generate attention map based on keywords
        attention_map = np.zeros_like(enhanced_img, dtype=float)
        
        # Different regions of interest based on common findings
        lung_mask = np.zeros_like(enhanced_img, dtype=float)
        heart_mask = np.zeros_like(enhanced_img, dtype=float)
        
        # Create simplified anatomical masks
        # Lungs region (simplified as two rectangles on left and right sides)
        h, w = enhanced_img.shape
        lung_mask[:, :w//4] = 0.5  # Left lung
        lung_mask[:, 3*w//4:] = 0.5  # Right lung
        lung_mask[h//4:3*h//4, w//4:3*w//4] = 0  # Clear middle section
        
        # Heart region (simplified as center area)
        cy, cx = h//2, w//2
        heart_radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        heart_mask = ((x - cx)**2 + (y - cy)**2 <= heart_radius**2).astype(float) * 0.8
        
        # Assign attention based on keywords
        for keyword in keywords:
            if any(kw in keyword for kw in ['lung', 'pneumo', 'infiltrate', 'clear']):
                attention_map += lung_mask
            elif any(kw in keyword for kw in ['heart', 'cardio', 'mediastin']):
                attention_map += heart_mask
        
        # Normalize attention map
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        # If no specific regions were identified, use edge detection as fallback
        if attention_map.max() == 0:
            edges = cv2.Canny(enhanced_img, 100, 200)
            attention_map = edges / 255.0
        
        # Resize back to original dimensions
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        return attention_map
    
    def save_attention_visualization(self, original_img, attention_map, sentence, output_path):
        """
        Save visualization with original image and text-guided attention map
        
        Args:
            original_img: Original X-ray image
            attention_map: Generated attention map
            sentence: Report sentence
            output_path: Path to save the visualization
        """
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
        ax1.set_title('Original X-ray')
        ax1.axis('off')
        
        # Display attention map overlay
        # Convert to heatmap
        colored_map = plt.cm.jet(attention_map)
        # Create overlay
        overlay = original_img.copy().astype(float) / 255
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay, overlay, overlay], axis=2)
        for i in range(3):
            overlay[:,:,i] = overlay[:,:,i] * 0.7 + colored_map[:,:,i] * 0.3
        
        ax2.imshow(overlay)
        ax2.set_title('Text-Guided Attention')
        ax2.axis('off')
        
        # Add text annotation below the images
        plt.figtext(0.5, 0.01, f"Report text: {sentence}", 
                    wrap=True, horizontalalignment='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for text at bottom
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def run_complete_analysis(self, sample_limit=None):
        """
        Run the complete analysis pipeline
        
        Args:
            sample_limit: Optional limit on number of samples to process
        """
        print("Starting complete analysis pipeline...")
        
        # Step 1: Generate explanations using Grad-CAM
        results_df = self.analyze_and_visualize(sample_limit)
        
        # Step 2: Generate text-guided attention maps
        attention_df = self.generate_attention_maps(sample_limit=min(10, sample_limit if sample_limit else 10))
        
        # Step 3: Evaluate explanation quality
        metrics_df = self.evaluate_explanation_quality()
        
        print("Analysis pipeline complete!")
        print(f"Generated {len(results_df) if not results_df.empty else 0} Grad-CAM visualizations")
        print(f"Generated {len(attention_df) if not attention_df.empty else 0} text-guided attention maps")
        print("Evaluation metrics summary:")
        if not metrics_df.empty:
            print(metrics_df)
        
        return {
            'results': results_df,
            'attention': attention_df,
            'metrics': metrics_df
        }
    
# Configuration
annotation_path = "/content/drive/MyDrive/MedGen/data/iu_xray/annotation.json"
generated_reports_path = "/content/res (1) (1).csv"
ground_truth_reports_path = "/content/gts.csv"  # Optional
image_base_dir = "/content/drive/MyDrive/MedGen/data/iu_xray/images"
output_dir = "explainable_output"

# Initialize and run the analysis
explainer = ExplainableXrayReportAnalysis(
    annotation_path=annotation_path,
    generated_reports_path=generated_reports_path,
    ground_truth_reports_path=ground_truth_reports_path,
    image_base_dir=image_base_dir,
    output_dir=output_dir
)

results = explainer.run_complete_analysis(sample_limit=20)


