import os
import time
import torch
import pandas as pd
from numpy import inf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import matplotlib.pyplot as plt

class BaseTrainer(object):
    def __init__(self, model, criterion, optimizer, args):
        self.args = args
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)
            # Print logged information to the screen
            for key, value in log.items():
                print(f"{key:15s}: {value}")
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: No GPU available. Training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(f"Warning: Configured {n_gpu_use} GPUs but only {n_gpu} are available.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print(f"Saving checkpoint: {filename}")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, tokenizer):
        super(Trainer, self).__init__(model, criterion, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer  # Tokenizer is needed for converting IDs to text
        self.rouge = Rouge()  # Initialize the Rouge metric
        # Lists to store metrics for visualization.
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_bleus = []
        self.val_rouges_l = []

    def _train_epoch(self, epoch):
        train_loss = 0
        all_predictions = []
        all_targets = []
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            # Collect predictions and targets for accuracy calculation
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(reports_ids.cpu().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {
            'train_loss': train_loss / len(self.train_dataloader),
            'train_accuracy': accuracy_score(all_targets, all_predictions),
            'train_f1': f1_score(all_targets, all_predictions, average='weighted'),
        }
        # Validation phase
        val_metrics = self._validate_epoch()
        log.update(val_metrics)
        # Store metrics
        self.train_losses.append(log['train_loss'])
        self.train_accuracies.append(log['train_accuracy'])
        self.val_losses.append(log['val_loss'])
        self.val_accuracies.append(log['val_accuracy'])
        self.val_bleus.append(log['val_bleu'])
        self.val_rouges_l.append(log['val_rouge_l'])
        self.lr_scheduler.step()
        return log

    def _validate_epoch(self):
        val_loss = 0
        all_references = []
        all_candidates = []
        all_val_predictions = []
        all_val_targets = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, reports_ids, mode='sample')  # Generate reports
                loss = self.criterion(output, reports_ids, reports_masks)
                val_loss += loss.item()
                # Convert generated and ground truth IDs to text
                generated_reports = self.tokenizer.decode_batch(output.cpu().numpy())
                target_reports = self.tokenizer.decode_batch(reports_ids.cpu().numpy())
                all_candidates.extend(generated_reports)
                all_references.extend(target_reports)
                # Collect predictions and targets for accuracy calculation during validation
                _, predicted_val = torch.max(output.data, 1)
                all_val_predictions.extend(predicted_val.cpu().numpy())
                all_val_targets.extend(reports_ids.cpu().numpy())
        # Calculate metrics
        val_loss /= len(self.val_dataloader)
        # Calculate BLEU scores
        bleu_scores = [sentence_bleu([ref.split()], cand.split()) for ref, cand in zip(all_references, all_candidates)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        # Calculate ROUGE scores
        try:
            scores = self.rouge.get_scores(all_candidates, all_references, avg=True)
            rouge_l_fscore = scores['rouge-l']['f']
        except ValueError:
            print("Rouge failed to compute scores. Check for empty sequences.")
            rouge_l_fscore = 0
        metrics = {
            'val_loss': val_loss,
            'val_accuracy': accuracy_score(all_val_targets, all_val_predictions),
            'val_f1': f1_score(all_val_targets, all_val_predictions, average='weighted'),
            'val_bleu': avg_bleu,
            'val_rouge_l': rouge_l_fscore,
        }
        return metrics

    def visualize_metrics(self):
        # Visualization of the results.
        plt.figure(figsize=(15, 10))
        # Plot Losses
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        # Plot Accuracies
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        # Plot BLEU Scores
        plt.subplot(2, 2, 3)
        plt.plot(self.val_bleus, label='Validation BLEU Score', color='orange')
        plt.title('Validation BLEU Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('BLEU Score')
        plt.legend()
        plt.grid()
        # Plot ROUGE-L Scores
        plt.subplot(2, 2, 4)
        plt.plot(self.val_rouges_l, label='Validation ROUGE-L Score', color='green')
        plt.title('Validation ROUGE-L Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('ROUGE-L Score')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
