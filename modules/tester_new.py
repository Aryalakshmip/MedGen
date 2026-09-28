import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader # Add this import if it's missing in the original

# Assuming generate_heatmap is in the same directory or a known path
try:
    from modules.utils import generate_heatmap
except ImportError:
    print("Warning: modules.utils not found. Plotting functionality may be limited.")
    generate_heatmap = None  # Or define a dummy function

from tqdm import tqdm


class BaseTester(object):
    def __init__(self, model, criterion, args):
        self.args = args

        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if hasattr(args, 'load') and args.load:  # Check if args has 'load' attribute and if it's not None
            self._load_checkpoint(args.load)
        else:
            self.logger.warning("No checkpoint loaded. Starting from scratch.")

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine, testing will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPUs configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu
                )
            )
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path, map_location=self.device)  # Map to correct device
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info("Checkpoint loaded successfully.")



class Tester(BaseTester):
    def __init__(self, model, criterion, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        test_gts, test_res = [], []

        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader), desc="Testing"):  # Added desc
                images, reports_ids, reports_masks = (
                    images.to(self.device),
                    reports_ids.to(self.device),
                    reports_masks.to(self.device),
                )
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # Save results and ground truths to CSV files
            test_res_df = pd.DataFrame(test_res)
            test_gts_df = pd.DataFrame(test_gts)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)
        self.logger.info('Testing finished.')

    def plot(self):
        if not (hasattr(self.args, 'batch_size') and self.args.batch_size == 1 and hasattr(self.args, 'beam_size') and self.args.beam_size == 1):
            self.logger.warning("Plotting requires batch_size=1 and beam_size=1. Skipping plotting.")
            return

        if generate_heatmap is None:
             self.logger.warning("generate_heatmap function not available. Plotting functionality will be limited.")
             return

        self.logger.info('Start to plot attention weights in the test set.')
        attentions_dir = os.path.join(self.save_dir, "attentions")
        os.makedirs(attentions_dir, exist_ok=True)

        mean = torch.tensor((0.485, 0.456, 0.406)).to(self.device)  # Move to device
        std = torch.tensor((0.229, 0.224, 0.225)).to(self.device)   # Move to device
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader), desc="Plotting"): # Added desc
                images, reports_ids, reports_masks = (
                    images.to(self.device),
                    reports_ids.to(self.device),
                    reports_masks.to(self.device),
                )
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean * 255), 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()

                attention_weights = []
                if hasattr(self.model.encoder_decoder.model.decoder, 'layers'):  # Check if layers exist
                     for layer in self.model.encoder_decoder.model.decoder.layers:
                        if hasattr(layer, 'src_attn') and hasattr(layer.src_attn, 'attn'): #Check for attention
                            attention_weights.append(layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0))
                        else:
                            self.logger.warning(f"Attention weights not found in layer {layer}. Skipping.")
                            attention_weights = None # Or use continue to next layer
                            break #stop if at least one is invalid

                if attention_weights is not None:
                    for layer_idx, attns in enumerate(attention_weights):
                        if len(attns) != len(report):
                            self.logger.warning(f"Layer {layer_idx}: Number of attention weights ({len(attns)}) does not match number of words in report ({len(report)}). Skipping layer.")
                            continue

                        layer_dir = os.path.join(attentions_dir, "{:04d}".format(batch_idx), "layer_{}".format(layer_idx))
                        os.makedirs(layer_dir, exist_ok=True)

                        for word_idx, (attn, word) in enumerate(zip(attns, report)):
                            heatmap = generate_heatmap(image, attn)
                            if heatmap is not None:
                                word_filename = os.path.join(layer_dir, "{:04d}_{}.png".format(word_idx, word))
                                cv2.imwrite(word_filename, heatmap)
                            else:
                                self.logger.warning(f"Could not generate heatmap for word '{word}'. Skipping.")
                else:
                   self.logger.warning("Attention weights not found. Skipping plotting.")
        self.logger.info('Plotting finished.')
