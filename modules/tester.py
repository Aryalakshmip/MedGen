import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch

from modules.utils import generate_heatmap
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

        self._load_checkpoint(args.load)

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
                "Warning: There's no GPU available on this machine, training will be performed on CPU."
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
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, args)
        self.test_dataloader = test_dataloader
        self.device = args.device

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        test_gts, test_res = [], []

        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
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

    def plot(self):
        try:
            assert self.args.batch_size == 1 and self.args.beam_size == 1
            self.logger.info('Start to plot attention weights in the test set.')
            os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
            
            for i, batch in enumerate(self.test_dataloader):
                images = batch['images']
                reports = batch['reports']
                outputs = self.model(images, mode='sample')
                
                # Generate attention heatmaps
                for j, (image, attn) in enumerate(zip(images[0], outputs['attention_weights'][0])):
                    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    attn = attn.cpu().numpy()
                    
                    try:
                        heatmap = generate_heatmap(image, attn)
                        save_path = os.path.join(self.save_dir, "attentions", f"sample_{i}_view_{j}.png")
                        cv2.imwrite(save_path, heatmap)
                    except Exception as e:
                        self.logger.error(f"Error generating heatmap for sample {i}, view {j}: {str(e)}")
                        continue
                
                if i % 100 == 0:
                    self.logger.info(f'Processed {i} samples')
                    
        except Exception as e:
            self.logger.error(f"Error in plotting: {str(e)}")
            raise
