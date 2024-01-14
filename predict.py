import csv
import os
import sys

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import CellDataset
from pgcnet import PGCNet
from utils import load_hdf5, predict_calculate_val_mae
from tqdm import tqdm

arguments = sys.argv[1]

with open(arguments, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PGCNet(in_channels=3, out_channels=1, init_features=32, height=224,
               up_sampling='GroupConv', down_sampling='DeepConv', bil=[[0,1], [0,0], [0,0]], caff={'name':'ICA'}, caff_enable=[0,0,1],
               channel_attention={'name':'ICA'}, channel_attention_enable=[0,0,0,0,0,0], gradient_initial=[0.4,0.8,0.4], gradm_stategylist=None)

saved_model_path = config['model_path']
saved_state_dict = torch.load(saved_model_path)
model.load_state_dict(saved_state_dict)

model.to(device)
model.eval()

data = load_hdf5(config['file_dir'], keys=['counts', 'imgs'])

dataset = CellDataset(data['imgs'], data['counts'], transform=None)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

output_folder = config['save_dir']
if not os.path.isdir(output_folder):
    os.makedirs(output_folder, exist_ok=True)

csv_filename = 'predictions.csv'
csv_path = os.path.join(output_folder, csv_filename)

with open(csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Index', 'Predicted_Count', 'Groundtruth_Count', 'MAE'])

    for idx, (inputs, counts) in enumerate(tqdm(test_loader, desc="Predicting")):
        inputs, counts = inputs.to(device), counts.to(device)
        with torch.no_grad():
            predict = model(inputs)/config['scale_num']
            simple_mae, gt_count, pre_count = predict_calculate_val_mae(predict, counts)
            predict.squeeze(dim=0)

        csv_writer.writerow([idx, pre_count, gt_count, simple_mae])

        image_folder = os.path.join(output_folder, str(idx))
        os.makedirs(image_folder, exist_ok=True)

        predicted_image_path = os.path.join(image_folder, 'predicted_image.png')
        predict_np = np.transpose(predict[0].detach().cpu().numpy(), (1, 2, 0))[:, :, 0]
        plt.imsave(predicted_image_path, predict_np, cmap='viridis')

        original_image_path = os.path.join(image_folder, 'original_image.png')
        inputs_np = np.transpose(inputs[0].detach().cpu().numpy(), (1, 2, 0))
        plt.imsave(original_image_path, inputs_np)

        groundtruth_image_path = os.path.join(image_folder, 'groundtruth_image.png')
        counts_np = np.transpose(counts[0].detach().cpu().numpy(), (1, 2, 0))[:, :, 0]
        plt.imsave(groundtruth_image_path, counts_np, cmap='viridis')

print(f"Predictions saved to: {csv_path}")
