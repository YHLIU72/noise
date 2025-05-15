import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData, NoiseDataFFT,NoiseDataFFTModify
from utils.transform import Normalizer
from model.nonlinear import NonLinearTypeBinModelModify
import torch
from torch.autograd import Variable
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='test0318.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--nc', dest='nc', type = int, default = 1600)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[363.80,46.21, 2457.96,149.38,67.70,7.65], std=[125.97, 199.17,941.75,5.73,6.91,0.10])

    if args.dataset == 'NoiseData':
        dataset = NoiseDataFFTModify(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=None, fft_out=26)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearTypeBinModelModify(nc = args.nc, bowl_idx=2, num_bins=26, num_sheets=4)
    saved_state_dict = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(saved_state_dict)
    model.eval()

    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    
    criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss(reduction='sum')
    test_cos_error = .0
    test_mse_error = .0
    total = 0
    for i, (inputs, outputs, sheet_idx,bowl_idx) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = Variable(inputs)
        labels = Variable(outputs)
        preds = model(inputs)
        batch_indices = torch.arange(preds.size(0))
        preds = preds[batch_indices, sheet_idx.squeeze(),bowl_idx.squeeze(), :]

        # calculate loss
        loss_flag = torch.ones(outputs.size(0))
        cos_loss = cos_criterion(preds, labels, loss_flag)
        mse_loss = criterion(preds, labels)
        test_cos_error += torch.sum(cos_loss)
        test_mse_error += torch.sum(mse_loss)
      #   print(inputs, types, sheet_idx)
      #   print(preds, labels, test_cos_error, torch.sum(test_mse_error))
    
    print('Test error on the ' + str(total) +' test samples. cos: %.4f mse: %.4f' % (test_cos_error / total, test_mse_error * batch_size/ total))