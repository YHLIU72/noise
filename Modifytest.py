import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData,NoiseDataModify
from utils.transform import Normalizer
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeModel,NonLinearTypeModelModify
import torch
from torch.autograd import Variable
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='test0318.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='./snapshots/nc1600_epoch_499.pth', type=str)
    parser.add_argument('--nc', dest='nc', type = int, default = 1600)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[362.69,60.67, 2372.96,149.45,67.89,7.65], std=[130.04, 209.28,930.67,5.79,6.88,0.10])

    if args.dataset == 'NoiseData':
        dataset = NoiseDataModify(dir=args.data_dir, filename='test0318.xlsx', transform=transformations, use_type=None)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearTypeModelModify(nc=args.nc).to(device)
    saved_state_dict = torch.load(snapshot_path, map_location=device, weights_only=False)   
    model.load_state_dict(saved_state_dict)
    
    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    criterion = nn.MSELoss().to(device)
    test_error = .0
    total = 0

    for i, (inputs, outputs, sheet_idx, bowl_idx) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = inputs.to(device)
        labels = outputs.to(device)
        sheet_idx = sheet_idx.to(device)
        
        preds = model(inputs)
        
        batch_indices = torch.arange(preds.size(0), device=device)
        preds = preds[batch_indices, sheet_idx.squeeze(), bowl_idx.squeeze()]
       
        test_loss = criterion(preds, labels.squeeze())
        test_error += torch.sum(test_loss)
        # print(preds, labels, test_loss, torch.sum(test_loss))
    
    print('Test error on the ' + str(total) +' test samples. MSE: %.4f' % (test_error * batch_size/ total))