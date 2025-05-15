import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from data.noisedata import NoiseData,NoiseDataModify
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeModel,NonLinearTypeModelModify
from utils.transform import Normalizer
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=500, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.005, type=float)
    parser.add_argument('--lr_decay', type = list, default = [100,200,300,400], help = 'learning rate decay')
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='train0318.xlsx', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--log_dir', dest='log_dir', type = str, default = 'logs/train')
    parser.add_argument('--nc', dest='nc', type = int, default = 400)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # 添加torch.backends.cudnn.benchmark以加速训练
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 添加显存清理
    torch.cuda.empty_cache()
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    transformations = Normalizer(mean=[363.80,46.21, 2457.96,149.38,67.70,7.65], std=[125.97, 199.17,941.75,5.73,6.91,0.10])

    if args.dataset == 'NoiseData':
        dataset = NoiseDataModify(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=None)

    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    
    model = NonLinearTypeModelModify(nc=args.nc).to(device)
    criterion = nn.MSELoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    milestones = args.lr_decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.01)
    
    # tensorboard visualization
    Loss_writer = SummaryWriter(log_dir = args.log_dir)

    for epoch in range(args.num_epochs):
        for i, (inputs, outputs, sheet_idx, bowl_idx) in tqdm(enumerate(train_loader)):

            inputs = inputs.to(device, non_blocking=True)  # 使用non_blocking加速数据传输
            labels = outputs.to(device, non_blocking=True)
            sheet_idx = sheet_idx.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            batch_indices = torch.arange(preds.size(0), device=device)
            preds = preds[batch_indices, sheet_idx.squeeze(), bowl_idx.squeeze()]
            # types = types.view(-1, 1)
            # preds = preds.gather(1, types)
            
            loss = criterion(preds, labels.squeeze())
            loss.backward()
            optimizer.step()

            Loss_writer.add_scalar('train_loss', loss, epoch)
            if (i+1) % 100 == 0 or (i+1) == len(dataset)//batch_size:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f'
                        %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss))
            # Save models at numbered epochs.
    
        scheduler.step()
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print('Taking snapshot...')
            if not os.path.exists('snapshots/'):
                os.makedirs('snapshots/')
            torch.save(
                model.state_dict(),
                f'snapshots/{args.output_string}_epoch_{epoch}.pth',
                _use_new_zipfile_serialization=True,
                pickle_protocol=4
            )