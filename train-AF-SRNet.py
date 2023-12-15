import argparse
import torch
import os
from models.AF_SRNet import AFSRNet
from trainers.regression import train_meteonet_regression
from loader.meteonet import MeteonetDataset
from loader.filesets import bouget21
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from datetime import datetime
from os.path import join
from tqdm import tqdm

# 设置命令行参数
parser = argparse.ArgumentParser(prog='train-AF-SRNet', description='Training AF-SRNet for Meteonet Nowcasting')

parser.add_argument('-dd', '--data-dir', type=str, help='Directory containing data', dest='data_dir', default='data')
parser.add_argument('-wd', '--wind-dir', type=str, help='Directory containing the wind data', dest='wind_dir', default=None)
parser.add_argument('-t', '--thresholds', type=float, nargs='+', help='Rainmap thresholds in mm/h', dest='thresholds', default=[0.1,1,2.5])
parser.add_argument('-Rd', '--run-dir', type=str, help='Directory to save logs and checkpoints', dest='run_dir', default="runs")
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', dest='epochs', default=20)
parser.add_argument('-b', '--batch-size', type=int, help='Batch size', dest='batch_size', default=128)
parser.add_argument('-lr', '--learning-rate', type=str, nargs='+', help='LR_WD format is: epoch:Learning rate, weight decay', dest='lr_wd', default=['0:8e-4,1e-5', '4:1e-4,5e-5'])
parser.add_argument('-nw', '--num-workers', type=int, help='Numbers of workers for Cuda', dest='num_workers', default=8)
parser.add_argument('-o', '--oversampling', type=float, help='Oversampling percentage of last class', dest='oversampling', default=0.9)
parser.add_argument('-ss', '--snapshot-step', type=int, help='Snapshot step', dest='snapshot_step', default=5)

args = parser.parse_args()

# 设置训练参数
input_len = 12
time_horizon = 6
stride = input_len
feature_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分离训练集和验证集
train_files, val_files, _ = bouget21(join(args.data_dir, 'rainmaps'))

# 数据集
indexes = [join(args.data_dir,'train.npz'), join(args.data_dir,'val.npz')]
train_ds = MeteonetDataset(train_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[0], tqdm=tqdm)
val_ds = MeteonetDataset(val_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[1], tqdm=tqdm)
val_ds.norm_factors = train_ds.norm_factors

# 采样器
train_sampler = meteonet_random_oversampler(train_ds, args.thresholds[-1], args.oversampling)
val_sampler = meteonet_sequential_sampler(val_ds)

# 数据加载器
train_loader = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

# 其余设置和数据加载代码...

# 解析学习率和权重衰减
lr_wd = {int(k.split(':')[0]): [float(v) for v in k.split(':')[1].split(',')] for k in args.lr_wd}
current_lr, current_wd = lr_wd[0]  # 假设初始学习率和权重衰减为第一个设置的值

# 模型初始化
model = AFSRNet(feature_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_wd)
criterion = torch.nn.MSELoss()

# 训练过程
rundir = join(args.run_dir, f'{datetime.now()}')
os.makedirs(rundir, exist_ok=True)
print(f'Run files will be recorded in directory {rundir}')

# 使用 train_meteonet_regression 进行训练
scores = train_meteonet_regression(
    train_loader, val_loader, model, args.thresholds, args.epochs, lr_wd,
    snapshot_step=args.snapshot_step, rundir=rundir, device=device
)

# 保存训练信息
torch.save({'hyperparams': vars(args), 'scores': scores}, join(rundir, 'run_info.pt'))
