import argparse
import datetime
import json
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
#from infobatch import InfoBatch
from infobatch.infobatch import InfoBatch
from torchvision import transforms
from model import *
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def safe_print(*args, **kwargs):
    if RANK in (-1, 0):
        print(*args)

def setup_ddp():
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group('nccl', rank=RANK, world_size=world_size)

def destroy_ddp():
    dist.destroy_process_group()

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}  # 确保key是str
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


plt.switch_backend('Agg')  # 关键设置：不依赖图形界面

def save_usage_plot(sample_usage, epoch, name = "Sample Usage Distribution (0-200 Epochs)", save_dir="figure/cifar100"):
    # 1. 准备数据
    usage_counts = list(sample_usage.values())

    # 2. 定义分组区间
    bins = np.arange(0, epoch, 10)

    # 3. 创建图表对象
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. 绘制柱状图
    hist, _ = np.histogram(usage_counts, bins=bins)
    bars = ax.bar(bins[:-1], hist, width=10, edgecolor='k', align='edge')

    # 5. 添加标签和样式
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}',
                   ha='center', va='bottom')
    
    # 6. 设置图表属性
    ax.set_xticks(bins)
    ax.set_xlim(0, epoch)
    ax.set_xlabel('Sample Usage Count (Grouped by 10s)')
    ax.set_ylabel('Number of Samples')
    ax.set_title(name)
    ax.grid(axis='y', linestyle='--')

    # 7. 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 8. 保存文件
    filename = os.path.join(save_dir, f"{name}.png")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)  # 显式关闭释放内存
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--use_info_batch', action='store_true',
                        help='whether use info batch or not.')
    parser.add_argument('--use_ddp', action='store_true', help='whether use ddp or not.')
    parser.add_argument('--fp16', action='store_true', help='use mix precision training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='different optimizers')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    # onecycle scheduling arguments
    parser.add_argument('--max-lr', default=0.05, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--dataset', type=str, default='cifar100',help='different dataset')
    parser.add_argument('--num_epoch', default=200,
                        type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--delta', default=0.875, type=float)
    parser.add_argument('--is_anealing', default=0, type=int)
    parser.add_argument('--model', default='r18', type=str)
    parser.add_argument('--available_GPU', type=int, default='0', help='GPU choice')
    args = parser.parse_args()
   
    available_cuda = args.available_GPU
    torch.cuda.set_device(available_cuda)
    device = torch.cuda.current_device()
    print(device)
    # if not torch.cuda.is_available():
    #     device = 'cpu'
    # elif args.use_ddp:
    #     device = 'cuda:%d' % LOCAL_RANK
    #     setup_ddp()
    # else:
    #     device = 'cuda:0'
    safe_print('==> Building model..')

    if args.model.lower() == 'r18':
        net = ResNet18(100).to(device)
    elif args.model.lower() == 'r50':
        net = ResNet50(num_classes=100).to(device)
    elif args.model.lower() == 'r101':
        net = ResNet101(num_classes=100).to(device)
    else:
        net = ResNet50(num_classes=100).to(device)
    net = net.to(device)
    # if args.use_ddp:
    #     safe_print('use ddp')
    #     net = torch.nn.parallel.DistributedDataParallel(net, [LOCAL_RANK], LOCAL_RANK)
    # else:
    #     safe_print('use normal data parallel')
    #     net = torch.nn.DataParallel(net)
    try:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction='none').to(device)
    except:
        safe_print('warning! This version has no label smooth.')
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0  # best test accuracy
    best_loss = 1e3 # best test loss
    best_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    if args.dataset.lower() == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, transform=train_transform,
                                                download=True)
        testset = torchvision.datasets.CIFAR100(
            root='./cifar100', train=False, download=True, transform=test_transform)
    elif args.dataset.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=train_transform,
                                                download=True)
        testset = torchvision.datasets.CIFAR10(
            root='./cifar10', train=False, download=True, transform=test_transform)
    # 1.Substitute dataset with InfoBatch dataset, optionally set r; delta and num_epoch are needed to anneal with full data
    if args.use_info_batch:
        safe_print('Use info batch.')
        trainset = InfoBatch(trainset, args.num_epoch, args.is_anealing, args.ratio, args.delta)
    else:
        safe_print('Use normal full batch.')

    # 2.Substitute sampler
    sampler = None
    train_shuffle = True
    if args.use_info_batch:
        sampler = trainset.sampler
        train_shuffle = False
    # if args.use_ddp and not args.use_info_batch:
    #     sampler = DistributedSampler(trainset, shuffle=True)
    #     train_shuffle = False
    safe_print(type(sampler))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0, sampler=sampler)



    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


    # Model
   

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lars':  # no tensorboardX
        from lars import Lars
        optimizer = Lars(net.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer = Lamb(net.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, steps_per_epoch=len(trainloader),
                                                    epochs=args.num_epoch, div_factor=args.div_factor,
                                                    final_div_factor=args.final_div, pct_start=args.pct_start)
    # if args.optimizer.lower() in ['sgd', 'adam']:
    #     lr_scheduler = optimizer
        
    train_acc = []
    valid_acc = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    data_for_epoch = []
    def train_info_batch(epoch):
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_sample = 0
        avg_grad_norms = []
        for batch_idx, blobs in enumerate(trainloader):
            inputs, targets = blobs
            inputs, targets = inputs.to(device), targets.to(device)
            total_sample += inputs.size(0)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(args.fp16):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                # 3. use <InfoBatch>.update(loss), all scoring/rescaling/getting mean is now conducted at the backend, see previous (research version) code for details.
                loss = trainset.update(loss)
            scaler.scale(loss).backward()


            total_grad_norm = 0.0
            num_params = 0

            for name, param in net.named_parameters():
                if param.grad is not None:
                    total_grad_norm += (param.grad.data ** 2).sum().item()  # 计算平均梯度
                    num_params += 1

            # 计算平均梯度的 2 范数
            if num_params > 0:
                avg_grad_norm = (total_grad_norm) ** 0.5  # 计算 2 范数
                avg_grad_norms.append(avg_grad_norm)            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        total_grad_norm_epoch, var_grad_norm_epoch, max_grad_norm_epoch, min_grad_norm_epoch= \
            np.mean(avg_grad_norms), np.var(avg_grad_norms), max(avg_grad_norms), min(avg_grad_norms)
        data_for_epoch.append((epoch, total_sample))
        safe_print('remain_data', total_sample, 'gradient_norm',total_grad_norm_epoch, 'var', \
                    var_grad_norm_epoch, 'max',max_grad_norm_epoch , 'min',min_grad_norm_epoch )
        safe_print('epoch:', epoch, '  Training Accuracy:', round(100. * correct /
            total, 3), '  Train loss:', round(train_loss / len(trainloader), 4))
        train_acc.append(correct / total)
        return total_grad_norm_epoch, var_grad_norm_epoch, max_grad_norm_epoch, min_grad_norm_epoch


    def train_normal(epoch):
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_sample = 0
        avg_grad_norms = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #random_size = inputs.size(0)
            total_sample += inputs.size(0)
            # # 随机选择一半的样本
            # half_size = random_size // 2
            # random_indices = np.random.choice(random_size, half_size, replace=False)

            # # 使用随机索引选择样本
            # inputs = inputs[random_indices]
            # targets = targets[random_indices]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(args.fp16):
                outputs = net(inputs)
                loss = torch.mean(criterion(outputs, targets))
            scaler.scale(loss).backward()

            total_grad_norm = 0.0
            num_params = 0

            for name, param in net.named_parameters():
                if param.grad is not None:
                    total_grad_norm += (param.grad.data ** 2).sum().item()  # 计算平均梯度
                    num_params += 1
            for name, param1 in net1.named_parameters():
                if param1.grad is not None:
                    total_grad_norm1 += (param1.grad.data ** 2).sum().item()  # 计算平均梯度
                    num_params1 += 1

            # 计算平均梯度的 2 范数
            if num_params > 0:
                avg_grad_norm = (total_grad_norm) ** 0.5  # 计算 2 范数
                avg_grad_norms.append(avg_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        total_grad_norm_epoch, var_grad_norm_epoch, max_grad_norm_epoch, min_grad_norm_epoch= \
            np.mean(avg_grad_norms), np.var(avg_grad_norms), max(avg_grad_norms), min(avg_grad_norms)
        data_for_epoch.append((epoch, total_sample))
        safe_print('remain_data', total_sample, 'gradient_norm',total_grad_norm_epoch, 'var', \
                    var_grad_norm_epoch, 'max',max_grad_norm_epoch , 'min',min_grad_norm_epoch )
        safe_print('epoch:', epoch, '  Training Accuracy:', round(100. * correct /
            total, 3), '  Train loss:', round(train_loss / len(trainloader), 4))
        train_acc.append(correct / total)
        return total_grad_norm_epoch, var_grad_norm_epoch, max_grad_norm_epoch, min_grad_norm_epoch


    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        global best_acc
        global best_loss
        global best_epoch
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = test_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        cur_acc = round(100. * correct / total, 3)
        cur_loss = round(test_loss / len(testloader), 4)
        safe_print('epoch: %d' % epoch, '  Test Acc: %.3f' % cur_acc, 
        '  Test loss: %.4f' % cur_loss, ' Best info epoch %d, acc %.3f, loss %.4f' % (best_epoch, best_acc, best_loss))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_epoch = epoch
        if cur_loss < best_loss:
            best_loss = cur_loss
        valid_acc.append(cur_acc)


    total_time = 0
    avg_grad_norms_epoch = []
    for epoch in range(args.num_epoch):
        
        # if args.use_ddp:
        #     trainloader.sampler.set_epoch(epoch)
        # 4. For epoch-based implementation, update the corresponding learning rate schedule according to steps of this epoch
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                        steps_per_epoch=len(trainloader),
                                                        epochs=args.num_epoch, div_factor=args.div_factor,
                                                        final_div_factor=args.final_div, pct_start=args.pct_start,
                                                        last_epoch=epoch * len(trainloader) - 1)
        end = time.time()
        if epoch == 200:
            break
        if args.use_info_batch:
            grad_norms_epoch, grad_var, grad_max, grad_min = train_info_batch(epoch)
        else:
            grad_norms_epoch, grad_var, grad_max, grad_min = train_normal(epoch)
        total_time += time.time() - end
        avg_grad_norms_epoch.append((grad_norms_epoch, grad_var, grad_max, grad_min))
        test(epoch)

    if args.use_info_batch:
        safe_print('Total saved sample forwarding: ', trainset.get_pruned_count())
        sample_usage = trainset.get_sample_usage()
    safe_print('Total training time: ', total_time)
    pref = 'full_batch' if not args.use_info_batch else 'info_batch'
    fn = '{}-{}-{}-epoch{}-{}-batchsize{}-pct{}-labelsm{}-IsAnealing{}-DELTA{}-{}-{}-{}_log.json'.format(
        args.model,
        str(args.max_lr),
        str(args.lr),
        args.num_epoch,
        args.ratio,
        args.batch_size,
        args.pct_start,
        args.label_smoothing,
        args.is_anealing,
        args.delta,
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), str(args.dataset),pref)
    #if LOCAL_RANK in (-1, 0):
    filepath = os.path.join("./cifar100", fn)
    #file = open(fn, 'w+')
    with open(filepath, 'w') as file:
        json.dump([total_time, trainset.get_pruned_count() if args.use_info_batch else 0,
            args.ratio, data_for_epoch, train_acc, valid_acc,avg_grad_norms_epoch,convert_numpy_types(dict(sample_usage))], file)
    
    saved_path = save_usage_plot(sample_usage,args.num_epoch,f"InfoBatch with prune ratio {trainset.get_pruned_count()/(args.num_epoch * 50000)} with ratio {args.ratio} at time { datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"sample usage path: {saved_path}")
    # if args.use_ddp:
    #     destroy_ddp()
