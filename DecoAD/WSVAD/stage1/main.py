import torch.optim as optim
import torch
import os
import random
import numpy as np
from WSVAD.stage1.dataset import gen_fusion_dataset_dataloader
from WSVAD.stage1.fusion import Model
# from stage1.fusion_nopath import Model
from WSVAD.stage1.train import train
from WSVAD.stage1.test import test
from WSVAD.stage1.args import init_parser
from datetime import datetime

# 初始化解析器
# from WSVAD.stage1.train_baocun import shuffle_and_cycle

parser = init_parser()

# 解析参数
args = parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 并行gpu


setup_seed(int(42))  # 1577677170  2023

def main(epochs = 0,auc_2=0,flag2=0,lr=0.001, weight_decay=0.0005,train_nloader = None, train_aloader=None, test_loader = None):
    if epochs == 0:
        epochs = args.max_epoch
    if lr == 0:
        lr = args.lr
    if weight_decay == 0:
        weight_decay = args.weight_decay
    max_auc = 0.
    device = torch.device('cuda:0' if 'cuda' in args.device and torch.cuda.is_available() else 'cpu')
    if train_nloader == None or train_aloader == None or test_loader == None:
        _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()

    model = Model().to(device)

    name = f'{args.supervise}|{args.dataset}'

    if auc_2 != 0:
        checkpoints = f'{args.WSVAD}stage{flag2}/ckpt/{name}/model'+'{:.5f}.pkl'.format(auc_2)
        # checkpoints = '/home/liuxinyu/PycharmProjects/VAD/DecoVAD/exp/Weakly_UB_78.4_82.4.pkl'  # ROC-AUC: 0.7426, AP: 0.2033, sigma:8
        # checkpoints = '/home/liuxinyu/PycharmProjects/VAD/DecoVAD/exp/Weakly_UB_78.4_82.4.pkl'  # ROC-AUC: 0.7426, AP: 0.2033, sigma:8
        checkpoint = torch.load(checkpoints, map_location=device)

        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = True

    train_nloader = train_nloader
    train_aloader = train_aloader

    if not os.path.exists(f'{args.WSVAD}stage1/ckpt'):
        os.makedirs(f'{args.WSVAD}stage1/ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y-%m-%d|%H:%M')

    # 创建文件夹
    folder_name = current_time

    if not os.path.exists(f'{args.WSVAD}stage1/ckpt/data'):
        os.makedirs(f'{args.WSVAD}stage1/ckpt/data')

    os.makedirs(f'{args.WSVAD}stage1/ckpt/data/{args.supervise}|{args.dataset}|{folder_name}', exist_ok=True)


    if not os.path.exists(f'{args.WSVAD}stage1/ckpt/{name}'):
        os.makedirs(f'{args.WSVAD}stage1/ckpt/{name}')
    folder_name = f'{args.supervise}|{args.dataset}|{folder_name}'
    print(f"文件夹已创建: {folder_name}")

    for epoch in range(epochs):
        avg_loss = train(train_nloader, train_aloader, model, args.batch_size, optimizer, device)
        scheduler.step()
        
        # Evaluate and save checkpoint only every 5 epochs (or at the last epoch) to speed up training
        if epoch % 5 == 0 or epoch == epochs - 1:
            roc, pr = test(test_loader, model, device)
            torch.save(model.state_dict(),
                       f'{args.WSVAD}stage1/ckpt/data/{folder_name}/'+'{:.5f}'.format(roc)+'_'+'{:.5f}.pkl'.format(pr))
            auc = roc
            if auc > max_auc:
                torch.save(model.state_dict(), f'{args.WSVAD}stage1/ckpt/{name}/' + 'model' + '{:.5f}.pkl'.format(auc))
                max_auc = auc
            print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc, max_auc))
        else:
            print('Epoch {0}/{1}: loss:{2:.6f}\n'.format(epoch, epochs, avg_loss))

    if max_auc>auc_2:
        flag = 1
    else:
        max_auc = auc_2
        flag = flag2
    return round(max_auc, 5),flag


if __name__ == '__main__':
    main()


