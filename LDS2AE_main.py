from scipy.io import savemat

import record
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time
import datetime
import numpy as np

from config import load_args
from data_read import readdata
from diffusion import create_diffusion
from generate_pic import generate
from hyper_dataset import HyperData
from augment import CenterResizeCrop
from util_CNN import test_batch, pre_train
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from model import mae_vit_HSIandLIDAR_patch3, vit_HSI_LIDAR_patch3, DDPM_LOSS

args = load_args()

mask_ratio = args.mask_ratio
windowsize = args.windowsize
dataset = args.dataset
type = args.type
num_epoch = args.epochs
num_fine_tuned =args.fine_tuned_epochs
lr = args.lr
train_num_per = args.train_num_perclass
num_of_ex = 100
batch_size= args.batch_size

net_name = 'LDS2AE'
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
halfsize = int((windowsize-1)/2)
val_num = 1000
Seed = 0
_, _, _, _, _,_,_, _, _, _, _,_, _,gt,s = readdata(type, dataset, windowsize,train_num_per, val_num, 0)
num_of_samples = int(s * 0.2)
nclass = np.max(gt).astype(np.int64)
print(nclass)

if args.dataset == 'Augsburg_SAR':
    in_chans_LIDAR = 4
elif args.dataset == 'Berlin':
    in_chans_LIDAR = 1
else:
    in_chans_LIDAR = 1

if args.dataset == '2013houston':
    size = 3
elif args.dataset == 'Berlin':
    size = 1
else:
    size = 1

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((num_of_ex, nclass))
af_result = np.zeros([nclass+3, num_of_ex])
criterion = nn.CrossEntropyLoss()

for num in range(0,num_of_ex):
    print('num:', num)    
    train_image, train_image_LIDAR, train_label, validation_image1, validation_image_LIDAR1, validation_label1, nTrain_perClass, nvalid_perClass, \
         train_index, val_index, index, image, image_LiDAR, gt,s = readdata(type, dataset, windowsize,train_num_per,num_of_samples,num)
    ind = np.random.choice(validation_image1.shape[0], 200, replace = False)
    validation_image = validation_image1[ind]
    validation_image_LIDAR = validation_image_LIDAR1[ind]
    validation_label= validation_label1[ind]
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    nband = train_image.shape[3]


    train_num = train_image.shape[0] 
    train_image = np.transpose(train_image,(0,3,1,2))
    train_image_LIDAR = np.transpose(train_image_LIDAR,(0,3,1,2))

    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_image1 = np.transpose(validation_image1,(0,3,1,2))
    validation_image_LIDAR = np.transpose(validation_image_LIDAR,(0,3,1,2))
    validation_image_LIDAR1 = np.transpose(validation_image_LIDAR1,(0,3,1,2))
    
    if args.augment:
        transform_train = [CenterResizeCrop(scale_begin = args.scale, windowsize = windowsize)]
        untrain_dataset = HyperData((train_image, train_image_LIDAR, train_label), transform_train)
    else:
        untrain_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
    untrain_loader = DataLoader(dataset = untrain_dataset, batch_size = batch_size, shuffle = True)

    print("=> creating model '{}'".format(net_name))


    ######################## pre-train    ########################
    diffusion = create_diffusion(timestep_respacing="1000")
    net = mae_vit_HSIandLIDAR_patch3(img_size=(windowsize,windowsize), in_chans=nband, in_chans_LIDAR=in_chans_LIDAR, hid_chans = args.hid_chans, hid_chans_LIDAR = args.hid_chans_LIDAR, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,
                              decoder_embed_dim=args.decoder_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, nb_classes=nclass, global_pool=False)

    net.cuda(1)
    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay= 1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5,T_mult=2)
   
    tic1 = time.time()
    for epoch in range(num_epoch):
        net.train()
        total_loss = 0
        for idx, (x, x_LIDAR, y) in enumerate(untrain_loader):

            x = x.cuda(1)
            x_LIDAR = x_LIDAR.cuda(1)
            y = y.cuda(1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).cuda(1)
            model_kwargs = dict(y=y)
            model_kwargs['mask_ratio'] = mask_ratio

            # Method 1 of forward process
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            noise_LIDAR = torch.randn_like(x_LIDAR)
            x_LIDAR_t = diffusion.q_sample(x_LIDAR, t, noise_LIDAR)

            # Method 2 of forward process
            # rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
            # sigma = (rnd_normal * 1.2 -1.2).exp()
            # t = sigma
            # n = torch.randn_like(x) * sigma
            # x_t = n + x
            # n_LIDAR =  torch.randn_like(x_LIDAR * sigma)
            # x_LIDAR_t  = n_LIDAR + x_LIDAR

            pred_imgs, pred_imgs_LIDAR, logits, mask, mask_LIDAR, Cross_pred_imgs, Cross_pred_imgs_LIDAR = net(x_t, x_LIDAR_t, t, **model_kwargs)

            cls_loss = criterion(logits / args.temperature, y)
            loss_mse_m, loss_mse_LIDAR_m, loss_mse_v, loss_mse_LIDAR_v = DDPM_LOSS(pred_imgs, pred_imgs_LIDAR, x, x_LIDAR, mask, mask_LIDAR, size)

            Cross_loss_mse_m, Cross_loss_mse_LIDAR_m, Cross_loss_mse_v, Cross_loss_mse_LIDAR_v = DDPM_LOSS(Cross_pred_imgs, Cross_pred_imgs_LIDAR, x, x_LIDAR, mask, mask_LIDAR, size)
            loss_cross = (Cross_loss_mse_m + Cross_loss_mse_LIDAR_m) * 0.01 + (Cross_loss_mse_v + Cross_loss_mse_LIDAR_v) * 0.1

            #removing classification  loss
            # loss =  0.1 * (loss_mse_m + loss_mse_LIDAR_m) + loss_mse_v + loss_mse_LIDAR_v

            #adding classification  loss
            loss =  0.1 * (loss_mse_m + loss_mse_LIDAR_m) + loss_mse_v + loss_mse_LIDAR_v + cls_loss  * args.cls_loss_ratio

            optimizer.zero_grad()                             
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss

        scheduler.step()
        total_loss = total_loss/(idx+1)
        state = {'model':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}       
        print('epoch:',epoch,
              'loss:',total_loss.data.cpu().numpy())

    toc1 = time.time()
    torch.save(state, './net.pt')    

    
    # ########################   finetune    # ########################
    model = vit_HSI_LIDAR_patch3(img_size=(windowsize,windowsize), in_chans=nband, in_chans_LIDAR=in_chans_LIDAR, hid_chans = args.hid_chans, hid_chans_LIDAR = args.hid_chans_LIDAR, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,num_classes = nclass, global_pool=False).cuda(1)
    checkpoint = torch.load('./net.pt')
    checkpoint_model = checkpoint['model']  
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias','head_LIDAR.weight', 'head_LIDAR.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)   
    tic2 = time.time()

    optimizer = optim.Adam(model.parameters(),lr = args.fine_tuned_lr, weight_decay= 1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [100, 131], gamma = 0.1, last_epoch=-1)
    model = pre_train(model, train_image, train_image_LIDAR, train_label,validation_image, validation_image_LIDAR, validation_label, num_fine_tuned, optimizer, scheduler, batch_size, diffusion, val = False)
    toc2 = time.time()
    model.load_state_dict(torch.load('Best_val_model/net_params.pkl'))
    state_finetune = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    true_cla, overall_accuracy, average_accuracy, kappa, true_label, test_pred, test_index, cm, pred_array = test_batch(model.eval(), image, image_LiDAR, index, 512,  nTrain_perClass,nvalid_perClass, halfsize, diffusion)
    toc3 = time.time()

    af_result[:nclass,num] = true_cla
    af_result[nclass,num] = overall_accuracy
    af_result[nclass+1,num] = average_accuracy
    af_result[nclass+2,num] = kappa

    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KAPPA.append(kappa)
    TRAINING_TIME.append(toc1 - tic1 + toc2 - tic2)
    TESTING_TIME.append(toc3 - toc2)
    ELEMENT_ACC[num, :] = true_cla
    classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, overall_accuracy, halfsize, dataset, day_str, num, net_name)
    savemat('result/'+ dataset + '/' + str(overall_accuracy)+'.mat', {'map':classification_map})
    torch.save(state_finetune, 'model/'+ dataset + '/' + str(overall_accuracy)+'net.pt')
result = np.mean(af_result, axis = 1)
print("--------" + net_name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                      'records/' + dataset + '/'+ net_name + '_' + day_str+ '_' + str(args.epochs)+ '_'  + str(args.fine_tuned_epochs) + '_train_num：' + str(train_image.shape[0]) +'_windowsize：' + str(windowsize)+'_mask_ratio_' + str(mask_ratio) + '_temperature_' + str(args.temperature) +
                      '_augment_' + str(args.augment) +'_aug_scale_' + str(args.scale) + '_loss_ratio_' + str(args.cls_loss_ratio) +'_decoder_dim_' + str(args.decoder_dim) + '_decoder_depth_' + str(args.decoder_depth)+ '.txt') 



