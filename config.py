import argparse

import torch.nn


def load_args():
    parser = argparse.ArgumentParser()



    # Pre training
    parser.add_argument('--train_num_perclass', type=int, default=40)
    # Pre training

    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=401)
    parser.add_argument('--fine_tuned_epochs', type=int, default=141)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fine_tuned_lr', type=float, default=1e-3)

    parser.add_argument('--type', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='Trento')
    
    # Network
    parser.add_argument('--mask_ratio', default=0.70, type=float,
                        help='mask_ratio (default: 0.8)')
    parser.add_argument('--mlp_ratio', default=2.0, type=float,
                        help='mlp ratio of encoder/decoder (default: 2.0)')
    parser.add_argument('--hid_chans', default=128, type=int,
                        help='hidden channels for dimension reduction (default: 128)')     
    parser.add_argument('--hid_chans_LIDAR', default=128, type=int,
                        help='hidden channels for dimension reduction (default: 128)')
    # Augmentation
    parser.add_argument('--augment', default=True, type=bool,
                        help='either use data augmentation or not (default: False)')
    parser.add_argument('--scale', default=9, type=int,
                        help='the minimum scale for center crop (default: 19)')
    
    # MAE encoder specifics
    parser.add_argument('--encoder_dim', default=128, type=int,
                        help='feature dimension for encoder (default: 64)')
    parser.add_argument('--encoder_depth', default = 4, type=int,
                        help='encoder_depth; number of blocks ')
    parser.add_argument('--encoder_num_heads', default=8, type=int,
                        help='number of heads of encoder (default: 8)')    
    
    # MAE decoder specifics
    parser.add_argument('--decoder_dim', default= 128, type=int,
                        help='feature dimension for decoder (default: 64)')
    parser.add_argument('--decoder_depth', default = 3, type=int,
                        help='decoder_depth; number of blocks ')    
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='number of heads of decoder (default: 8)') 
  
    # options for supervised MAE    
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature for classification logits')
    parser.add_argument('--cls_loss_ratio', default=0.005, type=float,
                        help='ratio for classification loss')
    args = parser.parse_args()
    return args
