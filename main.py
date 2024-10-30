import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import argparse
import re

from helpers import makedir
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

import logging
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-m', nargs=1, type=float, default=None)
parser.add_argument('-last_layer_fixed', nargs=1, type=str, default=None)
parser.add_argument('-subtractive_margin', nargs=1, type=str, default=None)
parser.add_argument('-using_deform', nargs=1, type=str, default=None)
parser.add_argument('-topk_k', nargs=1, type=int, default=None)
parser.add_argument('-deformable_conv_hidden_channels', nargs=1, type=int, default=None)
parser.add_argument('-num_prototypes', nargs=1, type=int, default=1200)
parser.add_argument('-dilation', nargs=1, type=float, default=2)
parser.add_argument('-incorrect_class_connection', nargs=1, type=float, default=0)
parser.add_argument('-rand_seed', nargs=1, type=int, default=None)
parser.add_argument('-model_name', nargs=1, type=str, default='')
parser.add_argument('-num_offset_head', nargs=1, type=int, default=1)
parser.add_argument('-dropout', nargs=1, type=float, default=0.)
parser.add_argument('-weight_decay', nargs=1, type=float, default=0.)
parser.add_argument('-self_mask', nargs=1, type=str, default=None)
parser.add_argument('-clip_offset', nargs=1, type=str, default=None)
parser.add_argument('-l1', nargs=1, type=str, default=None)

wandb.init(project='CV Project', config=parser.parse_args())

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
m = args.m[0]
rand_seed = args.rand_seed[0]
last_layer_fixed = args.last_layer_fixed[0] == 'True'
subtractive_margin = args.subtractive_margin[0] == 'True'
using_deform = args.using_deform[0] == 'True'
topk_k = args.topk_k[0]
deformable_conv_hidden_channels = args.deformable_conv_hidden_channels[0]
num_prototypes = args.num_prototypes[0]
model_name = args.model_name[0]
num_offset_head = args.num_offset_head[0]
dropout = args.dropout[0]
weight_decay = args.weight_decay[0]
self_mask = args.self_mask[0] == 'True'
clip_offset = args.clip_offset[0] == 'True'
l1 = args.l1[0] == 'True'

dilation = args.dilation
incorrect_class_connection = args.incorrect_class_connection[0]

# print("---- USING DEFORMATION: ", using_deform)
# print("Margin set to: ", m)
# print("last_layer_fixed set to: {}".format(last_layer_fixed))
# print("subtractive_margin set to: {}".format(subtractive_margin))
# print("topk_k set to: {}".format(topk_k))
# print("num_prototypes set to: {}".format(num_prototypes))
# print("incorrect_class_connection: {}".format(incorrect_class_connection))
# print("deformable_conv_hidden_channels: {}".format(deformable_conv_hidden_channels))

model_dir = './saved_models/' + model_name + '/'
# log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

# log("Model Name, %s", model_name)
# log("Uising deformation: %s", using_deform)
# log("Margin set to: %s", m)
# log("last_layer_fixed set to: %s", last_layer_fixed)
# log("subtractive_margin set to: %s", subtractive_margin)
# log("topk_k set to: %s", topk_k)
# log("num_prototypes set to: %s", num_prototypes)
# log("incorrect_class_connection: %s", incorrect_class_connection)
# log("deformable_conv_hidden_channels: %s", deformable_conv_hidden_channels)

print("Training information has been stored in log.")

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
# print("Random seed: ", rand_seed)
# log("Random seed: ", rand_seed)
    
print(os.environ['CUDA_VISIBLE_DEVICES'])

from settings import img_size, experiment_run, base_architecture

if 'resnet34' in base_architecture:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet152' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet50' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet121' in base_architecture:
    prototype_shape = (num_prototypes, 1024, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet161' in base_architecture:
    prototype_shape = (num_prototypes, 2208, 2, 2)
    add_on_layers_type = 'upsample'
else:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
# print("Add on layers type: ", add_on_layers_type)
# log("Add on layers type: ", add_on_layers_type)

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

from settings import train_dir, test_dir, train_push_dir

# model_dir = './saved_models/' + base_architecture + '/' + train_dir + '/' + experiment_run + '/'
# model_dir = './saved_models/' + model_name + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

log("Model Name, %s"% model_name)
log("Uising deformation: %s"% using_deform)
log("Margin set to: %s"% m)
log("last_layer_fixed set to: %s"% last_layer_fixed)
log("subtractive_margin set to: %s"% subtractive_margin)
log("topk_k set to: %s"% topk_k)
log("num_prototypes set to: %s"% num_prototypes)
log("incorrect_class_connection: %s"% incorrect_class_connection)
log("deformable_conv_hidden_channels: %s"% deformable_conv_hidden_channels)

log("Random seed: %d"% rand_seed)
log("Add on layers type: %s"% add_on_layers_type)

img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

from settings import train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

if 'stanford_dogs' in train_dir:
    num_classes = 120
else:
    num_classes = 200
log("{} classes".format(num_classes))

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

if 'augmented' not in train_dir:
    log("Using online augmentation")
    # train_dataset = datasets.ImageFolder(
    train_dataset = CustomImageFolder(
        train_dir,
        # transforms.Compose([
        #     transforms.RandomAffine(degrees=(-25, 25), shear=15),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.Resize(size=(img_size, img_size)),
        #     transforms.ToTensor(),
        #     normalize,
        # ]))
        transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            # transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Resize(size=(img_size, img_size)),  # 这个放在变换的最后，确保所有变换后的图像都有相同的尺寸
            transforms.ToTensor(),
            normalize,
        ]))
else:
    # train_dataset = datasets.ImageFolder(
    train_dataset = CustomImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
# train_push_dataset = datasets.ImageFolder(
train_push_dataset = CustomImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
# test_dataset = datasets.ImageFolder(
test_dataset = CustomImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_shape,
                            num_classes=num_classes, topk_k=topk_k, m=m,
                            add_on_layers_type=add_on_layers_type,
                            using_deform=using_deform,
                            incorrect_class_connection=incorrect_class_connection,
                            deformable_conv_hidden_channels=deformable_conv_hidden_channels,
                            prototype_dilation=2, num_offset_head=num_offset_head,
                            dropout=dropout, clip_offset=clip_offset)
    
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    joint_optimizer_lrs['features'] = 1e-5
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.conv_offset.parameters(), 'lr': joint_optimizer_lrs['conv_offset']},
 {'params': ppnet.last_layer.parameters(), 'lr': joint_optimizer_lrs['joint_last_layer_lr']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs, weight_decay=weight_decay)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
log("joint_optimizer_lrs: ")
log(str(joint_optimizer_lrs))

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs, weight_decay=weight_decay)
log("warm_optimizer_lrs: ")
log(str(warm_optimizer_lrs))

from settings import warm_pre_offset_optimizer_lrs
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    warm_pre_offset_optimizer_lrs['features'] = 1e-5
warm_pre_offset_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_pre_offset_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_pre_offset_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.features.parameters(), 'lr': warm_pre_offset_optimizer_lrs['features'], 'weight_decay': 1e-3},
]
warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs, weight_decay=weight_decay)

warm_lr_scheduler = None
if 'stanford_dogs' in train_dir:
    warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
    log("warm_pre_offset_optimizer_lrs: ")
    log(str(warm_pre_offset_optimizer_lrs))

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs, weight_decay=weight_decay)

# weighting of different training losses
from settings import coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_warm_epochs, num_train_epochs, push_epochs, \
                    num_secondary_warm_epochs, push_start
wandb.config.update({
    'num_warm_epochs': num_warm_epochs,
    'num_train_epochs': num_train_epochs,
    'push_epochs': push_epochs,
    'num_secondary_warm_epochs': num_secondary_warm_epochs,
    'push_start': push_start
})

# train the model
print('Start training.')
log('-------Training Process-------')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False, self_mask=self_mask, l1=l1)
        wandb.log({'train/accuracy': accu})
    elif epoch >= num_warm_epochs and epoch - num_warm_epochs < num_secondary_warm_epochs:
        tnt.warm_pre_offset(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_pre_offset_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False, self_mask=self_mask, l1=l1)
        wandb.log({'train/accuracy': accu})
        if 'stanford_dogs' in train_dir:
            warm_lr_scheduler.step()
    else:
        if epoch == num_warm_epochs + num_secondary_warm_epochs:
            ppnet_multi.module.initialize_offset_weights()
        tnt.joint(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=True, self_mask=self_mask, l1=l1)
        wandb.log({'train/accuracy': accu})
        joint_lr_scheduler.step()

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log, subtractive_margin=subtractive_margin, l1=l1)
    wandb.log({'test/accuracy': accu})
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        print('finish push')
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log, l1=l1)
        print('finish test')
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if not last_layer_fixed:
            tnt.last_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, 
                            subtractive_margin=subtractive_margin, self_mask=self_mask, l1=l1)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log, l1=l1)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
logclose()

