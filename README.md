# DAPPNet: Several Tricks to Augment Deformable ProtoPNet

## Abstract

The deformable prototypical part network (Deformable ProtoPNet, DPPN) is an interpretable image classifier that combines the strengths of deep learning and the interpretability of case-based reasoning, as proposed by Jon Donnelly et al. 
DPPN uses prototypes consisting of spatially deformable prototypical parts to present a flexible, interpretable and convincing image in training dataset that ``looks like" the input image.
However, there is room for further enhancement. We identified three primary issues with the initial model and implemented three specific tricks to address these shortcomings. These improvements not only boosted the model's accuracy by up to n\% but also substantially enriched the interpretability and detailed representation of visual features, enhancing overall model effectiveness.

## Declaration

1. The dataset is available at [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/).
2. Our code is based on [Deformable ProtoPNet](https://github.com/jdonnelly36/Deformable-ProtoPNet).
3. We implemented 4 tricks specified in our report.
4. The experiment is run on 2 NVIDIA Tesla V100.
5. Prerequisites: Python version 3.8.5; PyTorch (version 1.8.1), TorchVision (version 0.9.1), NumPy (version 1.20.2), cv2 (version 4.5.1)

## Usage

1. Split the dataset into train and test. 

1. Use `bash run.sh` to train the model. Example:
    ``` python
    python3 main.py -gpuid='0, 1' \
                    -m=0.1 \
                    -last_layer_fixed=False \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -dropout=0.0 \
                    -num_prototypes=1200 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=42 \
                    -num_offset_head=1 \
                    -weight_decay=0 \
                    -clip_offset=False \
                    -self_mask=False \
                    -l1=True \
                    -model_name='test'
    ```
3. Use `bash test.sh` to run local/ global test. Example:
    ```python
    python local_analysis.py -gpuid '1,2' \
                         -modeldir 'saved_models/multihead005' \
                         -model '50push0.8237.pth' \
                         -imgdir 'datasets/CUB_200_2011/test/035.Purple_Finch' \
                         -img 'Purple_Finch_0046_27295.jpg' \
                         -imgclass 34
    ```
    ```python
    python global_analysis.py -gpuid '0, 1' \
                         -modeldir 'saved_models/tune105_onlymask&dataaug' \
                         -model '50push0.8420.pth'
    ```

## Key Implemention

We implement 4 core tricks as mentioned in report.

### Data Augmentation

We try a severe version and a mild version.

Severe:

```python
transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
```

Mild:

```python
transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
```

What's more, we add dropout and larger top_k value to prevent overfitting. The option can be found in `run.sh`. You can use this option to control the model hyperparameters but we do not recommend this b ecause we find it doesn't improve the model performance during training. All the experimental results and analysis is at Section 3 of report.

### Multi-head Offset

This trick is mainly implemented in `model.cos_activation`, the core code is as follows,

```python
if self.num_offset_head == 1:
    # compute activation dot
else:
    offsets = torch.split(offset, self.num_offset_head, dim=1)
    activ_dots = []
    for ofst in offsets:
        activ_dot = NormPreserveDeformConvFunction.apply(x_normalized, ofst, 
                                                    normalized_prototypes, 
                                                    torch.zeros(self.prototype_shape[0]).cuda(), #bias
                                                    (1, 1), #stride
                                                    self.prototype_padding, #padding
                                                    prototype_dilation, #dilation
                                                    1, #groups
                                                    1, #deformable_groups
                                                    1, #im2col_step
                                                    True) #zero_padding
        activ_dots.append(activ_dot)
    stacked_activ_dots = torch.stack(activ_dots)
    activations_dot, _ = torch.max(stacked_activ_dots, dim=0)
```

The tensor `offsets` is of size `batchsize * (latent_dim * num_offset_head) * h * w` computed by a convolutional layer, then the dim `1` is split to get `num_offset_head` independent offset maps. The idea is to calculate maximal confidence along the axis of offset maps, and get a best representation overall. You can use the `-num_offset_head` option to modify the hyperparameter.

### Self-Masking Mechanism

Self-masking mechanism is used to prevent tarining image from recognizing itself in the prototype. To implement this, we first use a tensor to store the image number of each prototype in `model.py`:

```python
self.image_index_of_prototypes = torch.full((self.num_prototypes,), -1, dtype=torch.long)
```
where all the entries are initialized -1.

To update it, we use `proto_bound_boxes` provided in `push.push_prototypes`:
```python
image_index_of_prototypes_tensor = torch.tensor(proto_bound_boxes[:, 0], dtype=torch.long, device='cuda')
prototype_network_parallel.module.image_index_of_prototypes = image_index_of_prototypes_tensor
if isinstance(prototype_network_parallel, torch.nn.DataParallel):
    for device_id in prototype_network_parallel.device_ids:
        device_tensor = image_index_of_prototypes_tensor.to(device_id)
        getattr(prototype_network_parallel.module, 'image_index_of_prototypes').data.copy_(device_tensor)
```
`proto_bound_box[0]` exactly stores the image index of a prototype image. To ensure the multi-GPU cooperation, we use broadcast to update the parameter.

Next, the core idea of self-mask is to mask repeated images. To implement this, we must know exactly which image are in the batch. However, in contrast to the push process, in train process, the loaded images are shuffled, so we cannot use the same trick to know image index as before. To implement this, I modified the dataset class:
```python
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
```
While the standard `ImageFolder` class returns only `sample` and `target`, the `CustomImageFolder` returns also the index of image before shuffle. In this way, in train and test process, we can know indecies of images in the batch. 

```python
if is_train and self_mask:
        a = index.cuda()
        b = model.module.image_index_of_prototypes.cuda()
        mask = isin(a, b)
        elements_in_b = a[mask]
        i_can_see_myself = [torch.where(b == item)[0][0] for item in elements_in_b]  # positions of elements that can see themselves.
        
        if len(i_can_see_myself) > 0:
            log(f'i_can_see_myself: {i_can_see_myself}')
            i_can_see_myself = torch.tensor(i_can_see_myself, dtype=torch.long, device='cuda')
            original_prototypes = model.module.prototype_vectors[i_can_see_myself].clone()
            model.module.prototype_vectors.data[i_can_see_myself] = 0
```
These codes find out which images in the batch are in the prototype. We implement mask by set these prototypes to be zero, and restore it at the end of the batch.

Also, we've mentioned an expectation-completing method in the report, which can be further improved. We implemented this by transfer a more parameter in the `model.cos_activation` method:
```python
if is_train and not is_push and index_of_the_batch is not None and len(index_of_the_batch) > 0:
        index_of_the_batch = index_of_the_batch.cuda()
        # print("index_of_the_batch: ", index_of_the_batch)
        hist = torch.histc(index_of_the_batch, bins=self.num_classes, min=0, max=self.num_classes * self.num_prototypes_per_class - 1)  # how many prototypes per class
        full = torch.full((self.num_classes,), self.num_prototypes_per_class, dtype=torch.int).cuda()
        factor = (self.num_prototypes_per_class) / (full - hist + self.epsilon_val)  # the fraction should be multiplied to the prototype, 6/4
        factor = factor.repeat_interleave(6).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        normalized_prototypes = normalized_prototypes * factor
```
In these codes, `hist` enumerate how many repeated prototype there are in the batch for each class, and factor is the factor mentioned in Section 5.2 of the report. What's more, to implement this, we also find a bug in the oringinal code, that is, the flag `is_train` in the method `cos_activation` is not used as expected. To fix this bug, we add a more flag `is_push`. Although this approach is not so elegent, it maintains the usage that the code was expected to be.

### L1 Loss Tuning
We implement this trick to verify the discrepancy between the main theorem and practice. We introduce a new loss term, which is specified in Section 3 of the report. We implement this learning objective mainly in `_train_or_test`:
```python
if use_l1_mask:
    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
    if l1:
        l1 = ((model.module.last_layer.weight * l1_mask).norm(p=1)-(model.module.last_layer.weight * l1_mask).sum()) / 2
    else:
        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    
else:
    if l1:
        l1 = (model.module.last_layer.weight.norm(p=1) - model.module.last_layer.weight.sum()) / 2
    else:
        l1 = model.module.last_layer.weight.norm(p=1)
```
where the upper codes is our learning objective.

## Contributing

Zixuan Cao (2022011336): literature research, theoretical analysis, implementation of Trick \#3 and 4, writing report.

Xiang Ji (2022010881): proposal writing, experiments, implementation of Trick \#1 and 2, writing report.
