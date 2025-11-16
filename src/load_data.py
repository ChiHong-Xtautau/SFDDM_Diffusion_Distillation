from torchvision import datasets, transforms
import torch
import torchvision


def load_dataset(batch_size=8, dataset='CelebA', dataset_dir=None):
    train_loader = None
    if dataset == 'CelebA':
        def set_label(attr):
            attr1 = 20
            return attr[attr1]

        transform = transforms.Compose([
            transforms.CenterCrop((160, 160)),
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CelebA(root=dataset_dir, split='train', target_type="attr", download=True,
                        target_transform=set_label, transform=transform), batch_size=batch_size, shuffle=True)
    elif dataset == 'Cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif dataset == 'Bedroom':
        train_set = datasets.LSUN(root=dataset_dir, classes=['bedroom_train'],
                             transform=transforms.Compose([
                                        transforms.CenterCrop((256, 256)),
                                        transforms.Resize([128, 128]),
                                        # transforms.Resize([64, 64]),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

        subset_size = int(len(train_set) / 20)
        subset_indices = list(range(0, subset_size))
        train_set = torch.utils.data.Subset(train_set, subset_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    elif dataset == 'Dining_room':
        train_set = datasets.LSUN(root=dataset_dir, classes=['dining_room_train'],
                             transform=transforms.Compose([
                                        transforms.CenterCrop((256, 256)),
                                        # transforms.Resize([128, 128]),
                                        transforms.ToTensor(),
                                    ]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    elif dataset == 'Church':
        train_set = datasets.LSUN(root=dataset_dir, classes=['church_outdoor_train'],
                             transform=transforms.Compose([
                                        transforms.CenterCrop((256, 256)),
                                        transforms.Resize([64, 64]),
                                        # transforms.Resize([128, 128]),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    elif dataset == 'Cat':
        data_path = dataset_dir + '/lsun_cat/'
        print(data_path)
        transform = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif dataset == 'Imagenet':
        data_path = dataset_dir + '/tiny-imagenet-200/diff'
        print(data_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([32, 32]),
        ])
        train_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif dataset == 'CelebA_HQ':
        data_path = dataset_dir + '/celeba_hq_256/'
        print(data_path)
        transform = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize([32, 32]),
            # transforms.Resize([128, 128]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    return train_loader




