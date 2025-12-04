import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src import config

def get_dataloaders():
    """
    Returns DataLoaders for Train, Val, and Test
    """
    # Trasformations (Data Augmentation  only on Train)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #this number are from imagenet for resnet. we need to adjust the colors to resnet
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Dataset creation
    image_datasets = {x: datasets.ImageFolder(os.path.join(config.PROCESSED_DATA_DIR, x), 
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    # DataLoader creation
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=config.BATCH_SIZE, 
                                 shuffle=(x=='train'), # Shuffle only the train
                                 num_workers=config.NUM_WORKERS)
                   for x in ['train', 'val', 'test']}
    
    class_names = image_datasets['train'].classes
    
    return dataloaders, class_names