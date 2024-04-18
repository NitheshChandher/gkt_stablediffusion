import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomRotation, RandomHorizontalFlip
class Dataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.prepare_data()
        self.transform = transform
        
    
    def prepare_data(self):
        images=[]
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir,cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir,img_name)
                item = (img_path, self.class_to_idx[cls])
                images.append(item)
        
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path, label = self.images[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def makedataset(path, img_size, bs, mode):

    if mode == 'Train':
        transform = transforms.Compose([Resize((img_size, img_size)),
                           RandomRotation(30),
                           RandomHorizontalFlip(0.5),
                           ToTensor(), 
                           Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    elif mode == 'Validation': 
        transform = Compose([Resize((img_size, img_size)),
                           ToTensor(), 
                           Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    else:
        transform = Compose([Resize((img_size, img_size)),
                           ToTensor()]) 
        
    dataset = Dataset(path,transform)
    dataloder = DataLoader(dataset, batch_size=bs, shuffle=True)

    return dataloder