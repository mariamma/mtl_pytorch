import torch, os
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.io import read_image

class CelebaDataset(VisionDataset):
    def __init__(self, data_dir, split='train', image_size=64, transform=None) -> None:
        super().__init__(data_dir) 
        rep_file = os.path.join(data_dir, 'Eval/list_eval_partition.txt')
        self.img_dir = os.path.join(data_dir, 'Img/img_align_celeba/')
        self.ann_file = os.path.join(data_dir, 'Anno/list_attr_celeba.txt')
        self.image_size = image_size
        
        with open(rep_file) as f:
            rep = f.read()          
        rep = [elt.split() for elt in rep.split('\n')]
        rep.pop()
        

        with open(self.ann_file, 'r') as f:
            data = f.read()
        data = data.split('\n')
        names = data[1].split()
        data = [elt.split() for elt in data[2:]]
        data.pop()
        
        self.img_names = []
        self.labels = []
        for k in range(len(data)):
            assert data[k][0] == rep[k][0]
            if (split=='train' and int(rep[k][1])==0) or (split=='val' and int(rep[k][1])==1) or (split=='test' and int(rep[k][1])==2):
                self.img_names.append(data[k][0])
                self.labels.append([1 if elt=='1' else 0 for elt in data[k][1:]])
                # print(data[k][0], self.img_names[k], self.labels[k])
        
        self.transform = transform
        if transform is None:
            # self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
            # self.transform = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #          std=[0.229, 0.224, 0.225]), transforms.Resize(image_size)])
            self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])         
        


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, index):
        img = read_image(os.path.join(self.img_dir, self.img_names[index]))
        img = img.type(torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        labels = torch.tensor(self.labels[index], dtype=torch.float32)    
        return img, labels