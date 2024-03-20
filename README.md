# Fruits Classification 

Fruits image classification is a project where the main goal is having a clear classification of the images you display by images, or by a live webcam. The project was done in pytorch which is a framework written in python and as a model architecture is built around VGG-16 architecture. 

Links:

[VGG ARCHITECTURE](https://arxiv.org/pdf/1409.1556v6.pdf)

[PyTorch](https://pytorch.org/)

## Installation


```bash
#These are the main to libraries used:
pip install torch
pip install torchvision
```

## Installation

Clone the repository
```bash
git clone 
```
The way of running the model right now is through running inferences I've already written on a notebook file which you will find in the pytorch_utils folder.
```bash
model_testing.ipynb
```
Later updates to the project will enable the possibility to run the model on a live webcam feed.
## The development of the project and issues I've ran into
After getting not so long ago with the neural networks basics I decided to begin the development of a small project, the fruits image classification which has as a main goal to run inferences on both photos and real webcam. 

The project was done in pytorch and as a dataset I've used is [this](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification/data), which consists of about 10,000 images with 5 main classes. For the inital launch of the project I didn't look at a big dataset because I wanted to get the main image processing code done right.

For the image processing I made a custom image dataset class which extends from the Dataset class in PyTorch to ease up the process of training. 
```python
class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


```

The transforms I've used for each image are:
```python
transform_method = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
# Where normalize is calcuated based on the image dataset using another function written 
# in utils.py file that returns the mean and standard deviation for the image dataset.
```

![image](https://github.com/remy-byte/Fruits-image-classification/assets/80782419/119d8456-7182-48ac-b3b7-3ecfb9b6966e)


The main hyperparameters declared are:
```python
num_classes = 5
num_epochs = 30
batch_size = 10
learning_rate = 0.0001

# Where the loss is CrossEntropyLoss, and the optimizer is Adam.
```

### Issues I've ran into: 

While developing the project and training the model I've noticed the results aren't really that high.When training the model the accuracy on the validation set was around 74-77% which is not that big. But when running simple inferences on the model it does a good job at predicting the fruit in the photo.

When deploying the model I used the opencv library and, with it, I practically ran infereneces on the frames but as of right now the model ain't doing a good job at predicting the what fruits are on the webcam feed.


## Future planning phase

- Experiment with other, more complex datasets, but also with the hyperparameters in other to achieve better results with the model in the first place
- Try to properly deploy the model on a webcam that run inferences on each frame and predicts the fruit displayed
