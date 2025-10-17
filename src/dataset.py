import os
import PIL.Image
import numpy as np
import torchvision.transforms as ttran
from torch.utils.data import Dataset
import pandas as pd
import torch
from math import floor

def get_grid_cells(boxes, grid_size, img_size=(448, 448)):
    """
        For a list of boxes (c, x, y, w, h), gets the corresponding grid cells (indices).
        Values are expected to be normalized to image size.
    """
    
    cell_size = (img_size[0]/grid_size, img_size[1]/grid_size)

    return torch.tensor([[floor((box[1]*img_size[0])/cell_size[0]), 
             floor((box[2]*img_size[1])/cell_size[1])] for box in boxes]) # we only need to consider the center values


class PressureUlcers(Dataset):

    def __init__(self, images_path, labels_path, transform, device=None, grid_size=7, num_bounding_boxes=2, img_size=(448,448), num_classes=20):
        """
        - *images_path*: folder with each image to load.
        - *labels_path*: folder with .txt files with corresponding labels. Each file must be named exactly like its counterpart.
        - *transform*: transform to apply to an image before loading it.
        - *device*: device in which to keep loaded data (cuda / cpu).
        """
        super().__init__()
        # We need to load everything in a specific format, appropriate for YOLO
        # We'll store only the img paths to use and labels.

        self.transform = transform
        self.X = []
        self.y = []

        # For each folder
        i = 0
        for folder_name in os.listdir(images_path):

            # Get image paths
            folder_files = os.listdir(os.path.join(images_path, folder_name))

            # For each image, get its corresponding labels (it can have multiple)
            for file_name in folder_files:
                self.X.append(os.path.join(images_path, folder_name, file_name))
                file_name = f'{file_name.split('.')[0]}.txt'

                answer = torch.zeros(grid_size, grid_size, num_bounding_boxes*5 + num_classes, dtype=torch.float32) # empty matrix (filled with 0s)
                
                 # If it's a classified folder
                if folder_name != 'Invalid':
                    # Read file
                    label_file_path = os.path.join(labels_path, file_name)
                    data = torch.tensor(pd.read_csv(label_file_path, sep=' ', header=None).to_numpy(), dtype=torch.float32) # target bounding boxes (objects)
                
                    # Separate the data
                    #class_id, x, y, w, h = (data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4])
                    cell_indices = get_grid_cells(data, grid_size, img_size).to(torch.int)

                    bb_data = torch.zeros(data[:, 1:].shape[0], data[:, 1:].shape[1]+1) # +1 for the confidence value
                    bb_data[:, 0] = 1 # setting confidence values for these boxes which have objects
                    bb_data[:, 1:] = data[:, 1:] # bounding box data without class predictions

                    class_probabilities = torch.zeros(data.shape[0], num_classes, dtype=torch.float32) # empty class probabilities (all zeros)
                    class_probabilities[range(data.shape[0]), data[:, 0:1].to(torch.int)] = 1.0 # fill in the right classes with 1
                    # class probabilities is a matrix in which each row is an object-filled cell and each column is a class probability


                    answer[cell_indices[:, 0], cell_indices[:, 1], :bb_data.shape[1]] = bb_data # fill the cells with the target objects (if there are more available bounding box data, fill it with zeros)
                    answer[cell_indices[:, 0], cell_indices[:, 1], -num_classes:] = class_probabilities # fill in the class probabilities for each cell with an object
                    
                    #print(f'Object-filled indices: {cell_indices}')
                    #print(data, data.shape, data[:, 0:1])
                    #print(f'Answer format: {answer.shape}')
                    #print(f'data.shape: {data.shape}')
                    #print(f'bb_data: {bb_data}')
                    #print(f'Class probability shape: {class_probabilities.shape}, {data[:, 0:1].shape}')
                    #print(cell_indices, answer[cell_indices[:, 0], cell_indices[:, 1], :bb_data.shape[1]], answer.shape)
                    #print(f'Data shape')
                
               
                    # print(answer[answer[:, :, 0] == 1].shape) # showing objects in target images
                self.y.append(answer)
            print(f'folder: {folder_name}\nfiles: {folder_files}\n\n')
        self.y = torch.stack(self.y) # converting final answer from list to tensor

    def __getitem__(self, index):
        """
        Returns, for **index**:
            ~PIL_img: unaltered PIL image.~
            - transformed_img: tensor of transformed PIL image.
            ~labels: list of tensors, each with format [class, x, y, width, height].~
            - new_labels: tensor matrix of shape (grid_size, grid_size, num_of_bounding_boxes * 5 + C), 
                in which the 5 values are: confidence_value, x, y, w, h and the rest are the class probabilities.
                The class probabilities will be 0 for all classes except the target one, which will be 1.
                The confidence value will be 1 if there's an object. 0, otherwise.
        """
        pil_img = PIL.Image.open(self.X[index]).convert('RGB')
        return self.transform(pil_img).to(torch.float32), self.y[index]

    def __len__(self):
        return len(self.X)