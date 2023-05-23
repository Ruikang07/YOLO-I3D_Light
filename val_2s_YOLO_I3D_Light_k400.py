# by Ruikang Luo
import os
import sys
import copy
from pathlib2 import Path
from tqdm import tqdm
#from torchinfo import summary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from i3d_src.utils import *
from i3d_src.opts import parser
from i3d_src.i3d_s224 import Unit3Dpy
from i3d_src.i3d_s224 import I3D as I3D224
from yolo_i3d_n112 import Unit3Dpy, YOLO_3D_Bottom
from yolo_i3d_n112 import I3D as I3D112
from i3d_src.DataLoader_2s_s224t32d1f32_s112t128d2f64_ge128 import RGB_jpg_train_Dataset, RGB_jpg_val_Dataset

def val_model(model1, model2, model_yolo_3d_bottom, criterion, val_data_loader, val_dataset_size):
       
    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size

    # Each epoch has a training and validation phase        
    for phase in ['val']:
        model1.eval()  # Set model to evaluate mode
        model2.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_correct = 0

        # Iterate over data.
        progress = tqdm(data_loaders[phase])
        for idx, (data,labels) in enumerate(progress):  
        #for idx, (data,labels) in enumerate(data_loaders[phase]):                                    
            data1 = data[0]
            data2 = data[1]
            #print("data1.shape, data2.shape = ", data1.shape, data2.shape)
            data1 = data1.to(device)
            data2 = data2.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            #optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                _, out_logits1 = model1(data1)
                _, out_logits2 = model2(model_yolo_3d_bottom(data2))                              
                out_logits = 0.5*(out_logits1 + out_logits2)
                out_softmax = torch.nn.functional.softmax(out_logits, 1)

                probs, preds = torch.max(out_softmax.data.cpu(), 1)
                loss = criterion(out_softmax.cpu(), labels.cpu())                                         

                # statistics
                running_loss += loss.item() * data1.shape[0]
                running_correct += torch.sum(preds == labels.data.cpu())
                
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_correct.double() / dataset_sizes[phase]                    

        print('phase={}, Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return 


def load_and_freeze_model(model, num_classes, num_freeze=20):
    if model.num_classes != num_classes :
        model.num_classes = num_classes
        print("\nnew num_classes = ", num_classes)
        model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                    out_channels=num_classes,
                                    kernel_size=(1, 1, 1),
                                    activation=None,
                                    use_bias=True,
                                    use_bn=False)
        
    #freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    counter = 0
    print("\nnum_freeze = ", num_freeze)
    for child in model.children():
        counter += 1
        #print("layer number = ", counter)
        if counter <= num_freeze:
            print("L{}: Layer {} frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
        else:
            print("L{}: Layer {} unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True   
                param.requires_grad = False
    return model 


def main(device, val_data_loader, val_dataset_size):
    
    Num_Classes=400    
    DropoutProb = 0

    weight_path_i3dm224 = "weights/i3d_s224_k400_d1_f32_val_acc_0.6100.pth"
    weight_path_i3dm112 = "weights/yolo_i3d_top_n112_k400_d2_f64_val_acc_0.6146.pth"
    weight_yolo = 'weights/yolov5l.pt' 
     
    model_yolo_3d_bottom = YOLO_3D_Bottom(device, weight_yolo)        
    
    print("prepare model i3dm224")
    model224 = I3D224(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    model224 = load_and_freeze_model(model224, Num_Classes, 20)
    model224.to(device)      
    model224.load_state_dict(torch.load(weight_path_i3dm224))
    print("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path_i3dm224))   
    print("prepare model i3dm112")
    model112 = I3D112(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    model112 = load_and_freeze_model(model112, Num_Classes, 20)
    model112.to(device)      
    model112.load_state_dict(torch.load(weight_path_i3dm112))
    print("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path_i3dm112))
          
    criterion = nn.CrossEntropyLoss()                        

    val_model(model224, model112, model_yolo_3d_bottom, criterion, val_data_loader, val_dataset_size)    


if __name__ == "__main__":
    args = parser.parse_args()
    
    Num_Frames = 32 # it is a fake number
    Batch_Size = 8
    Num_Workers = 12
    data_dir = Path("data/k400_imgs")      
    data_json_dir = "data/k400_imgs/data_json_ge128"  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    file_name =  os.path.basename(__file__)
    print("\nstart running {}".format(file_name))
    print("device = {}".format(device))
    print("cpu_count = {}".format(os.cpu_count()))
    print("batch_size = {}".format(Batch_Size))    
    print("out_frame_num = {}".format(Num_Frames))    
    print("data_dir = {}".format(data_dir))   
    print("data_json_dir = {}".format(data_json_dir))    
    
    classes_path= data_dir / "classes.txt"
    class_names = [i.strip() for i in open(classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
    
    if not os.path.exists(data_json_dir):
        os.makedirs(data_json_dir) 
    data_pairs_file_root = data_json_dir + "/train_data_pairs"     
    
    x = 'val'
    data_pairs_file = data_json_dir + "/val_data_pairs.json"
    val_dataset = RGB_jpg_val_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                        out_frame_num=Num_Frames, x=x)
    val_data_loader = torch.utils.data.DataLoader(\
        val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers)          
        
    val_dataset_size = len(val_dataset)
    print("val_dataset_size = {}".format(val_dataset_size))  
               
    main(device, val_data_loader, val_dataset_size)
        
        