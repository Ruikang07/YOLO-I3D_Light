# by Ruikang Luo
import os
import torch
from i3d_src.utils import *
from yolo_i3d_n112 import YOLO_3D_Bottom
from yolo_i3d_n112 import I3D as I3D112

weight_path_i3dm112 = "weights/yolo_i3d_top_s112_k400_d2_f64_val_acc_0.6146.pth"
n112 = "yolo_i3d_top_n112_k400_d2_f64_val_acc_0.6146.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = "./model_n112/"
if not os.path.exists(save_path):
    os.mkdir(save_path)   


model112 = I3D112(num_classes=400, modality='rgb', dropout_prob=0)
model112.to(device)  

pre_trained = torch.load(weight_path_i3dm112)
new_weights = list(pre_trained.items())
my_model_kvpair = model112.state_dict()
count=0
for key,value in my_model_kvpair.items():
    layer_name, weights = new_weights[count]      
    #mymodel_kvpair[key]=weights.detach().clone()
    my_model_kvpair[key] = weights
    count+=1           
#model112.load_state_dict(torch.load(weight_path_i3dm112))
model112.load_state_dict(my_model_kvpair)

tmp_model112_path = save_path+n112
torch.save(model112.state_dict(), tmp_model112_path) 