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


def train_model(opt,save_path, model1, model2, model_yolo_3d_bottom, \
            criterion, optimizer, lr_i, lr_i_max, lr_list, num_epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size, s0, s224, s112):
       
    since = time.time()

    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size

    best_model1_wts = copy.deepcopy(model1.state_dict())
    best_model2_wts = copy.deepcopy(model2.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    optimizer.param_groups[0]['lr'] = lr_list.pop(0)
    lr_pre = optimizer.param_groups[0]['lr']
    len_train_loader = len(train_data_loaders)
    
    for epoch in range(num_epochs):
        for i in range(len_train_loader):     
            data_loaders['train'] = train_data_loaders[i]
            dataset_sizes['train'] = train_dataset_sizes[i]         
                           
            lr_cur = optimizer.param_groups[0]['lr']

            lr_i += 1
            
            if lr_i > lr_i_max :
                if lr_list: 
                    optimizer.param_groups[0]['lr'] = lr_list.pop(0)
                    lr_cur = optimizer.param_groups[0]['lr']
                    lr_i = 1
                else:
                    log("lr_list is empty")
                    return
                
            if lr_cur != lr_pre :
                model1.load_state_dict(best_model1_wts)  
                model2.load_state_dict(best_model2_wts) 
                optimizer.load_state_dict(best_optimizer_state) 
                optimizer.param_groups[0]['lr'] = lr_cur                         
                lr_pre = lr_cur   
                        
            #print("\n")
            log('-' * 40)
            log('Epoch={}, i={}, lr={}, lr_i={}, lr_i_max={}'.format(epoch, i, lr_cur, lr_i, lr_i_max))
            

            # Each epoch has a training and validation phase        
            for phase in ['train', 'val']:
                if phase == 'train':
                    model1.train()  # Set model to training mode
                    model2.train()  # Set model to training mode
                else:
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
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        _, out_logits1 = model1(data1)
                        _, out_logits2 = model2(model_yolo_3d_bottom(data2))                              
                        out_logits = 0.5*(out_logits1 + out_logits2)
                        out_softmax = torch.nn.functional.softmax(out_logits, 1)

                        probs, preds = torch.max(out_softmax.data.cpu(), 1)
                        loss = criterion(out_softmax.cpu(), labels.cpu())                   

                        if phase == "train":
                            loss.backward()
                            optimizer.step()                       

                        # statistics
                        running_loss += loss.item() * data1.shape[0]
                        running_correct += torch.sum(preds == labels.data.cpu())
                       
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_correct.double() / dataset_sizes[phase]                    

                if phase == 'train':
                    #scheduler.step()
                    train_acc = epoch_acc
                    train_loss = epoch_loss

                log('Epoch={}, i={}, phase={}, Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, i, phase, epoch_loss, epoch_acc))
                

                if phase == 'val':
                                       
                    log_loss('{}, {}, {:e}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.\
                        format(epoch,i,lr_cur,train_loss,train_acc,epoch_loss,epoch_acc))    
                                        
                    d_t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    s1 = s0 + d_t + "_"            

                    tmp_model1_path = save_path+s1+s224+"{}_lr{:e}_e{}_i{}_acc{:.4f}_{:.4f}.pth".\
                        format(opt, lr_cur, epoch, i, train_acc, epoch_acc)
                    torch.save(model1.state_dict(), tmp_model1_path) 
                    
                    tmp_model2_path = save_path+s1+s112+"{}_lr{:e}_e{}_i{}_acc{:.4f}_{:.4f}.pth".\
                        format(opt, lr_cur, epoch, i, train_acc, epoch_acc)
                    torch.save(model2.state_dict(), tmp_model2_path)      
                                                           
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_train_acc = train_acc
                    best_model1_wts = copy.deepcopy(model1.state_dict()) 
                    best_model2_wts = copy.deepcopy(model2.state_dict()) 
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    lr_i = 0              
                    log('*************** best_model_wt is from e_{} i_{}***************'.format(epoch, i))

            print()

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  
    log('Best train Acc: {:4f}, Best val Acc: {:4f}'.format(best_train_acc, best_acc))

    return 


def load_and_freeze_model(model, num_classes, num_freeze=0):
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
            log("L{}: Layer {} frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
        else:
            log("L{}: Layer {} unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True   
    return model   


def main(train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size, \
            model224, model112, model_yolo_3d_bottom):
    s0 = "2s_YOLO_I3D_Light_k400_"
    s224 = "s224_d1_f32_"
    s112 = "n112_d2_f64_"        
    
    Num_Epochs = 100

    lr_list = [1.0e-4, 1.0e-5, 1.0e-6]  
    lr = lr_list[0]
    lr_i = 0
    lr_i_max = num_sub_dataset

    opt_method = "Adam"
    weight_decay = 1.0e-4
    momentum = 0.9    
    
    save_path = "./model_i3d/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)     
        
       
    optim_paras1 = list(filter(lambda p: p.requires_grad, model224.parameters()))
    optim_paras2 = list(filter(lambda p: p.requires_grad, model112.parameters()))
    optim_paras = set(optim_paras1+optim_paras2)
    #optim_paras = filter(lambda p: p.requires_grad, model112.parameters())
    
    if opt_method == 'Adam' :
        optimizer = optim.Adam(optim_paras, lr=lr, weight_decay=weight_decay)
        log("adam: lr={}, weight_decay={:e}".format(lr, weight_decay)) 
        opt='adam'
    elif opt_method == 'SGD' :
        optimizer = optim.SGD(optim_paras, lr=lr, momentum=momentum, dampening=0, weight_decay=weight_decay) 
        log("sgd: lr={:e}, momentum={:.1f}, weight_decay={:e}".format(lr, momentum, weight_decay))
        opt='sgd'
    
    criterion = nn.CrossEntropyLoss()                
    optimizer.param_groups[0]['lr'] = lr          

    print("start running train_model()")
    train_model(opt,save_path, model224, model112, model_yolo_3d_bottom, \
            criterion, optimizer, lr_i, lr_i_max, lr_list, Num_Epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size, s0, s224, s112)    


if __name__ == "__main__":
    args = parser.parse_args()
        
    Num_Frames = 32 # it is a fake number

    Batch_Size = 8
    Num_Workers = 12
    Num_Classes=400    
    DropoutProb = 0.4
    
    data_dir = Path("data/k400_imgs")      
    data_json_dir = "data/k400_imgs/data_json_ge128"  
    num_sub_dataset = 10

    weight_path_i3dm224 = "weights/i3d_s224_k400_d1_f32_val_acc_0.6100.pth"
    weight_path_i3dm112 = "weights/yolo_i3d_top_n112_k400_d2_f64_val_acc_0.6146.pth"
    weight_yolo = 'weights/yolov5l.pt'  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_yolo_3d_bottom = YOLO_3D_Bottom(device, weight_yolo) 

    log = Log(dir='log')
    log_loss = Log_Loss(dir='log')

    file_name =  os.path.basename(__file__)
    log("\nstart running {}".format(file_name))
    log_loss('epoch,sub_epoch,lr,train_loss,train_acc,val_loss,val_acc')
    log("device = {}".format(device))
    log("cpu_count = {}".format(os.cpu_count()))
    log("batch_size = {}".format(Batch_Size))    
    log("out_frame_num = {}".format(Num_Frames))    
    log("data_dir = {}".format(data_dir))   
    log("data_json_dir = {}".format(data_json_dir))     
    
    log("prepare model i3dm224")
    model224 = I3D224(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    model224 = load_and_freeze_model(model224, Num_Classes, 0)
    model224.to(device)      
    model224.load_state_dict(torch.load(weight_path_i3dm224))
    log("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path_i3dm224))   
    log("prepare model i3dm112")
    model112 = I3D112(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    model112 = load_and_freeze_model(model112, Num_Classes, 10)
    model112.to(device)      
    model112.load_state_dict(torch.load(weight_path_i3dm112))
    log("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path_i3dm112))
    
    

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
    log("val_dataset_size = {}".format(val_dataset_size))  
           
    
    x = 'train'
    train_datasets = []
    train_data_loaders = []
    train_dataset_sizes = []
   
    log("number of sub-train-datasets = {}".format(num_sub_dataset)) 
    for i in range(num_sub_dataset):
        data_pairs_file = data_pairs_file_root+"_"+str(num_sub_dataset)+"_"+str(i+1)+".json"
        train_datasets.append(RGB_jpg_train_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                            out_frame_num=Num_Frames, x=x))  

        train_data_loaders.append(torch.utils.data.DataLoader(\
            train_datasets[i], batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers))

        train_dataset_sizes.append(len(train_datasets[i]))
        
        log("train_datasets_sizes[{}] = {}".format(i, train_dataset_sizes[i]))    
        
       
    main(train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size, \
            model224, model112, model_yolo_3d_bottom)
        
        