import os
#from config import *
import random
import torch
from tqdm import tqdm
import numpy as np
import models.resnet_o as resnet
from torch.utils.data import DataLoader
from torchvision import transforms
from sampler import SubsetSequentialSampler
from models.lenet import LeNet5
from kcenterGreedy import kCenterGreedy
import wandb
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import pickle
import copy  
import xarray as xr
from dask import array as da 
##
# Loss Prediction Loss
# CUDA_VISIBLE_DEVICES = int(os.environ['CUDA_VISIBLE_DEVICES'])

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss



def test(models, epoch, method, dataloaders, args, mode='val'):
    # assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    test_features_list, test_labels_list, test_images_list = [], [], []
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total

    else:
        with torch.no_grad():
            total_loss = 0
            Y_PRED, Y_TRUE, SCORES = [], [], []
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, feat,_ = models['backbone'](inputs)
                if len(test_features_list) < 10000:
                    test_features_list.append(feat.detach().cpu().squeeze())
                    test_labels_list.append(labels.detach().cpu().squeeze())
                    test_images_list.append(inputs.detach().cpu().squeeze())

                # output = F.log_softmax(scores, dim=1)
                # loss =  F.nll_loss(output, labels, reduction="sum")
                _, preds = torch.max(scores.data, 1)
                # total_loss += loss.item()
                # _, preds = torch.max(output, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                # correct += preds.eq(labels).sum()
            
                if (epoch == args.no_of_epochs - 1) or args.early_stop_now:
                    Y_PRED.append(preds.detach().cpu().numpy())
                    Y_TRUE.append(labels.detach().cpu().numpy())
                    SCORES.append(scores.detach().cpu().numpy())

            if (epoch == args.no_of_epochs - 1) or args.early_stop_now:
                Y_PRED = np.concatenate(Y_PRED, axis=0)
                Y_TRUE = np.concatenate(Y_TRUE, axis=0)
                SCORES = np.concatenate(SCORES, axis=0)

                if args.dataset in ['SnapshotSerengeti','SnapshotSerengetiSmall']:
                    target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_category_lut_adapted.df')
                    target_names = target_names['name'].values[:len(np.unique(Y_TRUE))]
                if args.dataset == 'SnapshotSerengeti10':
                    target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_balanced_top_10_category_lut.df')
                cl_report = classification_report(Y_TRUE, Y_PRED,target_names=target_names)
                print(cl_report)
                cycle = 0 # since args.total = True -> only one clyce
                np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_prediction_label',Y_PRED)
                np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_truth_label',Y_TRUE)
                np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_scores_label',SCORES)

        test_features_list = torch.cat(test_features_list, dim=0)
        test_labels_list = torch.cat(test_labels_list, dim=0)
        print(len(test_images_list))
        test_images_list = torch.cat(test_images_list, dim=0).reshape(-1,3,args.image_size,args.image_size)
        print(test_images_list.shape)
        # Number of elements to sample
        num_samples = 5000
        # Generate random indices
        random_indices = torch.randperm(test_features_list.size(0))[:num_samples]
        # Sample elements using the random indices
        test_features_list = test_features_list[random_indices]
        test_labels_list = test_labels_list[random_indices]
        test_images_list = test_images_list[random_indices]
        wandb_log_features(test_features_list,test_labels_list,test_images_list,epoch)
        
        return 100 * correct / total

def test_with_sampler(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if (method == 'JLS') or (method == 'TJLS'):
        models['sampler'].eval()
    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total

def test_with_ssl(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['classifier'].eval()

    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            Y_PRED, Y_TRUE, SCORES = [], [], []
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                _, features = models['backbone'](inputs, inputs, labels)
                scores = models['classifier'](features)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                print('DEBUGGING')
                print(f'scores : {scores.detach().cpu().numpy()}')
                print(f'scores shape: {scores.detach().cpu().numpy().shape}')
                
                if (epoch == args.no_of_epochs - 1) or args.early_stop_now:
                    Y_PRED.append(preds.detach().cpu().numpy())
                    Y_TRUE.append(labels.detach().cpu().numpy())
                    SCORES.append(scores.detach().cpu().numpy())
                print(f'scores shape list: {len(SCORES)}')
                print(f'scores shape concat: {np.concatenate(SCORES,axis=0).shape}')

            if (epoch == args.no_of_epochs - 1) or args.early_stop_now:
                Y_PRED = np.concatenate(Y_PRED, axis=0)
                Y_TRUE = np.concatenate(Y_TRUE, axis=0)
                if args.dataset in ['SnapshotSerengeti','SnapshotSerengetiSmall']:
                    target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_category_lut_adapted.df')
                    target_names = target_names['name'].values[:len(np.unique(Y_TRUE))]
                if args.dataset == 'SnapshotSerengeti10':
                    target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_balanced_top_10_category_lut.df')
                cl_report = classification_report(Y_TRUE, Y_PRED,target_names=target_names)
                print(cl_report)
                np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_prediction_label',Y_PRED)
                np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_truth_label',Y_TRUE)
        
        return 100 * correct / total


def test_with_ssl2(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    models['classifier'].eval()
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, feat = models['backbone'](inputs, inputs)
                scores = models['classifier'](feat)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total

# write method to log the results
def wandb_log_features(test_feature_list,test_labels_list,test_images_list,epoch,sampling_indices=None):
    tsne = TSNE(n_components=2, random_state=1234)
    tsne_embeddings = tsne.fit_transform(test_feature_list)
    d = {
        "feature_1": tsne_embeddings[:, 0],
        "feature_2": tsne_embeddings[:, 1],
        "index": np.arange(len(tsne_embeddings)),
        "labels": test_labels_list,
        "images": list(test_images_list),
        "sampling_indices": sampling_indices
    }
    d = pd.DataFrame(data=d)
    
    if sampling_indices is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
        data=d,
        x="feature_1",
        y="feature_2",
        hue="labels",
        style='sampling_indices',
        ax=ax,
        s=10,
        palette='tab20'
    )
        wandb.log({"tsne": wandb.Image(fig, caption="class + un-/labeled, sampled")})

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
        data=d,
        x="feature_1",
        y="feature_2",
        hue="sampling_indices",
        ax=ax,
        s=10,
        palette=['orange','blue','black','red']
    )
        wandb.log({"tsne": wandb.Image(fig, caption="un-/labeled, sampled")})

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=d,
            x="feature_1",
            y="feature_2",
            hue="labels",
            ax=ax,
            s=10,
            palette='tab20'
        )
        wandb.log({"tsne": wandb.Image(fig, caption="test_data")})
    
    if epoch == 201:
        tx, ty = d.feature_1, d.feature_2
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 3000
        height = 3000
        max_dim = 32
        full_image = Image.new('RGB', (width, height))
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        #mean = [0.4914, 0.4822, 0.4465]
        #std = [0.2023, 0.1994, 0.2010]
        unnormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
        for i in range(len(d)):
            image = d.iloc[i].images
            image = unnormalize(image)
            image = np.transpose(image,(1,2,0))
            #print(image.shape)
            tile = Image.fromarray(np.uint8(image*255),'RGB')
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                        Image.LANCZOS)
            full_image.paste(tile, (int((width-max_dim) * tx[i]),
                            int((height-max_dim) * ty[i])))
        
        wandb.log({"tsne": wandb.Image(full_image,caption='og images')})

def test_without_ssl2(models, epoch, no_classes, dataloaders, args, cycle, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['classifier'].eval()

    if epoch > 1:
        state_dict = torch.load('./MoBYv2AL/%s/backbonehcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))
        # state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder') and not ( k.startswith('encoder.classifier') or  k.startswith('encoder_k') ):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    # del state_dict
    if args.learner_architecture == "vgg16":
        models_b = resnet.dnn_16enc(no_classes).cuda()
    elif args.learner_architecture == "resnet18":
        models_b = resnet.ResNet18E(no_classes).cuda()
    elif args.learner_architecture == "wideresnet28":
        models_b = resnet.Wide_ResNet28(no_classes).cuda()
    elif args.learner_architecture == "lenet5":
        models_b = LeNet5().cuda()

    models['classifier'].eval()
    if epoch > 1:
        models_b.load_state_dict(state_dict, strict=False)
        models['classifier'].load_state_dict(torch.load('./MoBYv2AL/%s/classifierhcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle)))
    models_b.eval()
    total = 0
    correct = 0
    test_features_list, test_labels_list, test_images_list, Y_PRED = [], [], [], []
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            Y_PRED, Y_TRUE , SCORES= [], [], []
            for (inputs, labels) in dataloaders[mode]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                _, feat, _ = models['backbone'](inputs,inputs,labels)
                if len(test_features_list) <10000:
                    test_features_list.append(feat.detach().cpu().squeeze())
                    test_labels_list.append(labels.detach().cpu().squeeze())
                    test_images_list.append(inputs.detach().cpu().squeeze())
                # feat = models_b(inputs)
                scores = models['classifier'](feat)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                if (epoch == args.no_of_epochs-1) or args.early_stop_now:
                    Y_PRED.append(preds.detach().cpu().numpy())
                    Y_TRUE.append(labels.detach().cpu().numpy())
                    SCORES.append(scores.detach().cpu().numpy())

        test_features_list = torch.cat(test_features_list, dim=0)
        test_labels_list = torch.cat(test_labels_list, dim=0)
        print(len(test_images_list))
        test_images_list = torch.cat(test_images_list, dim=0).reshape(-1,3,args.image_size,args.image_size)
        print(test_images_list.shape)
        # Number of elements to sample
        num_samples = 5000
        # Generate random indices
        random_indices = torch.randperm(test_features_list.size(0))[:num_samples]
        # Sample elements using the random indices
        test_features_list = test_features_list[random_indices]
        test_labels_list = test_labels_list[random_indices]
        test_images_list = test_images_list[random_indices]
        wandb_log_features(test_features_list,test_labels_list,test_images_list,epoch)
        
        if (epoch ==args.no_of_epochs-1) or args.early_stop_now: #201 # 121
            Y_PRED = np.concatenate(Y_PRED, axis=0)
            Y_TRUE = np.concatenate(Y_TRUE, axis=0)
            SCORES = np.concatenate(SCORES, axis=0)
            if args.dataset in ['SnapshotSerengeti','SnapshotSerengetiSmall']:
                target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_category_lut_adapted.df')
                target_names = target_names['name'].values[:len(np.unique(Y_TRUE))]
            if args.dataset == 'SnapshotSerengeti10':
                target_names = pd.read_pickle(os.environ['DATA_DIR_PATH']+'/' + 'df_balanced_top_10_category_lut.df')
            cl_report = classification_report(Y_TRUE, Y_PRED,target_names=target_names)
            print(cl_report)
            np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_prediction_label',Y_PRED)
            np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_truth_label',Y_TRUE)
            np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_scores_label',SCORES)
            Y_PRED = Y_PRED[random_indices]
            wandb_log_confusion_matrix(Y_PRED,test_labels_list,"validation",args.dataset)
            wandb.log({'classification report':cl_report})
        
        return 100 * correct / total



iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):


    models['backbone'].train()
    if method == 'lloss':
        models['module'].train()
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        
            
        inputs = data[0].cuda()
        labels = data[1].cuda()


        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        target_loss = criterion(scores, labels)
        # target_loss =  F.nll_loss(F.log_softmax(scores, dim=1), labels)
        if method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss
        # loss = target_loss
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss':
            optimizers['module'].step()
    return loss

    
def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args, subset, labeled_set, data_unlabeled):
    
    print('>> Train a Model.')

    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step(loss)
        if method == 'lloss':
            schedulers['module'].step(loss)

        if epoch > 160 and epoch % 20  == 1:

            acc = test(models, epoch, method, dataloaders, args, mode='test')

            if args.dataset == 'icvl':

                if best_acc > acc:
                    best_acc = acc
                    
                print('Val Error: {:.3f} \t Best Error: {:.3f}'.format(acc, best_acc))
            else:
                if best_acc < acc:
                    best_acc = acc
                    torch.save(models['backbone'].state_dict(), f'./MoBYv2AL/{args.folder_path}/backbone.pth')
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
    return best_acc




def train_epoch_ssl(models, method, criterion, optimizers, dataloaders, 
                                        epoch, epoch_loss, l_lab, l_ulab,schedulers, targets=False):
    models['backbone'].train()
    TRAIN_CLIP_GRAD = True
    idx = 0
    num_steps = len(dataloaders['train'])
    for (samples_1, samples_2) in tqdm(zip(dataloaders['unlabeled'], dataloaders['unlabeled2']), leave=False, total=len(dataloaders['unlabeled'])):
        
        samples_r = samples_1[0].cuda(non_blocking=True)
        samples_a = samples_2[0].cuda(non_blocking=True)
        
        contrastive_loss = models['backbone'](samples_a, samples_r)

        loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
        optimizers['backbone'].zero_grad()

        loss.backward()

        optimizers['backbone'].step()
        schedulers['backbone'].step_update(epoch * num_steps + idx)
        idx +=1

    return loss





def train_with_ssl(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, 
                           epoch_loss, args, l_lab, l_ulab, cycle):
    print('>> Train a Model.')
    best_acc = 0.
    if os.path.isfile(f'./MoBYv2AL/{args.folder_path}/ssl_backbone.pth'):
        models['backbone'].load_state_dict(torch.load(f'./MoBYv2AL/{args.folder_path}/ssl_backbone.pth'))
    for epoch in range(num_epochs):

        best_loss = torch.tensor([99]).cuda()
        loss = train_epoch_ssl(models, method, criterion, optimizers, dataloaders, 
                                        epoch, epoch_loss, l_lab, l_ulab, schedulers, True)

        
        if True or epoch % 20  == 1:
            acc = test_with_ssl(models, epoch, method, dataloaders, args, mode='test')
            print(loss.item())

            if best_acc < acc:
                best_acc = acc
            if best_loss > loss:
                best_loss = loss
                torch.save(models['backbone'].state_dict(), f'./MoBYv2AL/{args.folder_path}/ssl_backbone.pth' )
                

            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    print('>> Finished.')

    
    return best_acc 

def wandb_log_confusion_matrix(y_pred,y_true,caption,dataset):
    if dataset == "cifar10":
        labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck",],
    elif dataset == "SnapshotSerengeti10":
        labels = ['gazellegrants','zebra','gazellethomsons','impala','elephant','giraffe','buffalo','hartebeest','guineafowl','wildebeest']
    elif dataset in ["SnapshotSerengeti",'SnapshotSerengetiSmall']:
        labels_path = os.environ['DATA_DIR_PATH'] + "/" + "df_category_lut_adapted.df"
        labels_df = pd.read_pickle(labels_path)
        labels =  labels_df.name.tolist()
    cf = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    df_cm = pd.DataFrame(cf,index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(40, 40))
    sns.heatmap(df_cm, annot=True, ax=ax)
    wandb.log({
        "confusion_matrix_validate": wandb.Image(fig, caption=caption),
    })

def train_epoch_ssl2(models, method, criterion, optimizers, dataloaders, 
                        epoch, schedulers, cycle, last_inter, args, early_stopper):
    models['backbone'].train()
    models['classifier'].train()
    TRAIN_CLIP_GRAD = True
    idx = 0
    num_steps = len(dataloaders['train'])
    c_loss_gain = 0.5 #- 0.05*cycle
    Y_PRED,Y_TRUE = [],[]
    for (samples,samples_a) in zip(dataloaders['train'],dataloaders['train2']):#, leave=False, total=len(dataloaders['train']):
        
        samples_a = samples_a[0].cuda(non_blocking=True)
        samples_r = samples[0].cuda(non_blocking=True)
        targets   = samples[1].cuda(non_blocking=True)

        contrastive_loss, features, _ = models['backbone'](samples_a, samples_r, targets)

        if (idx % 2 ==0):# or (idx <= last_inter):
            scores = models['classifier'](features)
            target_loss = criterion(scores, targets)
            t_loss = (torch.sum(target_loss)) / target_loss.size(0)
            c_loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
            loss = t_loss + c_loss_gain*c_loss
            # loss.backward()
            if epoch == args.no_of_epochs - 1 or early_stopper.counter + 1 == early_stopper.patience:
                Y_PRED.append(np.argmax(scores.detach().cpu().numpy(),axis=1))
                Y_TRUE.append(targets.detach().cpu().numpy())
        else:
            loss = c_loss_gain *(torch.sum(contrastive_loss)) / contrastive_loss.size(0)
        optimizers['backbone'].zero_grad()
        loss.backward()
        optimizers['backbone'].step()
        if (idx % 2 ==0):# or (idx <= last_inter):
            optimizers['classifier'].zero_grad()
            optimizers['classifier'].step()
    
        # if idx % 100 == 0:
        wandb.log({"task loss":t_loss, "contrastive loss":c_loss, "epoch":epoch, 
            'self-supervised loss':c_loss_gain *(torch.sum(contrastive_loss)) / contrastive_loss.size(0),
            'overall_loss': loss,
            })
        idx +=1
    
    if epoch == args.no_of_epochs - 1 or early_stopper.counter + 1 == early_stopper.patience:
        Y_PRED = np.concatenate(Y_PRED, axis=0)
        Y_TRUE = np.concatenate(Y_TRUE, axis=0)
        wandb_log_confusion_matrix(Y_PRED,Y_TRUE,"training",args.dataset)
    return loss

def evaluate_labeldispersion_unlabeled_data(label_dispersion_metric,models,dataloaders):
    models['backbone'].eval()
    models['classifier'].eval()
    with torch.no_grad():
        # iterate through unlabeled data
        for (inputs, labels, idx) in dataloaders['unlabeled']:
            # predict labeles
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, feat, _ = models['backbone'](inputs,inputs,labels)
            scores = models['classifier'](feat)
            _, preds = torch.max(scores.data, 1)

            # store labeles for index
            for (prediction, label, index) in zip(preds, labels, idx): # loop over batch
                prediction, label, index = prediction.item(), label.item(), index.item()
                if index not in label_dispersion_metric:
                    label_dispersion_metric[index] = [label] # ensure first element is truth label, for analysis purposes.
                    label_dispersion_metric[index].append(prediction)
                else:
                    label_dispersion_metric[index].append(prediction)
    # return label_dispersion_metric

# early stoper class, holding essential metrics
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def check_early_stop(self, validation_loss):
        # print(f'comparing validation_loss {validation_loss} with min_validation_loss{self.min_validation_loss}')
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def validate_epoch_ssl2(models, method, criterion, optimizers, dataloaders, epoch, cycle, args, no_classes):
    models['backbone'].eval()
    models['classifier'].eval()
    c_loss_gain = 0.5
    mode = 'val'
    if epoch > 1:
        state_dict = torch.load('./MoBYv2AL/%s/backbonehcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))
        # state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder') and not ( k.startswith('encoder.classifier') or  k.startswith('encoder_k') ):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    # del state_dict
    if args.learner_architecture == "vgg16":
        models_b = resnet.dnn_16enc(no_classes).cuda()
    elif args.learner_architecture == "resnet18":
        models_b = resnet.ResNet18E(no_classes).cuda()
    elif args.learner_architecture == "wideresnet28":
        models_b = resnet.Wide_ResNet28(no_classes).cuda()
    elif args.learner_architecture == "lenet5":
        models_b = LeNet5().cuda()

    models['classifier'].eval()
    if epoch > 1:
        models_b.load_state_dict(state_dict, strict=False)
        models['classifier'].load_state_dict(torch.load('./MoBYv2AL/%s/classifierhcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle)))
    models_b.eval()
    total = 0
    correct = 0
    test_features_list, test_labels_list, test_images_list, Y_PRED = [], [], [], []

    with torch.no_grad():
        total_loss = 0
        Y_PRED, Y_TRUE , INDICES= [], [], []
        for (inputs, labels, _) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            contrastive_loss, feat, _ = models['backbone'](inputs,inputs,labels)
            # feat = models_b(inputs)
            scores = models['classifier'](feat)
            target_loss = criterion(scores, labels)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            t_loss = (torch.sum(target_loss)) / target_loss.size(0)
            c_loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
            loss = t_loss + c_loss_gain*c_loss
    return loss

def train_with_ssl2(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, 
                           no_classes, args, labeled_data, unlabeled_data, data_train, cycle, last_inter, ADDENDUM, dim_latent):
    print('>> Train a Model.')
    best_acc = 0.
    arg = 0

    l_lab = 0
    l_ulab = 0
    if args.sampling_strategy == 'labeldispersion' and args.continuation:
        path = f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_labeldispersion_indices.pkl'
        with open(path, 'rb') as file:
            label_dispersion_metric = pickle.load(file)
    else:     
        label_dispersion_metric = {}
#    with profile(activities=[
#        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
 #       with record_function("model_training"):
            # if not os.path.isfile('models/moby_backbone_full.pth'):
    
    early_stopper_contrastive_loss = EarlyStopper(patience=5,min_delta=1e-5)
    early_stopper = EarlyStopper(patience=5,min_delta=1e-5)

    for epoch in range(num_epochs):
        print(f'EPOCH NUMBER: {epoch}')
        best_loss = torch.tensor([99]).cuda()
        # loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)
        loss = train_epoch_ssl2(models, method, criterion, optimizers, dataloaders, epoch, schedulers, cycle, last_inter, args, early_stopper)
        schedulers['classifier'].step(loss)
        schedulers['backbone'].step(loss)

        validation_loss = validate_epoch_ssl2(models, method, criterion, optimizers, dataloaders, epoch, cycle, args, no_classes)
        # print(f'early stopping loss: {early_stopper.min_validation_loss:.3f}')
        # print(f'early sstopping counter: {early_stopper.counter:.3f}')
        args.early_stop_now = early_stopper.check_early_stop(validation_loss) #boolean, True -> early stop now
        # args.early_stop_now = False
    

        if epoch == 0: # DEBUGGING
            torch.save(models['backbone'].state_dict(), './MoBYv2AL/%s/backbonehcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle)) 
            torch.save(models['classifier'].state_dict(), './MoBYv2AL/%s/classifierhcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))
        if (epoch > 1 and epoch % 5  == 1) or args.early_stop_now: # DEBUGGING, change back to True
            # acc = test_with_ssl(models, epoch, method, dataloaders, args, mode='test')
            # print(loss.item())

            acc = test_without_ssl2(models, epoch, no_classes, dataloaders, args, cycle, mode='test')
#            acc = 0
            if best_acc < acc:
                torch.save(models['backbone'].state_dict(), './MoBYv2AL/%s/backbonehcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))
                torch.save(models['classifier'].state_dict(), './MoBYv2AL/%s/classifierhcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))
                best_acc = acc

            print('Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
        
        if args.sampling_strategy in ['labeldispersion','corelb','corelbpseudo']:
            print('doing label dispersion loop!')
            if True or epoch % 10 == 1: # change back to 1 after DEBUGGING 
                evaluate_labeldispersion_unlabeled_data(label_dispersion_metric,models,dataloaders)
                with open(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_labeldispersion_indices.pkl', 'wb') as file:
                    pickle.dump(label_dispersion_metric, file, protocol=pickle.HIGHEST_PROTOCOL)

        wandb.log({"training loss":loss, "validation loss":validation_loss})
        
        if args.early_stop_now:
            break
#    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print('>> Finished.')
    
    if args.sampling_strategy=='random':
        # subset = len(unlabeled_data)
        # arg = np.random.randint(subset, size=subset)
        # change to directly returning datapoint indices instead of posistion of datapoints.
        arg = np.random.choice(unlabeled_data, ADDENDUM, replace=False)
        print('returning random points')
        return best_acc, arg
    
    models['classifier'].eval()
    models['backbone'].eval()
    # # 
    features = np.empty((args.batch, dim_latent))
    # #     c_loss =  torch.tensor([]).cuda()
    k_var = 2
    c_loss_m = np.zeros((k_var, args.batch*len(dataloaders['unlabeled'])))

    state_dict = torch.load('./MoBYv2AL/%s/backbonehcss_%s_%d.pth'%(args.folder_path,args.dataset,cycle))

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder') and not ( k.startswith('encoder.classifier') or  k.startswith('encoder_k') ):
            # remove prefix
            state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if args.learner_architecture == "vgg16":
        models_b = resnet.dnn_16enc(no_classes).cuda()
    elif args.learner_architecture == "resnet18":
        models_b = resnet.ResNet18E(no_classes).cuda()
    elif args.learner_architecture == "wideresnet28":
        models_b = resnet.Wide_ResNet28(no_classes).cuda()
    elif args.learner_architecture == "lenet5":
        models_b = LeNet5().cuda()
    models_b.load_state_dict(state_dict, strict=False)
    models_b.eval()

    if args.sampling_strategy == 'corelbpseudo' and cycle >0:
        human_labeled_indices = np.load(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle-1}_human_labeled_indices.npy')
        human_labeled_indices = list(human_labeled_indices)
        combined_dataset = DataLoader(data_train, batch_size=args.batch, 
                                    sampler=SubsetSequentialSampler(unlabeled_data+human_labeled_indices), # restriction applies to corelb pseudo 
                                    pin_memory=False, drop_last=False, num_workers = int(os.environ['NUM_WORKERS']))
    else:
        combined_dataset = DataLoader(data_train, batch_size=args.batch, 
                                    sampler=SubsetSequentialSampler(unlabeled_data+labeled_data), 
                                    pin_memory=False, drop_last=False, num_workers = int(os.environ['NUM_WORKERS']))
    
    features_indices = [] # tracking indices of features
    features_labels = []
    for ulab_data in combined_dataset:
        ulab_data = copy.deepcopy(ulab_data)
        unlab = ulab_data[0].cuda()
        # target = ulab_data[1].cuda()
        feat =  models_b(unlab)
        feat = feat.detach().cpu().numpy()
        feat = np.squeeze(feat)
        features = np.concatenate((features, feat), 0)
        for i in ulab_data[2]: 
            features_indices.append(i)
        for i in ulab_data[1]:
            features_labels.append(i)

    features = features[args.batch:,:] # omit the first batch, since these are empty.
    subset = len(unlabeled_data)
    BASE_BUDGET = 4500
    if cycle >0 and args.sampling_strategy == 'corelbpseudo':
        print(f'new labeled_data size: {len(human_labeled_indices)} old labeled_data size: {len(labeled_data)}')
    labeled_data_size = len(labeled_data)
    print(f'unlabeled set: {subset}, labeled_data_size: {labeled_data_size}, feature size: {len(features)}')
    np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_features',features)
    np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_features_label',features_labels)
    high_lb_indices = [] # for all other sampling strategy not using laber dispersion.
    
    if args.sampling_strategy=='coreset':
        # Apply CoreSet for selection
        new_av_idx = np.arange(subset,(subset + labeled_data_size))
        sampling = kCenterGreedy(features)  
        av_idx_batch = sampling.select_batch_(new_av_idx, ADDENDUM) # batch with new_av_idx
        #change new_av_idx to corresponding datapoint index
        tmp_unlabeled_data = np.array(unlabeled_data)
        batch = tmp_unlabeled_data[av_idx_batch]
            # print(min(batch), max(batch))
        

    if args.sampling_strategy=='labeldispersion':
        # take list of predictions over epoch of unlabeled dataset
        # path = f'./MoBYv2AL/models/{args.dataset}_{args.method_type}_{args.run_id}_labeldispersion_indices.pkl'
        # with open(path, 'rb') as file:
        #     label_dispersion_metric = pickle.load(file)

        # label_dispersion_metric
        sampling = []
        for index, label_list in label_dispersion_metric.items():
            label_list = label_list[1:] # ommits the first element, which is the true label
            # create a dictionary with the datapoint index as keys and the occurence as values
            occurence_dict = {i:label_list.count(i) for i in label_list}
            # sort the dictionary by highest values 
            sorted_occurence_dict = [(k, v) for k, v in sorted(occurence_dict.items(), key=lambda item: item[1],reverse=True)]
            most_common_label = sorted_occurence_dict[0][0]
            most_common_label_occurence = sorted_occurence_dict[0][1]

            # calculate the label dispersion
            label_dispersion = most_common_label_occurence/len(label_list)

            sampling.append((index,label_dispersion))
        
        sampling =sorted(sampling, key=lambda x: x[1],reverse=False) #reverse=False => small label dispersion (0.2) are first.
        batch = [k for k,v in sampling[:ADDENDUM]] # only taking the indices

    if args.sampling_strategy in ['corelb','corelbpseudo']:
        # first apply label dispersion, then coreset
        sampling = []
        for index, label_list in label_dispersion_metric.items():
            label_list = label_list[1:] # ommits the first element, which is the true label
            # create a dictionary with the datapoint index as keys and the occurence as values
            occurence_dict = {i:label_list.count(i) for i in label_list}
            # sort the dictionary by highest values 
            sorted_occurence_dict = [(k, v) for k, v in sorted(occurence_dict.items(), key=lambda item: item[1],reverse=True)]
            most_common_label = sorted_occurence_dict[0][0]
            most_common_label_occurence = sorted_occurence_dict[0][1]

            # calculate the label dispersion
            label_dispersion = most_common_label_occurence/len(label_list)

            sampling.append((index,label_dispersion))
        
        #select indices (unlabeled data) with high lb. filter these out before coreset
        high_lb_indices = [index for index, label_dispersion in sampling if label_dispersion >=1 ]
        position_of_high_lb_indices = [unlabeled_data.index(h_lb_i) for h_lb_i in high_lb_indices]
        # Create a boolean mask where True represents indices to exclude
        exclude_mask = np.isin(np.arange(len(features)), position_of_high_lb_indices)
        features_filtered = features[~exclude_mask]
        print(f'high lb indices length: {len(high_lb_indices)}')
        print(f'features length: {features_filtered.shape}')

        # only do coreset after filtering high lb
        subset = len(unlabeled_data) - len(high_lb_indices) 
        print(f'length subset: {subset}')

        if args.sampling_strategy == 'corelbpseudo' and cycle >0:
            new_av_idx = da.arange(subset,(subset + len(human_labeled_indices)))
        else:
            new_av_idx = da.arange(subset,(subset + labeled_data_size))
        # new_av_idx = np.arange(subset,subset)
        # new_av_idx = xr.DataArray(data=new_av_idx,dims=['0'])
        # new_av_idx = xr.DataArray(data=da.zeros((157784,112615)),dims=["0","1"])
        print(f'new av idx shape: {new_av_idx.shape}')
        print(f'new av idx type: {type(new_av_idx)}')
        print(f'new av idx: {new_av_idx}')
        # features_filtered = xr.DataArray(data=features_filtered,dims=['0','1'])
        # features_filtered = xr.DataArray(da.from_array(features_filtered),dims=['0','1'])
        sampling = kCenterGreedy(features_filtered)  
        av_idx_batch = sampling.select_batch_(new_av_idx, ADDENDUM) # batch with new_av_idx
        #change new_av_idx to corresponding datapoint index
        tmp_unlabeled_data = np.array(unlabeled_data)
        batch = tmp_unlabeled_data[av_idx_batch]

        if cycle <=0:
            human_labeled_indices = list(batch) + labeled_data
            np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_human_labeled_indices',human_labeled_indices)
        elif cycle >0:
            human_labeled_indices = np.load(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle-1}_human_labeled_indices.npy')
            human_labeled_indices = list(human_labeled_indices) + list(batch)
            np.save(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_human_labeled_indices',human_labeled_indices)
        
        for b in batch:
            if b not in unlabeled_data:
                raise Exception(f' {b} not in unlabeled_data')

        
        if args.sampling_strategy == 'corelbpseudo':
            if cycle == 0:
                pseudo_labels = {}
            else:
                with open(f'./MoBYv2AL/results/{args.dataset}_{args.method_type}_{args.run_id}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle-1}_pseudo_labels.pkl', 'rb') as file:
                    pseudo_labels = pickle.load(file)
            # save high label dispersion indices in a file in form{indices: pseudo_label}
            high_label_dispersion_indices = {}
            for index in high_lb_indices:
                pseudo_labels[index] = label_dispersion_metric[index][-1] # since all elements are the same, take the last one.
                high_label_dispersion_indices[index] = label_dispersion_metric[index][-1]

            with open(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_pseudo_labels.pkl', 'wb') as file:
                pickle.dump(pseudo_labels, file, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'./MoBYv2AL/{args.folder_path}/{args.dataset}_{args.method_type}_{args.run_id}_{cycle}_high_lb_indices.pkl', 'wb') as file:
                pickle.dump(high_label_dispersion_indices, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            batch = list(batch) + high_lb_indices

    # randomly choose subset of datapoints (mix of un-/labeled)
    # Number of elements to sample
    num_samples_unlabeled = 1000
    num_samples_labeled = 1000
    num_samples_chosen_samples = 250 if (len(high_lb_indices) > 250 or len(high_lb_indices)==0) else len(high_lb_indices) # 10% of 2500 ADDENDUM

    
    unlabeled_data_without_high_lb_indices = [i for i in unlabeled_data if i not in high_lb_indices]
    # Generate random indices for unlabeled data
    random_indices_unlabeled = random.sample(unlabeled_data_without_high_lb_indices, num_samples_unlabeled)
    # Generate random indices for labeled data
    random_indices_labeled = random.sample(labeled_data, num_samples_labeled)
    # Gerenate random indices for sampled data
    batch_without_high_lb_indices = [i for i in batch if i not in high_lb_indices]
    random_indices_chosen_samples = random.sample(batch_without_high_lb_indices, num_samples_chosen_samples)

    if len(high_lb_indices)==0: # no high_lb_indices for coreset
        random_indices_high_lb_indices = []
    else:
        random_indices_high_lb_indices = random.sample(high_lb_indices, num_samples_chosen_samples)
    
    combined_indices = random_indices_unlabeled + random_indices_labeled + random_indices_chosen_samples + random_indices_high_lb_indices
    include_mask = np.isin(features_indices, combined_indices)
    features_filtered = features[include_mask]

    features_labels_filtered = np.array(features_labels)[include_mask]
    features_indices_filtered = np.array(features_indices)[include_mask]

    label_list = ['empty'] * len(features_indices_filtered)
    is_sampled = ['empty'] * len(features_indices_filtered)

    for i, index in enumerate(features_indices_filtered):
        if index in random_indices_unlabeled:
            label_list[i] = 'unlabeled'
        elif index in random_indices_labeled:
            label_list[i] = 'labeled'
        elif index in random_indices_chosen_samples:
            label_list[i] = 'sampled'
        elif index in random_indices_high_lb_indices:
            label_list[i] = 'pseudo_labeled'

    is_sampled = [i if i== 'sampled' else 'old' for i in label_list] # crate list for plotting
    wandb_log_features(features_filtered, features_labels_filtered, np.zeros((features_filtered.shape)), epoch, sampling_indices=label_list)
    
    other_idx = [x for x in range(subset) if x not in batch]
    # # np.save("selected_s.npy", batch)
    # arg = np.array(other_idx + batch)
    arg = np.array(batch)

    return best_acc, arg
