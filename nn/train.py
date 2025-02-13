from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from sklearn import metrics
from nn.nn_utils import mkdirs
from nn.test_classifier import find_best_thr, run_net_with_true_lebels
from nn.test_experts import run_net4experts_prediction
from typing import Dict, List

def weighted_custom_loss(data, alv_name_std_num_dict, device, outputs, labels):
    w = []
    for alv_name in data["name"]:
        w.append(alv_name_std_num_dict[alv_name]) 
    w = torch.FloatTensor(w).to(device)
    loss = torch.mean(w * (torch.squeeze(outputs).float() - torch.squeeze(labels).float()) ** 2)
    return loss


def train(device, 
          net, 
          trainloader,
          loss_fun, 
          optimizer, 
          lr_scheduler,
          epochs,
          logdir,
          valloader=None,
          trained_ep_num: int=0,
          choose_thr: bool=True,
          use_label_smoothing: bool=False,
          experts_mode: str=None,
          metalearning_mode: bool=False,
          experts_markup: Dict=None,
          step2check_val_acc: int=1000,
          thresholds: Dict=None,
          ):

    len_trainloader = len(trainloader.dataset)
    if not logdir.exists():
        mkdirs(logdir)
    writer = SummaryWriter(logdir=logdir)

    for epoch in tqdm(range(epochs)): 
        running_loss = 0.0
        
        net.train(True)

        for data in trainloader:
            inputs = data['image'].to(device).float()

            labels = data['agg_type'] if experts_mode is None \
                else data[f'experts_{experts_mode}']
            labels = labels.to(device)

            if use_label_smoothing:
                for alv_id in range(len(labels)):
                    # if np.random.random() > 0.3:
                    # epsilon = np.log10(5 * experts_std + 1) / 2.2
                        # epsilon = 0.5 - abs(0.5 - np.mean(experts_markup[data["name"][alv_id]]["expert_votes_positions"]) / 4)
                        epsilon = 0.1
                        if labels[alv_id] == 0: 
                            labels[alv_id] = epsilon
                        else:
                            labels[alv_id] = 1.0 - epsilon
            optimizer.zero_grad()
            
            if metalearning_mode:
                meta_reagent = data['meta_reagent_type'].to(device)

            if metalearning_mode:
                if experts_mode is not None and experts_mode != "average_vote":
                    outputs = net(inputs, meta_reagent) / 2
                else:
                    outputs = net(inputs, meta_reagent)
            else:
                outputs = net(inputs)


            if isinstance(loss_fun, torch.nn.CrossEntropyLoss):
                loss = loss_fun(torch.squeeze(outputs).float(), torch.squeeze(labels))
            else:
                loss = loss_fun(torch.squeeze(outputs).float(), torch.squeeze(labels).float())

            running_loss += loss.item()

            loss.backward()

            optimizer.step()
        
        lr_scheduler.step()

        writer.add_scalar('Training loss', running_loss / len_trainloader, epoch + trained_ep_num)
        
        if valloader is not None and (epoch % step2check_val_acc == 0):
            if experts_mode is None:
                if choose_thr:
                    thr, acc = find_best_thr(device=device, net=net, valloader=valloader, print_thr=False,
                                            metalearning_mode=metalearning_mode)
                    thresholds[epoch + trained_ep_num] = str(thr)
                else:
                    thr = 0.5
                    y_pred, y_true = run_net_with_true_lebels(device, net, valloader, thr, 
                                                              metalearning_mode=metalearning_mode)                
                    acc = metrics.accuracy_score(y_true, y_pred)
            else:
                acc = run_net4experts_prediction(device, net, valloader, metalearning_mode, 
                                                 experts_mode, experts_markup, epsilon=0.0625)
            if experts_mode == "class":
                for i in range(5):
                    writer.add_scalar(f"Class {i} accuracy on validation", acc[i], epoch + trained_ep_num)
            else:
                writer.add_scalar('Validation accuracy', acc, epoch + trained_ep_num)
    
    writer.close()

    return net, thresholds 