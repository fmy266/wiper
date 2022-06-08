import torch
import torch.nn as nn
import sys
import os
sys.path.append("..")
import utils
import backdoor_attack

def save_model(info):
    def val_func(model, trigger, clean_acc, poison_acc, device):
        checkpoint = {
            "model":model.state_dict(),
            "trigger":trigger,
            "clean_acc":clean_acc,
            "poison_acc":poison_acc,
        }
        torch.save(checkpoint, os.path.join(os.getcwd(), "saved_models", info+".pth"))
    return val_func


if __name__ == "__main__":
    dataset_name = "cifar10"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    train_loader, test_loader = utils.DataManger.get_dataloader(dataset_name, root = "../data")
    poison_label = 0
    model_name = "resnet18"
    model = utils.ModelManger.get_model(dataset_name, model_name, device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    poison_prob = 0.05
    trigger_params = {"size":[3,3,3], "poison_label":poison_label, "random_pos":False, "dataset_name":dataset_name,\
                "normalize_param" : utils.DataManger.get_normalize_params(dataset_name),\
                "trigger" : None, "random_init" : True, "trigger_path" : None}
    attack = backdoor_attack.poison_attack.BadNet(model = model, dataloader = train_loader, loss_func = nn.CrossEntropyLoss(),\
                                                   optim = optimizer, poison_prob = poison_prob, device = device,\
                                                   trigger_params = trigger_params, scheduler = scheduler, val_loader = test_loader)
    attack.train(epoch = 10,
        valid_func = save_model("{dataset_name}_{model_name}_{attack_method}".format(
                                 dataset_name = dataset_name, model_name = model_name, attack_method = attack.name)))
