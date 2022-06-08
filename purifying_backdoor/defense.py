import torch
import torch.nn as nn
import sys
import os
sys.path.append("..")
import utils
import backdoor_attack


if __name__ == "__main__":
    dataset_name = "cifar10"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    poison_label = 0
    model_name = "resnet18"
    purifying_layers = ["linear"]
    sampling_rate = 0.05
    alpha = 0.01
    regular_way = "adaptive"
    eta = 0.9
    purifying_strategy = "dynamic"
    beta = 0.5
    epoch = 10
    model = utils.ModelManger.get_model(dataset_name, model_name, device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
    scheduler = None
    poison_prob = 0.05
    trigger_params = {"size":[3,3,3], "poison_label":poison_label, "random_pos":False, "dataset_name":dataset_name,\
                "normalize_param" : utils.DataManger.get_normalize_params(dataset_name),\
                "trigger" : None, "random_init" : True, "trigger_path" : None}
    attack = backdoor_attack.poison_attack.BadNet(model = model, dataloader = train_loader, loss_func = nn.CrossEntropyLoss(),\
                                                   optim = optimizer, poison_prob = poison_prob, device = device,\
                                                   trigger_params = trigger_params, scheduler = scheduler, val_loader = test_loader)
    path = os.path.join(os.getcwd(), "saved_models", "{dataset_name}_{model_name}_{attack_name}.pth".format(
                                 dataset_name = dataset_name, model_name = model_name, attack_name = attack.name))
    attack.load(path)

    sm_defense = backdoor_attack.PurifyingBackdoor(model = model, train_loader = train_loader, test_loader = test_loader,
                                                  purifying_layers = purifying_layers, alpha = alpha, beta = beta, eta = eta, device = device)
    sm_defense.defense(epoch, regular_way, purifying_strategy)
    clean_acc, poison_acc = attack.validate()

