import torch

def Optimizer(model,args,cfg=None):
    try:
        WEIGHT_DECAY = cfg.OPTIMIZER.WEIGHT_DECAY
    except AttributeError:
        WEIGHT_DECAY = 0
    
    try:
        if args.model == 'IMDN':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE)
        elif args.model == "FDADN":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE)
        elif args.model == 'FDAN':
            bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
            others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
            arameters = [{'params': others_list}, {'params': bias_list, 'weight_decay': 0, 'lr': cfg.OPTIMIZER.LEARNING_RATE / 10}]
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    except AttributeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer