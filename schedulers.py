import torch
 
def Scheduler(optimizer,args,cfg=None):
    try:
        if args.model == 'IMDN':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA)
        elif args.model == '':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SCHEDULER.MILESTONES, gamma=cfg.SCHEDULER.GAMMA)
        elif args.model == 'FDADN':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA)
        elif args.model == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SCHEDULER.T_MAX)
        elif args.model == 'FDAN':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_mult=cfg.SCHEDULER.T_MULT, T_0=cfg.SCHEDULER.T_0)
    except AttributeError:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1)
    return scheduler