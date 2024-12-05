import warnings
from val import run, parse_opt
warnings.filterwarnings("ignore")
import os
import logging
import random
import numpy as np
from Trainer import Trainer
from config.option import Base_options
from data.unaligned_dataset import UnalignedDataset
from publicutil.comutils import get_yaml_data, set_yaml_to_args
import torch
import time
from publicutil.visualizer import Visualizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(20)

def main_worker():
    # load yaml file by mode
    base_option = Base_options()
    args = base_option.get_args()
    yamls_dict = get_yaml_data('./config/TM.yaml')
    set_yaml_to_args(args, yamls_dict)
    if not os.path.exists('./checkSave/'):
        os.makedirs('./checkSave/')
    logging.basicConfig(filename=args.log_path + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    dataset = UnalignedDataset(args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=int(args.nThreads),
        drop_last=True,
    )
    trainer = Trainer(args)
    visualizer = Visualizer(args)
    dataset_size = len(data_loader)
    total_steps = 0
    best_map = 0
    last_map = 0
    for epoch in range(1, args.niter + args.niter_decay + 1):
        for i, data in enumerate(data_loader):
            # break
            iter_start_time = time.time()
            trainer.set_input(data)
            trainer.train()
            total_steps += 1
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            if total_steps % args.print_freq == 0:
                errors = trainer.TModel_trainer.get_current_errors(epoch)
                errors['map'] = last_map
                t = (time.time() - iter_start_time) / args.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if args.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, args, errors)

            if total_steps % args.display_freq == 0:
                visualizer.display_current_results(trainer.TModel_trainer.get_current_visuals(), epoch)

        print('saving the latest model (epoch %d, total_steps %d)' %
              (epoch, total_steps))
        trainer.TModel_trainer.save(epoch, args)

        # ------------------
        if epoch % 1 == 0:
            opt = parse_opt()
            opt.VaT = trainer.TModel_trainer.TModel_A
            opt.plots = False
            # opt.task = "val"
            mAP = run(**vars(opt))[0][2]
            print('mAP:%g' % (mAP))
            print('best mAP:%g' % (best_map))
            if mAP > best_map:
                best_map = mAP
                trainer.TModel_trainer.save('best', args)


if __name__ == '__main__':
    main_worker()
