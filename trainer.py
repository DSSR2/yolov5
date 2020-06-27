import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
import os
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import *
from utils.utils import *
import glob
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    #print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters
def get_hyp():
    return {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 5e-4,  # optimizer weight decay
            'giou': 0.05,  # giou loss gain
            'cls': 0.58,  # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'iou_t': 0.20,  # iou training threshold
            'anchor_t': 4.0,  # anchor-multiple threshold
            'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
            'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
            'degrees': 0.0,  # image rotation (+/- deg)
            'translate': 0.0,  # image translation (+/- fraction)
            'scale': 0.5,  # image scale (+/- gain)
            'shear': 0.0}  # image shear (+/- deg)

def get_cfg(cfg, num_classes):
    fname = ".\\models\\yolov5s.yaml"
    if(cfg == "Y0"):
        fname = ".\\models\\yolov5s.yaml"
    elif(cfg == "Y1"):
        fname = ".\\models\\yolov5m.yaml"
    elif(cfg == "Y2"):
        fname = ".\\models\\yolov5l.yaml"
    elif(cfg == "Y3"):
        fname = ".\\models\\yolov5x.yaml"
    else:
        print(cfg, " not found, loading Y0")
        fname = ".\\models\\yolov5s.yaml"

    with open(fname) as f:
        list_doc = yaml.load(f, Loader=yaml.FullLoader)
    list_doc["nc"] = num_classes
    with open(fname, "w") as f:
            yaml.dump(list_doc, f)

    return fname
    

class object_detector():
    def __init__(self, model='Y0', adam=False, batch_size=16, bucket='', cache_images=False, data='.\\data\\My_data.yaml', device='', epochs=3, classes=["Defect"], evolve=False, img_size=[640, 640], multi_scale=False, name='', nosave=False, notest=False, rect=False, resume=False, single_cls=False, weights='', flag=0):
        self.adam = adam
        self.batch_size = batch_size
        self.bucket = bucket
        self.cache_images = cache_images
        self.classes = classes
        self.num_classes = len(self.classes)
        self.cfg = get_cfg(model, self.num_classes)
        self.data = data
        self.device = device
        self.epochs = epochs
        self.evolve = evolve
        self.img_size = img_size
        self.multi_scale = multi_scale
        self.name = name
        self.nosave = nosave
        self.notest = notest
        self.rect = rect
        self.resume = resume
        self.single_cls = single_cls
        self.weights = weights

        self.weights = last if self.resume else self.weights
        self.cfg = glob.glob('./**/' + self.cfg, recursive=True)[0]  # find file
        self.data = glob.glob('./**/' + self.data, recursive=True)[0]  # find file
        self.img_size.extend([self.img_size[-1]] * (2 - len(self.img_size)))  # extend to 2 sizes (train, test)
        self.device = torch_utils.select_device(self.device, apex=False, batch_size=self.batch_size)
        # check_git_status()
        if self.device.type == 'cpu':
            mixed_precision = False

        # Train
        
    def load_data(self, path="../Dataset", save_name="My_data.yaml", classes=["Defect"], validation_split=0.2, test_path=" "):
        #with open("./data/My_data.yaml") as f:
        #    list_doc = yaml.load(f, Loader=yaml.FullLoader)

        list_doc = {}
        if(os.path.exists(path+"/Train/") and not os.path.exists(path+"/Validation/")):
            print("No validation, making validation folder")
            os.makedirs(path+"/Validation/")
            os.makedirs(path+"/Validation/Images/")
            os.makedirs(path+"/Validation/Labels/")

            img_files = glob.glob(path+"/Train/Images/*.jpg") + glob.glob(path+"/Train/Images/*.jpeg") + glob.glob(path+"/Train/Images/*.png") + glob.glob(path+"/Train/Images/*.bmp")
            print(len(img_files), " Images found")

            for i in img_files:
                fname = i.split("\\")[-1].split(".")[0]
                if(not os.path.exists(path+"/Train/Labels/"+fname+".txt")):
                    f = open(path+"/Train/Labels/"+fname+".txt", 'w')
                    f.close() 
            txt_files = glob.glob(path+"/Train/Labels/*.txt")
            assert(len(img_files) == len(txt_files))
            shuff = np.random.permutation(len(txt_files))
            shuff = shuff[:int(len(shuff)*validation_split)]
            valid_images = np.asarray(img_files)[shuff]
            valid_labs = np.asarray(txt_files)[shuff]
            print("Making Validation Data")
            for i in trange(len(valid_images)):
                fname = valid_images[i].split("\\")[-1].split(".")[0]
                shutil.move(valid_images[i], path+"/Validation/Images/")
                shutil.move(valid_labs[i], path+"/Validation/Labels/")
            print("Validation Data Created!")

        list_doc["nc"] = len(classes)
        list_doc["names"] = classes
        list_doc["train"] = path + "/Train/Images/"
        list_doc["val"] = path + "/Validation/Images/"
        list_doc["test"] = test_path

        print(list_doc)
        with open("./data/"+save_name, "w") as f:
            yaml.dump(list_doc, f)

    def train(self, epochs):
        #epochs = self.epochs  # 300
        hyp=get_hyp()
        batch_size = self.batch_size  # 64
        weights = self.weights  # initial training weights

        # Configure
        init_seeds(42)
        with open(self.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        train_path = data_dict['train']
        test_path = data_dict['val']
        nc = 1 if self.single_cls else int(data_dict['nc'])  # number of classes

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

        # Create model
        model = Model(self.cfg).to(self.device)
        assert model.md['nc'] == nc, '%s nc=%g classes but %s nc=%g classes' % (self.data, nc, self.cfg, model.md['nc'])

        # Image sizes
        gs = int(max(model.stride))  # grid size (max stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in self.img_size]  # verify imgsz are gs-multiples

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        optimizer = optim.Adam(pg0, lr=hyp['lr0']) if self.adam else \
            optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        start_epoch, best_fitness = 0, 0.0
        if weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(weights, map_location=device)  # load checkpoint

            # load model
            try:
                ckpt['model'] = \
                    {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(ckpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
                    % (self.weights, self.cfg, self.weights)
                raise KeyError(s) from e

            # load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            start_epoch = ckpt['epoch'] + 1
            del ckpt

        # # Mixed precision training https://github.com/NVIDIA/apex
        # if mixed_precision:
        #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1  # do not move

        # Dataset
        dataset = LoadImagesAndLabels(train_path, imgsz, batch_size,
                                    augment=True,
                                    hyp=hyp,  # augmentation hyperparameters
                                    rect=self.rect,  # rectangular training
                                    cache_images=self.cache_images,
                                    single_cls=self.single_cls)
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, self.cfg)

        # Dataloader
        batch_size = min(batch_size, len(dataset))
        nw = 0#min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=nw,
                                                shuffle=not self.rect,  # Shuffle=True unless rectangular training is used
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)

        # Testloader
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                    hyp=hyp,
                                                                    rect=True,
                                                                    cache_images=self.cache_images,
                                                                    single_cls=self.single_cls),
                                                batch_size=batch_size,
                                                num_workers=nw,
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)

        # Model parameters
        hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device)  # attach class weights
        model.names = data_dict['names']

        # class frequency
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels)

        # Exponential moving average
        ema = torch_utils.ModelEMA(model)

        # Start training
        t0 = time.time()
        nb = len(dataloader)  # number of batches
        n_burn = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % nw)
        print('Starting training for %g epochs...' % epochs)
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train(True)

            # Update image weights (optional)
            if dataset.image_weights:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

            mloss = torch.zeros(4, device=self.device)  # mean losses
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

                # Burn-in
                if ni <= n_burn:
                    xi = [0, n_burn]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

                # Multi-scale
                if self.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(self.device), model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Backward
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = 'train_batch%g.jpg' % i  # filename
                    res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)

                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            scheduler.step()

            # mAP
            ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not self.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(self.data,
                                                batch_size=batch_size,
                                                imgsz=imgsz_test,
                                                save_json=final_epoch and self.data.endswith(os.sep + 'coco.yaml'),
                                                model=ema.ema,
                                                single_cls=self.single_cls,
                                                dataloader=testloader,
                                                fast=ni < n_burn)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(self.name) and self.bucket:
                os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (self.bucket, self.name))

            # Tensorboard

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not self.nosave) or (final_epoch and not self.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)
                del ckpt

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        n = self.name
        if len(n):
            n = '_' + n if not n.isnumeric() else n
            fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
            for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    strip_optimizer(f2) if ispt else None  # strip optimizer
                    os.system('gsutil cp %s gs://%s/weights' % (f2, self.bucket)) if self.bucket and ispt else None  # upload

        if not self.evolve:
            plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results
