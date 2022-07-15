import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics,  AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
from datetime import datetime
from torchmetrics.functional import precision_recall
import math
from torchvision.utils import save_image
from random import randrange

class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,num_classes,
                val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger)
        
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader

        self.ignore_index = config['ignore_index']
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = num_classes #self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize([-0.3074, -0.1277, -0.0625], [ 0.8775, 0.8860, 0.8869]), #select correct mean/std for B2-B3-B4
            transforms.ToPILImage()])
        self.restore_transform_L4S = transforms.Compose([transforms.ToPILImage()])
        #self.viz_transform = transforms.Compose([
        #    transforms.Resize((128, 128)), # correct resize 
        #    transforms.ToTensor()])
        self.viz_transform = transforms.Compose([ 
            transforms.ToTensor()])

        self.start_time = time.time()
        self.start_datetime = datetime.now()



    def _train_epoch(self, epoch):
        self.html_results.save()
        
        self.logger.info('\n')
        self.model.train()

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=135)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.mode == 'supervised':
                #print(next(dataloader))
                (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
            else:
                (input_l, target_l), (input_ul, target_ul) = next(dataloader)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)

            #print('input_l.shape 1,', input_l.shape) # ([4, 2000, 2000, 4]) # L4Storch.Size([4, 11, 128, 128])
            #input_l = input_l.permute(0,3,1,2) # change order of Tensor
            #print(input_l.size())
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            self.optimizer.zero_grad() ### deprecation warning 
            #self.optimizer.autograd.grad() # added this
            #print('input_l.shape,', input_l.shape) # input_l.shape, torch.Size([2, 4, 2000, 2000]) # torch.Size([8, 11, 128, 128]) # batch size, num channels, image dims
            #print(target_l.dtype) # torch.int32
            target_l = target_l.clone().detach() ## added this to change  dtype to long
            target_l = target_l.type(torch.long)
            #print(target_l.dtype) # torch.int64
            #print(target_ul.dtype) #torch.float64
            if target_ul is not None:
                target_ul = target_ul.clone().detach() ## added this to change  dtype to long
                target_ul = target_ul.type(torch.long)
            #print(target_ul.dtype)
            #print('target_l.shape, ',target_l.shape) # torch.Size([8, 128, 128]), torch.Size([4, 640, 640])
            #print('target_ul, ', target_ul.shape) # AttributeError: 'NoneType' object has no attribute 'shape'/target_ul,  torch.Size([8, 128, 128])
            
            #print('input_l',input_l.shape) # torch.Size([8, 11, 128, 128])
            #print('input_ul', input_ul.shape)
            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                        curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1)
            #print('output shapes',outputs) # {'sup_pred': tensor([[[[0.3230, 0.0000, 0.3230,  ..., 0.0000, 0.3230, 0.0000],
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()
            
            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch-1)
            logs = self._log_values(cur_losses)
            
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                self._write_scalars_tb(logs)

            if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs
            
            elapsed_time = datetime.now() - self.start_datetime
            
            
            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f} time {}|'.format(
                epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_weakly.average,
                self.pair_wise.average, self.mIoU_l, self.mIoU_ul,elapsed_time))

            self.lr_scheduler.step(epoch=epoch-1)

        return logs



    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        all_precisions, all_recalls = [], []
        all_f1 = []

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                target = target.clone().detach() ## added this to change dtype to long
                target = target.type(torch.long)
                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data)
                output = output[:, :, :H, :W]


                img1 = output[0] #torch.Size(
                print(img1.shape)
                save_image(img1, '/geomatics/gpuserver-0/scuypers/CCT_output_L4S/trainer/img{}.png'.format(randrange(200)))

                # LOSS
                #print(output.shape) # torch.Size([1, 2, 128, 128]) # 2 output classes
                #print(target.shape) # torch.Size([1, 128, 128])
                #target = torch.tensor(target, dtype=torch.long) # Maybe?
                #target.type(torch.LongTensor) # this doesn't make a diff
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

               
                ## with ignore_index = 0 ? Does this mean precision and recall only for the 1 class?
                precision, recall = precision_recall(output, target, average='macro', mdmc_average='global', ignore_index=0, num_classes=self.num_classes, threshold=0.5, top_k=None, multiclass=None)
                precision, recall = np.round(precision.item(),5), np.round(recall.item(),5)

                if precision + recall == 0: f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                    f1 = np.round(f1,6)

                all_precisions.append(precision)
                avg_precision = sum(all_precisions)/len(all_precisions)
                all_recalls.append(recall)
                avg_recall = sum(all_recalls)/len(all_recalls)
                all_f1.append(f1)
                avg_f1 = sum(all_f1)/len(all_f1)

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                
                total_inter, total_union = total_inter+inter, total_union+union
                total_correct, total_label = total_correct+correct, total_label+labeled

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    if isinstance(data, list): data = data[0]
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Precision": np.round(avg_precision,3),"Recall": np.round(avg_recall,3),"f1":np.round(avg_f1,3),
                                "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}|'.format( epoch,
                                                total_loss_val.average, pixAcc, mIoU, avg_precision,avg_recall,avg_f1))

            self._add_img_tb(val_visual, 'val')

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            self.html_results.add_results(epoch=epoch, seg_results=log)
            self.html_results.save()

            self._save_checkpoint(epoch, save_best=self.improved) # added this
            #if (time.time() - self.start_time) / 3600 > 22:
            #    self._save_checkpoint(epoch, save_best=self.improved)u
        return log



    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup  = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}



    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_weakly" in cur_losses.keys():
            self.loss_weakly.update(cur_losses['loss_weakly'].mean().item())
        if "pair_wise" in cur_losses.keys():
            self.pair_wise.update(cur_losses['pair_wise'].mean().item())



    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if self.mode == 'semi':
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()
            


    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union



    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }



    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs


    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)



    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            ## divide by 7 (= max value)
            imgs = [self.restore_transform_L4S(i[0:3,:,:]/4) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                        else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)



    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')

