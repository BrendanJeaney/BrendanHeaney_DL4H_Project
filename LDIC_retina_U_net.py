import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou
from LIDC_retinaHead import retHead
from LIDCDataLoader import LIDCDataLoader
from LIDC_UNet import UnNet, train_unet, eval


class retinaUnNet(nn.Module):
    def __init__(self, unnet, ret_head):
        super().__init__()
        self.unet = unnet
        self.retina = retHead(in_channels=16)


    def forward(self, x):
        segmentation, feat = self.unet(x,return_feat=True)
        b_preds, preds = self.retina(feat)

        return segmentation, b_preds, preds



def train_retunet(model, dataset, epochs=5, batch_size=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    seg_criterion = nn.BCELoss()
    bbox_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, s_masks, b_targets, cls_targets in dataloader:
            # imgs = imgs.to(device)
            # seg_masks = seg_masks.to(device)
            # box_targets = box_targets.to(device).float()
            # cls_targets = cls_targets.to(device).float()
            
            optimizer.zero_grad()

            segmentation, b_preds, preds = model(imgs)
            seg_loss = seg_criterion(segmentation, s_masks)
            b_loss = bbox_criterion(b_preds, b_targets)
            cls_loss = cls_criterion(preds, cls_targets)

            loss = seg_loss + b_loss + cls_loss
            # print(loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")




def get_box(mask):
    p = np.where(mask > 0.3)
    # print(p[0])
    # print(len(p))
    if len(p[0]) != 0:
        p0 = p[0]
        p1 = p[1]
        min_y = p0.min()
        max_y = p0.max()
        min_x = p1.min()
        max_x = p1.max()

        return [min_y, max_y, min_x, max_x]

    return None



def eval_retUnet(model, dataset, iou_threshold=0.1):
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for img, gt_mask in dataset:
            # print(img)
            pred_mask = model(img.unsqueeze(0)).squeeze().cpu().numpy()
            pred_box = get_box(pred_mask)
            gt_box = get_box(gt_mask.squeeze().cpu().numpy())

            # print(f"Pred mask max: {pred_mask.max():.2f}, min: {pred_mask.min():.2f}")

            if gt_box is None and pred_box is None:
                # print(f'gtbox {gt_box}')
                continue  

            if gt_box is not None and pred_box is not None:
                # iou = box_iou(
                #     torch.tensor([pred_box], dtype=torch.float32),
                #     torch.tensor([gt_box], dtype=torch.float32)
                # ).item()
                iou = get_iou(pred_box, gt_box)
                print(f"IoU: {iou:.3f}")
                if iou >= iou_threshold:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif gt_box is not None:
                fn += 1
            elif pred_box is not None:
                fp += 1

    # print(tp)
    # print(f'{tp+fp}')
    epsilon=1e-6
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    # f1 = (2*precision*recall)/(precision+recall)
    # print(f'f1 score: {f1}')
    print(f"IoU ≥ {iou_threshold:.2f} \n precision: {precision:.3f},\n recall: {recall:.3f},\n mAp: {precision:.3f}")




def evaluate_retina_unet(model, dataset, iou_threshold=0.1):
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for img, gt_mask in dataset:
            pred_mask, cls_logit, pred_box = model(img.unsqueeze(0))
            pred_mask = pred_mask.squeeze().numpy()
            pred_box = pred_box.squeeze()
            gt_box = get_box(gt_mask.squeeze().numpy())

            if gt_box is None and (pred_mask > 0.5).sum() == 0:
                continue
            elif gt_box is not None and (pred_mask > 0.5).sum() == 0:
                fn += 1
            elif gt_box is None and (pred_mask > 0.5).sum() > 0:
                fp += 1
            else:
                iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                if iou >= iou_threshold:
                    tp += 1
                else:
                    fp += 1
                    fn += 1

    # print(tp)
    # print(f'{tp+fp}')
    epsilon=1e-6
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    # f1 = (2*precision*recall)/(precision+recall)
    # print(f'f1 score: {f1}')
    print(f"IoU ≥ {iou_threshold:.2f} \n precision: {precision:.3f},\n recall: {recall:.3f},\n mAp: {precision:.3f}")





def get_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    boxA_area = max(0, pred_box[2] - gt_box[0]) * max(0, pred_box[3] - pred_box[1])
    boxB_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])

    union = boxA_area + boxB_area - inter_area

    if union > 0:
        return inter_area/union 
    else:
        0.0