import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou


class UnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(16, 1, 1)



    def forward(self, x, ret_feat=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))

        d1 = self.dec1(torch.cat([self.up1(m), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))

        net_output = torch.sigmoid(self.out(d2))

        if ret_feat:
            return net_output, m
        
        return net_output 



def train_unet(model, dataset, epochs=5, batch_size=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, masks in dataloader:
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")




def get_box(mask):
    p = np.where(mask > 0.4)
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



def eval(model, dataset, iou_threshold=0.1):
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
                iou = box_iou(
                    torch.tensor([pred_box], dtype=torch.float32),
                    torch.tensor([gt_box], dtype=torch.float32)
                ).item()
                # iou = get_iou(pred_box, gt_box)
                # print(f"IoU: {iou:.3f}")
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