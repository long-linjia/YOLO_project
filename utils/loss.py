import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # NOTE: This assumes predictions are NOT yet processed with sigmoid or softmax
        # This is a simplified version — for full YOLOv4 support, you'd need anchor handling

        # Dummy unpacking — modify this for real output format
        # predictions shape: [B, S*S, 5 + C]
        # targets: list of [N_i, 5] (class, x, y, w, h)

        total_loss = 0.0
        for batch_idx, target in enumerate(targets):
            pred = predictions[batch_idx]  # shape: [S*S, 5+C]
            obj_mask = target.shape[0] > 0

            if not obj_mask:
                total_loss += self.bce(pred[:, 4], torch.zeros_like(pred[:, 4]))
                continue

            # Extract true box center and size
            true_classes = target[:, 0].long()
            true_boxes = target[:, 1:]  # [x, y, w, h]

            # For simplicity: assume each target maps to a specific anchor/grid
            pred_boxes = pred[:target.shape[0], :4]
            pred_obj = pred[:target.shape[0], 4]
            pred_cls = pred[:target.shape[0], 5:]

            box_loss = self.mse(pred_boxes, true_boxes)
            obj_loss = self.bce(pred_obj, torch.ones_like(pred_obj))
            cls_loss = self.ce(pred_cls, true_classes)

            # No-object loss
            no_obj_loss = self.bce(pred[:, 4], torch.zeros_like(pred[:, 4]))

            total = (
                self.lambda_coord * box_loss +
                obj_loss +
                self.lambda_noobj * no_obj_loss +
                cls_loss
            )

            total_loss += total

        return total_loss / len(targets)