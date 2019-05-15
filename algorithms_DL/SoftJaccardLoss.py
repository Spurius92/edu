class SoftJaccardLoss(_Loss):

   def __init__(self, num_classes, from_logits=True, weight=None, reduction='elementwise_mean'):

       super(SoftJaccardLoss, self).__init__(reduction=reduction)

       self.from_logits = from_logits

       self.weight = weight if weight is not None else np.ones(num_classes)



   def forward(self, y_pred: Tensor, y_true: Tensor):

       """



       :param y_pred: NxCxHxW

       :param y_true: NxHxW

       :return: scalar

       """

       if self.from_logits:

           y_pred = y_pred.softmax(dim=1)



       n_classes = y_pred.size(1)

       smooth = 1e-3



       loss = torch.zeros(n_classes, dtype=torch.float, device=y_pred.device)



       for class_index, class_weight in enumerate(self.weight):

           jaccard_target = (y_true == class_index)

           jaccard_output = y_pred[:, class_index, ...]



           num_preds = jaccard_target.long().sum()



           if num_preds == 0:

               loss[class_index] = 0

           else:

               jaccard_target = jaccard_target.float()

               intersection = (jaccard_output * jaccard_target).sum()

               union = jaccard_output.sum() + jaccard_target.sum()

               iou = intersection / (union - intersection + smooth)

               loss[class_index] = (1.0 - iou) * class_weight



       if self.reduction == 'elementwise_mean':

           return loss.mean()



       if self.reduction == 'sum':

           return loss.sum()



       return loss
