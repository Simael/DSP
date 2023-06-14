import torch
from torch import nn
from torch.nn.functional import one_hot
import copy

"""
DSP output head, i.e. prototypes and the computation of similarity between pixel-embeddings and prototypes.
"""
class dsp_outputhead(nn.Module):
    def __init__(self, in_channels, num_classes, num_prototypes_per_class, temperature):
        super().__init__()
        # Temperature hyperparameter
        self.temperature = temperature

        # Non-linear projection of pixelwise features
        self.pixelwise_transform = nn.Sequential(nn.BatchNorm2d(in_channels),
                                                nn.Conv2d(in_channels, in_channels, 1),
                                                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                                                nn.Conv2d(in_channels, in_channels, 1))

        # Setup random prototypes
        self.prototypes = torch.randn((num_classes, num_prototypes_per_class, in_channels))
        # Calculate norm
        norm = self.prototypes.pow(2).sum(2, keepdim=True).pow(1. / 2.)
        # Initialize prototypes as learnable parameters
        self.prototypes = torch.nn.Parameter(self.prototypes.div(norm), requires_grad=True)

    """
    Here the tensor x is a batch of pixel-wise embeddings of shape B x D x H x W, i.e. a D dimensional feature vector for each pixel as produced by a segmentation backbone
    """
    def forward(self, x):
            # Project the pixel-wise features, normalize the output to obtain pixel-wise embeddings
            proj = self.pixelwise_transform(x)
            norm = proj.pow(2).sum(1, keepdim=True).pow(1. / 2.)
            embeddings = proj.div(norm)

            # Normalize prototypes
            prototype_norm = self.prototypes.pow(2).sum(2, keepdim=True).pow(1. / 2.)

            # Calculate the similarity between each pixel-embedding and each prototype, divide by temperature
            similarities = torch.mm(embeddings.permute(1, 0, 2, 3).reshape(embeddings.shape[1], -1).t(),
                                    self.prototypes.div(prototype_norm).permute(2, 0, 1).reshape(self.prototypes.shape[2], -1)) / self.temperature

            # Average the similarities for prototypes representing the same class
            logits = torch.nn.functional.avg_pool1d(similarities, kernel_size=self.prototypes.shape[1],
                                                    stride=self.prototypes.shape[1])

            # Reshape the logits into format B x C x H x W
            logits = logits.reshape(embeddings.shape[0], embeddings.shape[-2], embeddings.shape[-1],
                                    self.prototypes.shape[0]).permute(0, 3, 1, 2)

            return embeddings, logits

"""
Make sure coordinates are sorted from small to large
"""
def sort_coordinates(x, x2):
    if x <= x2:
        return x, x2
    else:
        return x2, x

"""
The DSP loss is calculated based on the output logits of the network as well as the diverse annotations present in the batch.

logits: B x C x H x W
class_mask: B x H x W, index of respective class at pixel locations
image_level_labels: [], list of B tensors which is made of the indices of the present classes in the batch item
boundingboxes: [N x 6], list with B tensors of shape N x 6, where N is the number of boxes for the batch item and the 6 dimensions refer to box information instance_id, class_index, x, y, x2, y2
points: [N x 4], list with B tensors of shape N x 4, where N is the number of points for the batch item and the 4 dimensions refer to point information instance_id, class_index, x, y
connected_components: B x H x W, tensor which includes unique indices for each pixel of the same connected component, connected components are enumerated
valid_regions: B x H x W, binary tensor which can be used to ignore certain regions in the image, e.g. not annotated regions, if the whole image should be considered pass a tensor with ones
ignore index: index indicating the ignore class label

If there is no mask present for a certain image, initialize the whole mask with the ignore index.
If there is no image-level label, bounding box or point for a batch item, make sure to add a None element for that batch item at the respective position.
"""
def decoupled_semantic_prototypical_loss(logits, class_mask, image_level_labels, boundingboxes, points,
                                         connected_components, valid_regions, ignore_index=255):
    batch_size, num_classes, height, width = logits.shape
    device = logits.device
    epsilon = 1e-7

    # Apply softmax to the (temporal scaled) logits
    preds = torch.nn.functional.softmax(logits, dim=1)

    valid_regions = valid_regions.float()

    # Initialize total losses for the different annotation-types
    image_loss = torch.tensor(0., device=logits.device)
    bounding_box_loss = torch.tensor(0., device=logits.device)
    point_loss = torch.tensor(0., device=logits.device)
    mask_loss = torch.tensor(0., device=logits.device)

    # Counters for how often the respective annotation-type occurs in the batch
    image_level_norm = 0.
    bounding_box_norm = 0.
    point_norm = 0.
    mask_norm = 0.

    # Go through each class, such that each class has been the positive class once
    for class_index in range(num_classes):
        # Initialize the masking tensor for the negatives (will be multiplied with preds later to aggregate the negatives in preds)
        global_negatives = torch.stack([valid_regions] * num_classes).permute(1,0,2,3)

        # Set associations to the positive class in the mask to 0 (will be adapted with annotation information)
        # All associations to other classes are seen as negatives
        global_negatives[:, class_index, :, :] = 0

        # Initialize the nominator for the contrastive term for each annotation-type
        image_level_nominator = torch.tensor(1., device=device)
        bounding_box_nominator = torch.tensor(1., device=device)
        point_nominator = torch.tensor(1., device=device)
        mask_nominator = torch.tensor(1., device=device)

        # Initialize counter variable for each annotation instance (e.g. number of bounding boxes for a single image)
        image_level_n = 0
        bounding_box_n = 0
        point_n = 0
        mask_n = 0

        # Go through each batch item, and check which annotation-type is available
        # to gather information about the positives (nominators) and negatives (denominator / global_negatives)
        for batch_index in range(batch_size):

            # Check if the image-level is available for the batch item
            if image_level_labels[batch_index] is not None:
                image_level_norm += 1
                if class_index in image_level_labels[batch_index]:
                    # Positive class is in image-level label: adjust positives with mean pooled associations to positive class
                    image_level_nominator += torch.log(
                        (preds[batch_index, class_index] * valid_regions[batch_index]).sum() / valid_regions[
                            batch_index].sum() + epsilon)

                    # Make sure to not use the class associations as negatives for this batch item
                    global_negatives[batch_index, class_index] = 0.

                    image_level_n += 1
                else:
                    # Positive class is not in image-level label: Add all positive class associations to the negatives
                    global_negatives[batch_index] = valid_regions[batch_index]

            # Check if the bounding boxes are available for the batch item
            if boundingboxes[batch_index] is not None:
                bounding_box_norm += 1

                # Variable to remember the locations of positive bounding boxes in terms of a binary mask
                positive_box_mask = torch.zeros_like(global_negatives[batch_index])

                # Go through all bounging boxes of the batch item
                for box_index in range(boundingboxes[batch_index].shape[0]):
                    cur_class = boundingboxes[batch_index][box_index, 1]
                    if class_index == cur_class:
                        # Positive class is associated to the bounding box
                        cur_x = boundingboxes[batch_index][box_index, 2]
                        cur_y = boundingboxes[batch_index][box_index, 3]
                        cur_x2 = boundingboxes[batch_index][box_index, 4]
                        cur_y2 = boundingboxes[batch_index][box_index, 5]

                        # Sanity check, that box coordinates are ordered correctly
                        cur_x, cur_x2 = sort_coordinates(cur_x, cur_x2)
                        cur_y, cur_y2 = sort_coordinates(cur_y, cur_y2)

                        k = max(1, max(cur_x2 - cur_x, cur_y2 - cur_y))

                        # Check that box is valid, i.e. not covering a point (sanity check)
                        if cur_x != cur_x2 and cur_y != cur_y2 and k > 1:
                            # Take all maximal associations along the vertical and horizontal dimension as positives (put them in the nominator)
                            bounding_box_nominator += torch.log(((1. / ((cur_x2 - cur_x) + (cur_y2 - cur_y))) * (
                                    preds[batch_index, class_index, cur_y:cur_y2, cur_x:cur_x2].max(dim=0)[0].sum() +
                                    preds[batch_index, class_index, cur_y:cur_y2, cur_x:cur_x2].max(dim=1)[0].sum())) + epsilon)

                            # Remember the location of the positive box (in the class_index dimension!), these regions have to be deleted from the global negatives
                            positive_box_mask[class_index, cur_y:cur_y2, cur_x:cur_x2] = 1
                            bounding_box_n += 1

                # Delete the regions of the positive bounding boxes from the negatives, as they contain positive pixels
                global_negatives[batch_index] = valid_regions[batch_index] - positive_box_mask

            # Check if point annotations are available for the batch item
            if points[batch_index] is not None:
                point_norm += 1

                # Go through all point annotations of the batch item
                for point_index in range(points[batch_index].shape[0]):
                    cur_class = points[batch_index][point_index, 1]
                    cur_x = points[batch_index][point_index, 2]
                    cur_y = points[batch_index][point_index, 3]
                    if cur_class == class_index:
                        # Positive class is associated to the point annotation, add it to the positives
                        point_nominator += torch.log(preds[batch_index, class_index, cur_x, cur_y] + epsilon)

                        # Set global negatives such that for the positive class, the point location is not used as negative
                        global_negatives[batch_index, :, cur_x, cur_y] = 1
                        global_negatives[batch_index, class_index, cur_x, cur_y] = 0

                        point_n += 1
                    else:
                        # Set point locations where the annotation is not the positive class to be used as negative
                        global_negatives[batch_index, :, cur_x, cur_y] = 1

            # Check if the mask annotation is available for the batch item
            if class_mask[batch_index].unique() != [ignore_index]:
                mask_norm += 1
                if class_index in class_mask[batch_index].unique():
                    # Positive class is associated to the mask annotation

                    # Initialize an empty mask where positive connected components will be registerd
                    positive_mask_mask = torch.zeros_like(global_negatives[batch_index])

                    # Via point annotation, successively select the connected components which are associated to the positive class
                    for point_index in range(points[batch_index].shape[0]):
                        cur_class = points[batch_index][point_index, 1]
                        cur_x = points[batch_index][point_index, 2]
                        cur_y = points[batch_index][point_index, 3]

                        # Select current connected component
                        cur_compnent_id = connected_components[batch_index, cur_x, cur_y]
                        cur_mask = connected_components[batch_index] == cur_compnent_id
                        if cur_class == class_index:
                            # Connected component is associated to the positive class

                            # Add the associations at the location of the connected component to the positives (nominator) via mean pooling
                            mask_nominator += torch.log(((preds[batch_index, class_index] * cur_mask).sum() / cur_mask.sum()) + epsilon)

                            # Save the connected component locations in this variable
                            positive_mask_mask[class_index] += cur_mask
                            mask_n += 1

                    # Delete the positive connected component regions from the negatives (denominator)
                    global_negatives[batch_index] = valid_regions[batch_index] - positive_mask_mask

        # Precalculate the denominator which is shared by all annotation-type losses
        denominator = torch.log((preds * global_negatives).sum() + epsilon)

        # Calculate the contrastive loss for all annotation-types (using their gathered positives, i.e. their nominators, as well as the shared denominator)
        if image_level_norm != 0 and image_level_n != 0:
            image_loss += (1. / image_level_norm) * (denominator - image_level_nominator / image_level_n)

        if bounding_box_norm != 0 and bounding_box_n != 0:
            bounding_box_loss += (1. / bounding_box_norm) * (denominator - bounding_box_nominator / bounding_box_n)

        if point_norm != 0 and point_n != 0:
            point_loss += (1. / point_norm) * (denominator - point_nominator / point_n)

        if mask_norm != 0 and mask_n != 0:
            mask_loss += (1. / mask_norm) * (denominator - mask_nominator / mask_n)

    # Return the individual annotation-type based losses
    return image_loss, bounding_box_loss, point_loss, mask_loss

"""
A function which filters a prediction with diverse associated annotations such as image-level labels, bounding boxes, point annotations, mask annotations and applies the softmax function. 

preds: B x C x H x W
class_mask: B x H x W, index of respective class at pixel locations
image_level_labels: [], list of B tensors which is made of the indices of the present classes in the batch item
boundingboxes: [N x 6], list with B tensors of shape N x 6, where N is the number of boxes for the batch item and the 6 dimensions refer to box information instance_id, class_index, x, y, x2, y2
points: [N x 4], list with B tensors of shape N x 4, where N is the number of points for the batch item and the 4 dimensions refer to point information instance_id, class_index, x, y
ignore index: index indicating the ignore class label

If there is no mask present for a certain image, initialize the whole mask with the ignore index.
If there is no image-level label, bounding box or point for a batch item, make sure to add a None element for that batch item at the respective position.
"""
def prediction_filtering_with_softmax(preds, class_mask=None, image_level_labels=None, bounding_boxes=None, points=None, ignore_index=255):
    # Initialize predictions to filter
    filtered_preds = copy.deepcopy(preds.detach())

    # Go through each batch item, and check which annotation-type is available
    # use the available information to filter the prediction
    for batch_index in range(filtered_preds.shape[0]):

        # Check if the image-level is available for the batch item
        if image_level_labels[batch_index] is not None:
            # Setup a filter mask for setting predictions not in coherence with the image-level label to negative infinity
            image_level_filter = one_hot(image_level_labels[batch_index], num_classes=filtered_preds.shape[1]).sum(dim=0).float()

            # Apply filter mask with the accumulated image-level information
            filtered_preds[batch_index][image_level_filter != 1, :, :] = -torch.inf

        # Check if bounding boxes are available for the batch item
        if bounding_boxes[batch_index] is not None:
            # Setup a filter mask for setting predictions of classes not falling into a bounding box of that class to negative infinity
            box_filter = torch.zeros_like(filtered_preds[batch_index])

            # Go through each box associated to the batch item and add its information to the filter mask
            for box_index in range(bounding_boxes[batch_index].shape[0]):
                cur_class = bounding_boxes[batch_index][box_index, 1]
                if cur_class != ignore_index:
                    cur_x = bounding_boxes[batch_index][box_index, 2]
                    cur_y = bounding_boxes[batch_index][box_index, 3]
                    cur_x2 = bounding_boxes[batch_index][box_index, 4]
                    cur_y2 = bounding_boxes[batch_index][box_index, 5]

                    cur_x, cur_x2 = sort_coordinates(cur_x, cur_x2)
                    cur_y, cur_y2 = sort_coordinates(cur_y, cur_y2)

                    # Update filter mask with current class and box
                    box_filter[cur_class, cur_y:cur_y2, cur_x:cur_x2] = 1

            # At regions where no boxes are found, keep the predictions
            box_filter[:, box_filter.sum(dim=0) == 0] = 1

            # Apply filter mask with the accumulated box information
            filtered_preds[batch_index][box_filter != 1] = -torch.inf

        # Check if the point annotation is available for the batch item
        if points[batch_index] is not None:
            # Go through each point associated to the batch item and set the prediction at the point location to the point class
            for point_index in range(points[batch_index].shape[0]):
                cur_class = points[batch_index][point_index, 1]
                if cur_class != ignore_index:
                    cur_x = points[batch_index][point_index, 2]
                    cur_y = points[batch_index][point_index, 3]

                    # Transform the point annotation into a pixel prediction
                    t = torch.ones((filtered_preds.shape[1]), device=filtered_preds.device) * -torch.inf
                    t[cur_class] = 1

                    # Enter the information of the point into the prediction
                    filtered_preds[batch_index, :, cur_x, cur_y] = t

    # Apply softmax, negative infinity values will be mapped to 0
    filtered_preds = torch.softmax(filtered_preds, dim=1)

    # Check if the mask annotation is available for the batch item
    if class_mask is not None:
        # Get regions where the mask annotation is present
        mask = (class_mask == ignore_index).unsqueeze(1)

        # Add the mask annotation as prediction into the predictions
        filtered_preds = filtered_preds * mask + ~mask * one_hot(class_mask * ~mask.squeeze(1),
                                                                 num_classes=filtered_preds.shape[1]).permute(0, -1, 1, 2)

    # Returns pseudo-label filtered, "softmaxed" predictions
    return filtered_preds


"""
Small dummy example for the usage of the different modules used in DSP.
"""
if __name__ == '__main__':
    # Batch data configuration
    batchsize, channels, height, width, num_classes = 5, 64, 512, 512, 11
    ignore_index = 255

    # Dummy labels for batch (mask, box, points, image-level label, unlabeled)
    # Setting up a mask with four connected components belonging to three classes
    class_mask = (torch.ones((batchsize, height, width)) * ignore_index).long()
    class_mask[0] = 0
    class_mask[0, 100:200, 100:200] = 2
    class_mask[0, 400:500, 400:500] = 2
    class_mask[0, 50:80, 50:80] = 6
    class_mask[0, 60:70, 40:90] = 6

    connected_components = torch.zeros((batchsize, height, width), dtype=torch.long)
    connected_components[0, 100:200, 100:200] = 1
    connected_components[0, 400:500, 400:500] = 2
    connected_components[0, 50:80, 50:80] = 3
    connected_components[0, 60:70, 40:90] = 3

    image_level_labels = [torch.tensor([0,2,6]), 
                          torch.tensor([1, 5, 8]), 
                          torch.tensor([4, 10]), 
                          torch.tensor([3, 7, 9, 10]), 
                          None]
    
    bounding_boxes = [torch.tensor([[0, 0, 0, 0, 512, 512], [1, 2, 100, 100, 200, 200], [2, 2, 400, 400, 500, 500], [3, 6, 50, 40, 80, 90]]), 
                      torch.tensor([[0, 1, 100, 150, 350, 370], [1, 5, 50, 30, 200, 300], [2, 8, 10, 400, 75, 500]]), 
                      None, 
                      None, 
                      None]
    
    points = [torch.tensor([[0, 0, 10, 10], [1, 2, 150, 150], [2, 2, 450, 450], [3, 6, 65, 65]]), 
              None, 
              torch.tensor([[0, 4, 280, 500], [1, 10, 55, 140]]), 
              None, 
              None]
    
    valid_regions = torch.ones((5, 512, 512,))

    # Dummy pixel-wise features as obtained by any segmentation network without pre output 1x1 convolution
    seg_features = torch.randn((batchsize, channels, height, width))

    # Setup DSP outputhead
    dsp_head = dsp_outputhead(in_channels=64, num_classes=11, num_prototypes_per_class=5, temperature=0.05)

    # Forward pass through DSP outputhead
    _, logits = dsp_head(seg_features)

    # Loss calculation for DSP loss with different annotation-types
    image_loss, bounding_box_loss, point_loss, mask_loss = decoupled_semantic_prototypical_loss(logits, class_mask, image_level_labels, bounding_boxes, points, connected_components, valid_regions)

    # Standard pixel-wise cross-entropy loss using the available masks
    cross_entropy_loss = torch.nn.functional.cross_entropy(logits, class_mask, ignore_index=ignore_index)

    # Filtering the networkpredictions with the available annotations
    _, filtered_pseudo_label = prediction_filtering_with_softmax(logits, class_mask, image_level_labels, bounding_boxes, points, ignore_index).max(dim=1)

    # Pixel-wise cross-entropy loss using the filtered pseudo-label
    pseudo_label_filter_loss = torch.nn.functional.cross_entropy(logits, filtered_pseudo_label, ignore_index=ignore_index)

    # Add up all loss parts
    total_loss = image_loss + bounding_box_loss + point_loss + mask_loss + cross_entropy_loss + pseudo_label_filter_loss

    print("Total DSP loss:", total_loss)
