
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import re
import numpy as np
import os
import copy
from skimage.transform import resize
from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
import pandas as pd
import ast
import png

import time
import torch

from helpers import list_of_distances, make_one_hot


def main():
    test_image_dir = 'datasets/FETAL_PLANES_DB/test/Fetal_abdomen/'
    test_image_name = 'Patient00631_Plane2_1_of_1.png'
    test_image_label = 0

    test_image_path = os.path.join(test_image_dir, test_image_name)

    # load the model
    check_test_accu = False

    load_model_dir = 'saved_models/densenet161/003/pruned_prototypes_epoch10_k6_pt3/'  # '/usr/xtmp/mammo/alina_saved_models/vgg16/finer_1118_top2percent_randseed=1234/'
    load_model_name = '10_19_26prune0.9421.pth'  # '100_9push0.9258.pth'

    model_base_architecture = load_model_dir.split('/')[-3]
    experiment_run = '/'.join(load_model_dir.split('/')[-2:])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    load_model_path = os.path.join(load_model_dir, load_model_name)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    model = ppnet_multi

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    class_specific = True

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    # load the test data and check test accuracy
    from settings import test_dir

    test_batch_size = 100

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    print('test set size: {0}'.format(len(test_loader.dataset)))

    cm = np.zeros((7, 7))

    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(test_loader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)


                l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)

            # update confusion matrix
            for j in range(len(predicted)):
                cm[predicted[j]][label[j]] += 1
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        del input
        del target
        del output
        del predicted
        del min_distances

    print(cm)
    end = time.time()

    #write cm to a csv
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('cm.csv')



if __name__ == '__main__':
    main()
