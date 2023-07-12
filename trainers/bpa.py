"""
Author: Rui Hu
All rights reserved.
"""

import torch
from kmeans_pytorch import kmeans

from tqdm import tqdm
from .erm import ERMTrainer
from utils import AverageMeter
from model.centroids import AvgFixedCentroids


update_cluster_iter = 10


class BPATrainer(ERMTrainer):
    method_name = "bpa"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def init_biased_model(self, train_loader):
        args = self.args
        args.bias_epoch = 1
        # train bias model
        self.model.train()
        losses = AverageMeter()

        pbar = tqdm(range(1, args.bias_epoch + 1))
        for epoch in pbar:
            for _, (image, target) in train_loader:
                image = image.cuda()
                label = target[:, 0].cuda()
                bs = len(image)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    output, _ = self.model(image)
                    loss = self.criterion(output, label).mean()

                losses.update(loss.item(), bs)

                self._loss_backward(loss)
                self._optimizer_step(self.optimizer)
                self._scaler_update()
                self.optimizer.zero_grad(set_to_none=True)

                pbar.set_description(
                    f"init_biased_model loss: {losses.avg:.6f}"
                )

    @torch.no_grad()
    def _extract_features(self, model=None):
        args = self.args
        if model is None:
            model = self.model

        model.eval()

        features, labels = [], []
        ids = []
        for index, (image, target) in tqdm(self.train_loader, desc='Feature extraction for clustering..'):
            image, target, index = image.cuda(), target.cuda(), index.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                _, feat = model(image)

            features.append(feat)
            labels.append(target[:, 0])
            ids.append(index)

        features = torch.cat(features)
        labels = torch.cat(labels)
        ids = torch.cat(ids)
        return features, labels, ids

    def _cluster_features(self, data_loader, features, labels, ids, num_clusters):
        N = len(data_loader.dataset)
        num_classes = 2
        sorted_target_clusters = torch.zeros(N).long().cuda() + num_clusters * num_classes
        target_clusters = torch.zeros_like(labels) - 1
        cluster_centers = []

        for t in range(num_classes):
            target_assigns = (labels == t).nonzero().squeeze()
            feautre_assigns = features[target_assigns]
            cluster_ids, cluster_center = kmeans(
                X=feautre_assigns,
                num_clusters=num_clusters,
                distance='cosine',
                device=0
            )
            cluster_ids_ = cluster_ids + t * num_clusters

            target_clusters[target_assigns] = cluster_ids_.cuda()
            cluster_centers.append(cluster_center)

        sorted_target_clusters[ids] = target_clusters
        cluster_centers = torch.cat(cluster_centers, 0)
        return sorted_target_clusters, cluster_centers

    @torch.no_grad()
    def inital_clustering(self):
        data_loader = self.train_loader
        self.model.eval()

        features, labels, ids = self._extract_features()
        num_clusters = 8
        cluster_assigns, cluster_centers = self._cluster_features(
            data_loader,
            features,
            labels,
            ids,
            num_clusters
        )
        cluster_counts = cluster_assigns.bincount().float()
        print("Cluster counts : {}, len({})".format(cluster_counts, len(cluster_counts)))
        return cluster_assigns, cluster_centers

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _method_special_setups(self):
        self.centroids = AvgFixedCentroids(
            num_classes=2,
            per_clusters=8
        )

    def _before_train(self):
        self.init_biased_model(self.train_loader)
        if not self.centroids.initialized:
            cluster_assigns, cluster_centers = self.inital_clustering()
            self.centroids.initialize_(cluster_assigns, cluster_centers)
        self._setup_model()
        self._setup_optimizer()

    def train(self):
        args = self.args

        i = 0
        losses = AverageMeter()
        pbar = tqdm(self.train_loader)
        for ids, (image, target) in pbar:
            image = image.cuda()
            label = target[:, 0].cuda()
            ids = ids.cuda()
            bs = len(image)
            i += 1

            weight = self.centroids.get_cluster_weights(ids)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = self.model(image)
                loss_per_sample = self.criterion(output, label)
                loss_weighted = loss_per_sample * weight
                loss = loss_weighted.mean()

            losses.update(loss.item(), bs)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.centroids.initialized:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    self.centroids.update(output, label, ids)
                    if i % update_cluster_iter == 0:
                        self.centroids.compute_centroids()

            pbar.set_description(
                f"[{self.cur_epoch}/{args.num_epoch}] "
                f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:g} "
                f"loss: {losses.avg:.6f}"
            )

        self.log_to_wandb({
            "loss": losses.avg,
            "custom_step": self.cur_epoch
        })