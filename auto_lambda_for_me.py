import copy
import torch

class AutoLambda:
    def __init__(self, model, device, task_num=2, weight_init=[1,1]):
        self.model = model
        # 之所以要copy一下模型，就是为了试探性更新网络参数来优化的损失权重，因为是试探性的所以不能够直接就把网络参数改了
        # 其实他本质思想就是根据较临近的未来网络状态在未见过的图像（即新训练数据，因为预处理的存在，在训练时每个batch都有random_crop和随机翻转处理）上的表现（loss）
        # 去预测较临近的未来损失权重
        self.model_ = copy.deepcopy(model)
        assert len(weight_init) == task_num, "The length of weight_init must be equal to task_num."
        self.meta_weights = torch.tensor(weight_init, requires_grad=True, device=device, dtype=torch.float)

    def virtual_step(self, batch, alpha, model_optim,yolo_loss,salience_loss):
        """
        Compute unrolled network theta' (virtual step)
        """
        # forward & compute loss
        images, pnges, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if torch.cuda.is_available():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            else:
                images = torch.from_numpy(images).type(torch.FloatTensor)
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

        outputs = self.model(images)
        object_loss_value = 0
        for l in range(len(outputs[:-1])):
            loss_item = yolo_loss(l, outputs[l], targets)
            object_loss_value += loss_item

        salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

        loss = self.meta_weights[0] * object_loss_value + self.meta_weights[1] * salience_loss_value

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.filt_trainable_params())

        # # 检查哪些参数的梯度为 None
        # for (name, param), grad in zip(self.model.named_parameters(), gradients):
        #     if grad is None:
        #         print(f"No gradient for layer: {name}")
        #     else:
        #         print(f"Gradient computed for layer: {name}")

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.filt_trainable_params(), self.model_.filt_trainable_params(), gradients):
                # if grad is None:
                #     grad = torch.zeros_like(weight)  # 为None的梯度替换为零张量
                # if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                #     m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                # else:
                #     m = 0
                #weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))
                weight_.copy_(weight - alpha * (grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, batch, meta_batch, alpha, model_optim,yolo_loss,salience_loss):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(batch, alpha, model_optim,yolo_loss,salience_loss)

        # compute validation data loss on primary tasks

        images, pnges, targets = meta_batch[0], meta_batch[1], meta_batch[2]
        with torch.no_grad():
            if torch.cuda.is_available():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            else:
                images = torch.from_numpy(images).type(torch.FloatTensor)
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

        outputs = self.model_(images)
        object_loss_value = 0
        for l in range(len(outputs[:-1])):
            loss_item = yolo_loss(l, outputs[l], targets)
            object_loss_value += loss_item

        salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

        loss = object_loss_value + salience_loss_value

        # compute hessian via finite difference approximation

        model_weights_ = list(self.model_.filt_trainable_params())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, batch,yolo_loss,salience_loss)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, batch,yolo_loss,salience_loss):
        norm = torch.cat([w.reshape(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.filt_trainable_params(), d_model):
                p += eps * d

        images, pnges, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if torch.cuda.is_available():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            else:
                images = torch.from_numpy(images).type(torch.FloatTensor)
                pnges = torch.from_numpy(pnges).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

        outputs = self.model(images)
        object_loss_value = 0
        for l in range(len(outputs[:-1])):
            loss_item = yolo_loss(l, outputs[l], targets)
            object_loss_value += loss_item

        salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

        loss = self.meta_weights[0] * object_loss_value + self.meta_weights[1] * salience_loss_value

        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.filt_trainable_params(), d_model):
                p -= 2 * eps * d

        outputs = self.model(images)
        object_loss_value = 0
        for l in range(len(outputs[:-1])):
            loss_item = yolo_loss(l, outputs[l], targets)
            object_loss_value += loss_item

        salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

        loss = self.meta_weights[0] * object_loss_value + self.meta_weights[1] * salience_loss_value

        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.filt_trainable_params(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian