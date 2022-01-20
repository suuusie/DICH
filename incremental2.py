import torch
import torch.optim as optim
import time
import models.CNNF as CNNF
import models.mlp as mlp
import utils.evaluate as evaluate
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from models.incremental_loss_img import incremental_Loss_img
from models.incremental_loss_txt import incremental_Loss_txt
from data.data_loader import sample_dataloader


def increment(
        query_dataloader,
        seen_dataloader,
        unseen_dataloader,
        retrieval_dataloader,
        old_B,
        code_length,
        device,
        lr_img,
        lr_txt,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        dataset,
        lamda,
        mu,
        theta,
        gamma,
        eta,
        topk,
        num_seen_class,
        PRETRAIN_MODEL_PATH,
):
    """
    Increment model.

    Args
        query_dataloader, unseen_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        old_B(torch.Tensor): Old binary hash code.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma, theta(float): Hyper-parameters.
        topk(int): Top k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    pretrain_model = sio.loadmat(PRETRAIN_MODEL_PATH)
    ydim = seen_dataloader.dataset.get_tag().size(1)
    logger.debug('loading alexnet model....')
    model_img = CNNF.load_model(code_length, pretrain_model).to(device)
    model_img.train()
    logger.debug('loading MLP model....')
    model_txt = mlp.load_model(ydim,code_length).to(device)
    model_txt.train()

    optimizer = optim.Adam([
        {'params': model_img.parameters()},
        {'params': model_txt.parameters()},
    ], lr=lr_img, weight_decay=0.0005)


    criterion_img = incremental_Loss_img(code_length, gamma, eta)
    criterion_txt = incremental_Loss_txt(code_length,gamma, eta)
    lr_scheduler = ExponentialLR(optimizer, 0.9)

    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    seen_targets = seen_dataloader.dataset.get_onehot_targets().to(device)
    unseen_targets = unseen_dataloader.dataset.get_onehot_targets().to(device)
    num_unseen_class = len(unseen_targets.T) - num_seen_class
    L_seen = seen_targets[:,:num_seen_class]
    L_unseen = unseen_targets[:,num_seen_class:]
    L_unseen_seen = unseen_targets[:, :num_seen_class]
    L_seen = L_seen.to(torch.float32)
    L_unseen = L_unseen.to(torch.float32)
    L_unseen_seen = L_unseen_seen.to(torch.float32)

    num_unseen = len(unseen_dataloader.dataset)
    num_seen = len(old_B)
    U_x = torch.zeros(num_samples, code_length).to(device)
    U_y = torch.zeros(num_samples, code_length).to(device)
    old_B = old_B.to(device)
    new_B = torch.randn(num_unseen, code_length).sign().to(device)
    Y_new = torch.randn(num_unseen_class, code_length).to(device)
    Y_old = torch.randn(num_seen_class, code_length).to(device)

    B = torch.cat((old_B, new_B), dim=0).to(device)


    S = torch.zeros(num_seen_class, num_unseen_class)
    S = S - 1
    S = S.to(device)


    total_time = time.time()
    evl_time = 0
    for it in range(max_iter):
        iter_time = time.time()
        lr_scheduler.step()

        # logger.debug('sampling....')
        logger.debug('[iter:{}/{}]'.format(it +1, max_iter))
        # Sample training data for cnn learning
        train_dataloader, sample_index, unseen_sample_in_unseen_index, unseen_sample_in_sample_index = sample_dataloader(retrieval_dataloader, num_samples, num_seen, batch_size, root, dataset)

        for epoch in range(max_epoch):
            # logger.debug('[epoch:{}/{}]'.format(epoch +1, max_epoch))
            # Training CNN model
            for batch, (image, tag, targets, index) in enumerate(train_dataloader):
                image, tag, targets, index = image.to(device), tag.to(device) ,targets.to(device), index.to(device)
                sample_index_gpu = torch.from_numpy(sample_index)
                sample_index_gpu = sample_index_gpu.to(device)
                omega = sample_index_gpu[index]
                tag = tag.to(torch.float32)
                optimizer.zero_grad()

                F_img = model_img(image)
                F_txt = model_txt(tag)
                U_x[index, :] = F_img.data
                U_y[index, :] = F_txt.data
                cnn_loss_img = criterion_img(F_img, B, omega )
                cnn_loss_txt = criterion_txt(F_txt, B, omega)

                # logger.debug('[image_loss:{}]'.format(cnn_loss_img))
                loss = cnn_loss_img + cnn_loss_txt

                loss.backward()
                optimizer.step()
                # logger.debug('[image_batch:{}/{}]'.format(batch +1, len(train_dataloader)))


        # update Y_old
        # logger.debug('updating Y_old ...')
        # # DCC
        G = lamda * L_seen.t() @ old_B + mu * L_unseen_seen.t() @ new_B + theta * S @ Y_new
        for bit in range(code_length):
            g = G[:, bit]
            b_old = old_B[:, bit]
            b_new = new_B[:, bit]
            y_new = Y_new[:, bit]

            Y_old_prime = torch.cat((Y_old[:, :bit], Y_old[:, bit + 1:]), dim=1)
            Y_new_prime = torch.cat((Y_new[:, :bit], Y_new[:, bit + 1:]), dim=1)
            B_old_prime = torch.cat((old_B[:, :bit], old_B[:, bit + 1:]), dim=1)
            B_new_prime = torch.cat((new_B[:, :bit], new_B[:, bit + 1:]), dim=1)


            Y_old[:, bit] = (code_length * g - lamda * Y_old_prime @ B_old_prime.t() @ b_old - mu * Y_old_prime @ B_new_prime.t() @ b_new - theta * Y_old_prime @ Y_new_prime.t() @ y_new ).sign()

        # update Y_new
        # logger.debug('updating Y_new ...')
        # DCC
        G = mu * L_unseen.t() @ new_B + theta * S.t() @ Y_old
        for bit in range(code_length):
            g = G[:, bit]
            b_new = new_B[:, bit]
            y_old = Y_old[:, bit]

            Y_old_prime = torch.cat((Y_old[:, :bit], Y_old[:, bit + 1:]), dim=1)
            Y_new_prime = torch.cat((Y_new[:, :bit], Y_new[:, bit + 1:]), dim=1)
            B_new_prime = torch.cat((new_B[:, :bit], new_B[:, bit + 1:]), dim=1)

            Y_new[:, bit] = (code_length * g - mu * Y_new_prime @ B_new_prime.t() @ b_new - theta * Y_new_prime @ Y_old_prime.t() @ y_old).sign()

        # Update B_new
        # logger.debug('updating B_new...')

        expand_Ux = torch.zeros(num_unseen, code_length).to(device)
        expand_Ux[unseen_sample_in_unseen_index, :] = U_x[unseen_sample_in_sample_index, :]
        expand_Uy = torch.zeros(num_unseen, code_length).to(device)
        expand_Uy[unseen_sample_in_unseen_index, :] = U_y[unseen_sample_in_sample_index, :]
        # DCC
        G = mu * code_length * L_unseen @ Y_new + mu * code_length * L_unseen_seen @ Y_old + gamma * expand_Ux + gamma * expand_Uy
        for bit in range(code_length):
            g = G[:, bit]
            y_new = Y_new[:, bit]
            y_old = Y_old[:, bit]

            B_prime = torch.cat((new_B[:, :bit], new_B[:, bit + 1:]), dim=1)
            Y_new_prime = torch.cat((Y_new[:, :bit], Y_new[:, bit + 1:]), dim=1)
            Y_old_prime = torch.cat((Y_old[:, :bit], Y_old[:, bit + 1:]), dim=1)

            new_B[:, bit] = (g - mu * B_prime @ Y_new_prime.t() @ y_new - mu * B_prime @ Y_old_prime.t() @ y_old).sign()

        B = torch.cat((old_B, new_B), dim=0).to(device)

        if (it + 1 ) == 1:
            # Total loss
            evl_iter_start = time.time()

            # Each iter map
            # logger.debug('generate code ...')
            query_code_img = generate_code_img(model_img, query_dataloader, code_length, device)
            query_code_txt = generate_code_txt(model_txt, query_dataloader, code_length, device)
            # logger.debug('caculating map ...')
            #
            mAP_i2t = evaluate.mean_average_precision(
                query_code_img.to(device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_dataloader.dataset.get_onehot_targets().to(device),
                device,
                topk,
            )
            mAP_t2i = evaluate.mean_average_precision(
                query_code_txt.to(device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_dataloader.dataset.get_onehot_targets().to(device),
                device,
                topk,
            )
            evl_time_iter = time.time() - evl_iter_start
            evl_time = evl_time + evl_time_iter

            # logger.debug('one iter has finished ...')
            logger.info('[iter:{}/{}][time:{:.2f}][mapit2:{:.4f}][mapt2i:{:.4f}]'.format(it + 1, max_iter,

                                                                                                      time.time() - iter_time - evl_time_iter,
                                                                                                      mAP_i2t, mAP_t2i))
        if (it + 1) % 5 == 0:
            # Total loss
            evl_iter_start = time.time()

            # Each iter map
            # logger.debug('generate code ...')
            query_code_img = generate_code_img(model_img, query_dataloader, code_length, device)
            query_code_txt = generate_code_txt(model_txt, query_dataloader, code_length, device)
            # logger.debug('caculating map ...')
            #
            mAP_i2t = evaluate.mean_average_precision(
                query_code_img.to(device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_dataloader.dataset.get_onehot_targets().to(device),
                device,
                topk,
            )
            mAP_t2i = evaluate.mean_average_precision(
                query_code_txt.to(device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_dataloader.dataset.get_onehot_targets().to(device),
                device,
                topk,
            )

            evl_time_iter = time.time() - evl_iter_start
            evl_time = evl_time + evl_time_iter

            # logger.debug('one iter has finished ...')
            logger.info('[iter:{}/{}][time:{:.2f}][mapit2:{:.4f}][mapt2i:{:.4f}]'.format(it + 1, max_iter,

                                                                                                      time.time() - iter_time - evl_time_iter,
                                                                                                      mAP_i2t, mAP_t2i))

    logger.info('Training incremental finish, time:{:.2f}'.format(time.time()-total_time - evl_time))

    # Evaluate
    query_code_img = generate_code_img(model_img, query_dataloader, code_length, device)
    query_code_txt = generate_code_txt(model_txt, query_dataloader, code_length, device)

    mAP_i2t = evaluate.mean_average_precision(
        query_code_img.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_dataloader.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    mAP_t2i = evaluate.mean_average_precision(
        query_code_txt.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_dataloader.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    logger.info('[incremental][mapi2t:{:.4f}][mapt2i:{:.4f}]'.format(mAP_i2t, mAP_t2i))

    return mAP_i2t, mAP_t2i




def calc_loss(L_seen, old_B, Y_old, L_unseen, new_B, Y_new, L_unseen_seen, S, B, U_x, U_y, omega, lamda, mu, theta, gamma, eta, code_length):
    """
    Calculate loss.
    """
    omega = np.array(omega.cpu())  # win10√ linux×
    loss1 = ((code_length * L_seen - old_B @ Y_old.t()) ** 2).sum()
    loss2 = ((code_length * L_unseen - new_B @ Y_new.t()) ** 2).sum() + ((code_length * L_unseen_seen - new_B @ Y_old.t()) ** 2).sum()
    loss3 = ((code_length * S - Y_old @ Y_new.t()) ** 2).sum()
    quantization_loss1 = ((U_x - B[omega, :]) ** 2).sum()
    quantization_loss2 = ((U_y - B[omega, :]) ** 2).sum()
    correlation_loss1 = ((U_x.t() @ torch.ones(U_x.shape[0], 1, device=U_x.device)) **2).sum()
    correlation_loss2 = ((U_y.t() @ torch.ones(U_y.shape[0], 1, device=U_y.device)) **2).sum()

    loss = (lamda * loss1 + mu * loss2 + theta * loss3 + gamma * (quantization_loss1 + quantization_loss2) + eta * (correlation_loss1 + correlation_loss2)) / (U_x.shape[0] * B.shape[0])
    return loss.item(), loss1, loss2, loss3, quantization_loss1, quantization_loss2, correlation_loss1, correlation_loss2


def generate_code_img(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])

        for image,_,_, index in dataloader:
            image = image.to(device)
            hash_code = model(image)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code

def generate_code_txt(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])

        for _, tag,_, index in dataloader:
            tag = tag.to(device)
            tag = tag.to(torch.float32)
            hash_code = model(tag)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
