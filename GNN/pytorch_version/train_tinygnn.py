from GNN.globals import *
from GNN.pytorch_version.models import GraphSAINT, StudentGraphSAINT
from GNN.pytorch_version.minibatch import Minibatch
from GNN.utils import *
from GNN.metric import *
from GNN.pytorch_version.utils import *
# from GNN.pytorch_version.gpu_profile import gpu_profile
import hashlib
import torch
import time
import os

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
    When calculating the F1 score, we will mask the relevant root nodes.
    """
    time_s = time.time()
    loss, preds, labels = model.eval_step(*minibatch.one_batch(mode=mode))
    torch.cuda.synchronize()
    time_e = time.time()
    node_val_test = minibatch.node_val if mode == 'val' else minibatch.node_test
    f1_scores = calc_f1(to_numpy(labels[node_val_test]),
                        to_numpy(preds[node_val_test]), model.sigmoid_loss)
    # node_test=minibatch.node_test
    # f1_test=calc_f1(to_numpy(labels[node_test]),to_numpy(preds[node_test]),model.sigmoid_loss)
    # printf(' ******TEST:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss,f1_test[0],f1_test[1]),style='yellow')
    del labels
    del preds
    return loss, f1_scores[0], f1_scores[1], time_e - time_s

def evaluate_full_batch_teacher(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
    When calculating the F1 score, we will mask the relevant root nodes.
    """
    time_s = time.time()
    loss, preds, labels = model.eval_step(*minibatch.one_batch(mode=mode))
    torch.cuda.synchronize()
    time_e = time.time()
    node_val_test = minibatch.node_val if mode == 'val' else minibatch.node_test
    f1_scores = calc_f1(to_numpy(labels[node_val_test]),
                        to_numpy(preds[node_val_test]), model.sigmoid_loss)
    # node_test=minibatch.node_test
    # f1_test=calc_f1(to_numpy(labels[node_test]),to_numpy(preds[node_test]),model.sigmoid_loss)
    # printf(' ******TEST:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss,f1_test[0],f1_test[1]),style='yellow')
    return loss, f1_scores[0], f1_scores[1], time_e - time_s, preds


def evaluate_minibatch(model_eval, minibatch_eval, inf_params, mode='test'):
    time_s = time.time()
    nodes = minibatch_eval.node_test if mode == 'test' else minibatch_eval.node_val
    preds, labels, t_forward, t_sampling = model_eval.minibatched_eval(
        nodes, minibatch_eval.adj_full_norm_sp, inf_params)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1_scores = calc_f1(labels, preds, model_eval.sigmoid_loss)
    t_total = time.time() - time_s
    return f1_scores[0], f1_scores[1], t_forward, t_sampling, t_total

def approx_evaluate_minibatch(model_eval, minibatch_eval, inf_params, mode='test', model_cpu=None, minibatch_cpu=None):
    if not args_global.cpu_eval:
        historical_feat = model_eval.get_input_activation(*minibatch_eval.one_batch(mode='val'), 1)
        if not model_eval.use_cuda:
            historical_feat = historical_feat.cpu()
    else:
        historical_feat = model_cpu.get_input_activation(*minibatch_cpu.one_batch(mode='val'), 1)
        if model_eval.use_cuda:
            historical_feat = historical_feat.cuda()
    time_s = time.time()
    preds, labels, t_forward, t_sampling = model_eval.approx_minibatched_eval(
        minibatch_eval.node_test, minibatch_eval.adj_full_norm_sp, inf_params, historical_feat,minibatch_eval.node_trainval.astype(np.int32))
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1_scores = calc_f1(labels, preds, model_eval.sigmoid_loss)
    t_total = time.time() - time_s
    return f1_scores[0], f1_scores[1], t_forward, t_sampling, t_total


def prepare(train_data, train_params, teach_arch_gcn, student_arch_gcn):
    adj_full, adj_train, adj_val, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_val = adj_val.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    adj_val_norm = adj_norm(adj_val)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, adj_val_norm, role,
                          train_params)
    teacher_model = GraphSAINT(num_classes, teacher_arch_gcn, train_params, feat_full,
                       class_arr)
    student_model = StudentGraphSAINT(num_classes, student_arch_gcn, train_params, feat_full,
                       class_arr)
    minibatch_eval = Minibatch(adj_full_norm,
                               adj_train,
                               adj_val_norm,
                               role,
                               train_params,
                               cpu_eval=True)
    if args_global.gpu >= 0:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
    return teacher_model, student_model, minibatch, minibatch_eval


def train(train_phases,
          model,
          minibatch,
          minibatch_eval,
          model_eval,
          student_model,
          path_saver=None,
          inf_params=None):
    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    if path_saver is None:
        path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(
            args_global.dir_log, timestamp)
    print('Training teacher model...')
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip), style='underline')
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e), style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train, preds_train, labels_train = model.train_step(
                    *minibatch.one_batch(mode='train'))
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),
                                             to_numpy(preds_train),
                                             model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
            if args_global.cpu_eval:
                torch.save(model.state_dict(), 'tmp.pkl')
                model_eval.load_state_dict(
                    torch.load('tmp.pkl',
                               map_location=lambda storage, loc: storage))
                model_full = model_eval
                minibatch_full = minibatch_eval
            else:
                model_full = model
                minibatch_full = minibatch
            printf(
                ' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'
                .format(f_mean(l_loss_tr), f_mean(l_f1mic_tr),
                        f_mean(l_f1mac_tr), time_train_ep))
            if not args_global.minibatch_eval:
                loss_val, f1mic_val, f1mac_val, f_time = evaluate_full_batch(
                    model_full, minibatch_full, mode='val')
            else:
                loss_val = 0
                f1mic_val, f1mac_val, _, _, f_time = evaluate_minibatch(model, minibatch, inf_params, mode='val')
            printf(
                ' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttime = {:.4f}s'
                .format(loss_val, f1mic_val, f1mac_val, f_time),
                style='yellow')
            if f1mic_val > f1mic_best:
                f1mic_best, ep_best = f1mic_val, e
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                printf('  Saving model ...', style='yellow')
                torch.save(model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        printf('  Restoring model ...', style='yellow')
        model.load_state_dict(torch.load(path_saver))
        model_eval.load_state_dict(
            torch.load(path_saver,
                        map_location=lambda storage, loc: storage))
    loss_val, f1mic_val, f1mac_val, f_time, from_teacher = evaluate_full_batch_teacher(
        model_full, minibatch_full, mode='val')
    printf(
        "Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(ep_best, f1mic_val, f1mac_val, f_time),
        style='red')
    loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
        model_full, minibatch_full, mode='test')
    printf(
        "Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(f1mic_test, f1mac_test, f_time),
        style='red')
    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    time_train = 0
    print('Training student model...')
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip), style='underline')
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e), style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train, preds_train, labels_train = student_model.train_step(
                    *minibatch.one_batch(mode='train'), from_teacher)
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),
                                             to_numpy(preds_train),
                                             model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
            printf(
                ' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'
                .format(f_mean(l_loss_tr), f_mean(l_f1mic_tr),
                        f_mean(l_f1mac_tr), time_train_ep))
            loss_val, f1mic_val, f1mac_val, f_time = evaluate_full_batch(
                student_model, minibatch, mode='val')
            printf(
                ' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttime = {:.4f}s'
                .format(loss_val, f1mic_val, f1mac_val, f_time),
                style='yellow')
            if f1mic_val > f1mic_best:
                f1mic_best, ep_best = f1mic_val, e
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                printf('  Saving model ...', style='yellow')
                torch.save(student_model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        printf('  Restoring model ...', style='yellow')
        student_model.load_state_dict(torch.load(path_saver))
    loss_val, f1mic_val, f1mac_val, f_time, from_teacher = evaluate_full_batch_teacher(
        student_model, minibatch, mode='val')
    printf(
        "Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(ep_best, f1mic_val, f1mac_val, f_time),
        style='red')
    loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
        student_model, minibatch, mode='test')
    printf(
        "Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(f1mic_test, f1mac_test, f_time),
        style='red')

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    # sys.settrace(gpu_profile)
    train_params, train_phases, train_data, teacher_arch_gcn, student_arch_gcn = parse_n_prepare_tinygnn(
        args_global)
    teacher_model, student_model, minibatch, minibatch_eval = prepare(
        train_data, train_params, teacher_arch_gcn, student_arch_gcn)
    train(train_phases,
          teacher_model,
          minibatch,
          minibatch_eval,
          teacher_model,
          student_model)
