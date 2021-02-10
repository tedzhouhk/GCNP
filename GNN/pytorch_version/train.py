from GNN.globals import *
from GNN.pytorch_version.models import GraphSAINT, PrunedGraphSAINT
from GNN.pytorch_version.minibatch import Minibatch
from GNN.utils import *
from GNN.metric import *
from GNN.pytorch_version.utils import *
from GNN.pytorch_version.prune import Lasso
from GNN.pytorch_version.plot import *
# from GNN.pytorch_version.gpu_profile import gpu_profile
import hashlib
import torch
import time
import os

def store_batch_level_result(fname, preds, labels, t_forward, t_sampling, sigmoid_loss):
    if not os.path.isdir('yelpchi_batch_result'):
        os.mkdir('yelpchi_batch_result')
    fname = 'yelpchi_batch_result/' + fname + '.txt'
    t_total = t_forward + t_sampling
    with open(fname,'w') as f:
        for i in range(t_total.shape[0]):
            f1 = calc_f1(labels[i], preds[i], sigmoid_loss)[0]
            f.write('{:.4f}\t{:.6f}\n'.format(f1, t_total[i]))

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


def evaluate_minibatch(model_eval, minibatch_eval, inf_params, mode='test'):
    nodes = minibatch_eval.node_test if mode == 'test' else minibatch_eval.node_val
    if 'oversample' in inf_params and isinstance(model_eval, PrunedGraphSAINT):
        eval_func = model_eval.minibatched_eval_with_over_sampling
        store_each = True
    else:
        eval_func = model_eval.minibatched_eval
        store_each = False
    preds, labels, t_forward, t_sampling = eval_func(
        nodes, minibatch_eval.adj_full_norm_sp, inf_params)
    if store_each:
        store_batch_level_result(args_global.train_config.split('/')[-1].split('.')[0], preds, labels, np.array(t_forward), np.array(t_sampling), model_eval.sigmoid_loss)
        t_forward = np.sum(np.array(t_forward))
        t_sampling = np.sum(np.array(t_sampling))
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1_scores = calc_f1(labels, preds, model_eval.sigmoid_loss)
    return f1_scores[0], f1_scores[1], t_forward, t_sampling, t_forward + t_sampling

def approx_evaluate_minibatch(model_eval, minibatch_eval, inf_params, mode='test', model_cpu=None, minibatch_cpu=None):
    if not args_global.cpu_eval:
        historical_feat = model_eval.get_input_activation(*minibatch_eval.one_batch(mode='val'), 1)
        if not model_eval.use_cuda:
            historical_feat = historical_feat.cpu()
    else:
        historical_feat = model_cpu.get_input_activation(*minibatch_cpu.one_batch(mode='val'), 1)
        if model_eval.use_cuda:
            historical_feat = historical_feat.cuda()

    if 'oversample' in inf_params and isinstance(model_eval, PrunedGraphSAINT):
        eval_func = model_eval.approx_minibatched_eval_with_over_sampling
        store_each = True
        known_nodes = minibatch_eval.node_trainval.astype(np.int32)[1:]
    else:
        eval_func = model_eval.approx_minibatched_eval
        store_each = False
        known_nodes = minibatch_eval.node_trainval.astype(np.int32)
    preds, labels, t_forward, t_sampling = eval_func(
        minibatch_eval.node_test, minibatch_eval.adj_full_norm_sp, inf_params, historical_feat, known_nodes)
    if store_each:
        store_batch_level_result(args_global.train_config.split('/')[-1].split('.')[0]+'_approx', preds, labels, np.array(t_forward), np.array(t_sampling), model_eval.sigmoid_loss)
        t_forward = np.sum(np.array(t_forward))
        t_sampling = np.sum(np.array(t_sampling))
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1_scores = calc_f1(labels, preds, model_eval.sigmoid_loss)
    return f1_scores[0], f1_scores[1], t_forward, t_sampling, t_forward + t_sampling


def prepare(train_data, train_params, arch_gcn):
    adj_full, adj_train, adj_val, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_val = adj_val.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    adj_val_norm = adj_norm(adj_val)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, adj_val_norm, role,
                          train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full,
                       class_arr)
    minibatch_eval = Minibatch(adj_full_norm,
                               adj_train,
                               adj_val_norm,
                               role,
                               train_params,
                               cpu_eval=True)
    model_eval = GraphSAINT(num_classes,
                            arch_gcn,
                            train_params,
                            feat_full,
                            class_arr,
                            cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases,
          model,
          minibatch,
          minibatch_eval,
          model_eval,
          path_saver=None,
          inf_params=None):
    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    if path_saver is None:
        path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(
            args_global.dir_log, timestamp)
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
        if args_global.cpu_minibatch:
            model_mini = model_eval
            minibatch_mini = minibatch_eval
        else:
            model_mini = model
            minibatch_mini = minibatch
        if args_global.cpu_eval:
            model_full = model_eval
            minibatch_full = minibatch_eval
        else:
            model_full = model
            minibatch_full = minibatch
    loss_val, f1mic_val, f1mac_val, f_time = evaluate_full_batch(
        model_full, minibatch_full, mode='val')
    printf(
        "Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(ep_best, f1mic_val, f1mac_val, f_time),
        style='red')
    if args_global.profile_fullbatch:
        if 'prof' in locals() or 'prof' in globals():
            del prof
        with torch.autograd.profiler.profile(
                use_cuda=model_full.use_cuda) as prof:
            loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
                model_full, minibatch_full, mode='test')
        sort_key = "cuda_time_total" if model_full.use_cuda else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    else:
        loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
            model_full, minibatch_full, mode='test')
    printf(
        "Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(f1mic_test, f1mac_test, f_time),
        style='red')
    if args_global.store_result != "":
        with open(args_global.store_result, encoding="utf-8",mode="a") as f:  
            f.write(args_global.train_config + '\t' + str(f1mic_test) + '\n')  
    
    if inf_params is not None:
        if args_global.profile_minibatch:
            if 'prof' in locals() or 'prof' in globals():
                del prof
            with torch.autograd.profiler.profile(
                    use_cuda=model_mini.use_cuda) as prof:
                f1mic_test, f1mac_test, t_forward, t_sampling, t_total = evaluate_minibatch(
                    model_mini, minibatch_mini, inf_params, mode='test')
            sort_key = "cuda_time_total" if model_mini.use_cuda else "cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
        else:
            f1mic_test, f1mac_test, t_forward, t_sampling, t_total = evaluate_minibatch(
                model_mini, minibatch_mini, inf_params, mode='test')
        printf(
            "Full test stats (minibatched): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tf_total = {:.4f}s\t t_forward = {:.4f}s\tt_sampling = {:.4f}s"
            .format(f1mic_test, f1mac_test, t_forward + t_sampling, t_forward,
                    t_sampling),
            style='red')
        if type(model) is PrunedGraphSAINT and inf_params['approx']:
            if args_global.profile_approxminibatch:
                if 'prof' in locals() or 'prof' in globals():
                    del prof
                with torch.autograd.profiler.profile(
                        use_cuda=model_mini.use_cuda) as prof:
                    f1mic_test, f1mac_test, t_forward, t_sampling, t_total = approx_evaluate_minibatch(
                        model_mini, minibatch_mini, inf_params, mode='test', model_cpu=model_eval, minibatch_cpu=minibatch_eval)
                sort_key = "cuda_time_total" if model.use_cuda else "cpu_time_total"
                print(prof.key_averages().table(sort_by=sort_key,
                                                row_limit=10))
            else:
                f1mic_test, f1mac_test, t_forward, t_sampling, t_total = approx_evaluate_minibatch(
                    model_mini, minibatch_mini, inf_params, mode='test', model_cpu=model_eval, minibatch_cpu=minibatch_eval)
            printf(
                "Full test stats (approximated, minibatched): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tf_total = {:.4f}s\t t_forward = {:.4f}s\tt_sampling = {:.4f}s"
                .format(f1mic_test, f1mac_test, t_forward + t_sampling, t_forward,
                        t_sampling),
                style='red')        

def get_model(train_phases, train_params, arch_gcn, model, minibatch,
              minibatch_eval, model_eval, inf_params):
    text = args_global.data_prefix
    for k, v in arch_gcn.items():
        text += str(k) + str(v)
    for train_phase in train_phases:
        for k, v in train_phase.items():
            text += str(k) + str(v)
    for k, v in train_params.items():
        text += str(k) + str(v)
    path_saver = 'pytorch_models/' + hashlib.md5(
        text.encode('utf-8')).hexdigest() + '.pkl'
    if os.path.exists(path_saver):
        printf("Found existing model: {}".format(path_saver))
        model.load_state_dict(torch.load(path_saver))
        model_eval.load_state_dict(
            torch.load(path_saver,
                        map_location=lambda storage, loc: storage))
        if args_global.cpu_minibatch:
            model_mini = model_eval
            minibatch_mini = minibatch_eval
        else:
            model_mini = model
            minibatch_mini = minibatch
        if args_global.cpu_eval:
            model_full = model_eval
            minibatch_full = minibatch_eval
        else:
            model_full = model
            minibatch_full = minibatch
        if args_global.profile_fullbatch:
            if 'prof' in locals() or 'prof' in globals():
                del prof
            with torch.autograd.profiler.profile(
                    use_cuda=model_full.use_cuda) as prof:
                loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
                    model_full, minibatch_full, mode='test')
            sort_key = "cuda_time_total" if model_full.use_cuda else "cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
        else:
            loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
                model_full, minibatch_full, mode='test')
        printf(
            "Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
            .format(f1mic_test, f1mac_test, f_time),
            style='red')
        if args_global.profile_minibatch:
            if 'prof' in locals() or 'prof' in globals():
                del prof
            with torch.autograd.profiler.profile(
                    use_cuda=model_mini.use_cuda) as prof:
                f1mic_test, f1mac_test, t_forward, t_sampling, t_total = evaluate_minibatch(
                    model_mini, minibatch_mini, inf_params, mode='test')
            sort_key = "cuda_time_total" if model_mini.use_cuda else "cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
        else:
            f1mic_test, f1mac_test, t_forward, t_sampling, t_total = evaluate_minibatch(
                model_mini, minibatch_mini, inf_params, mode='test')
        printf(
            "Full test stats (minibatched): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tf_total = {:.4f}s\t t_forward = {:.4f}s\tt_sampling = {:.4f}s"
            .format(f1mic_test, f1mac_test, t_forward + t_sampling, t_forward,
                    t_sampling),
            style='red')
    else:
        train(train_phases,
              model,
              minibatch,
              minibatch_eval,
              model_eval,
              path_saver=path_saver,
              inf_params=inf_params)
    return model, model_eval
    
def prune(model, model_eval, retrain_params, prune_params, minibatch, minibatch_eval):
    if args_global.cpu_eval:
        model_full = model_eval
        minibatch_full = minibatch_eval
    else:
        model_full = model
        minibatch_full = minibatch
    prune_device = torch.device("cuda")
    if args_global.cpu_prune:
        prune_device = torch.device("cpu")
    mask = torch.ones(model_full.num_classes, dtype=bool)
    layers = [model_full.classifier]
    names = ['classifier']
    layer_steps = [len(model_full.aggregators)]
    for i in reversed(range(len(model_full.aggregators))):
        layers.append(model_full.aggregators[i])
        names.append('conv{}'.format(i))
        layer_steps.append(i)
    lassos = list()
    first_layer_pruned = True
    prune_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        name = names[i]
        layer_step = layer_steps[i]
        if i == len(layers) - 1:
            if prune_params['budget'][i] == 1:
                first_layer_pruned = False
            # for first GCN layer, prune each order seperately
            optimize_phase = list(range(layer.order + 1))
            stack_feature = 0
            mask = torch.split(mask,
                               split_size_or_sections=int(mask.shape[0] /
                                                          (layer.order + 1)))
            if prune_params['dynamic'] == 'static':
                budgets = [1 - prune_params['budget'][i]] * (layer.order + 1)
            elif prune_params['dynamic'] == 'linear':
                split = layer.order + 1
                dim = int(lassos[-1].beta.shape[0] / split)
                total = torch.where(lassos[-1].mask_out == True)[0].shape[0]
                budgets = list()
                for o in range(split):
                    budgets.append(1 -
                                   torch.where(lassos[-1].mask_out[o * dim:(
                                       o + 1) * dim] == True)[0].shape[0] /
                                   total * split * prune_params['budget'][i])
            elif prune_params['dynamic'] == 'svd':
                budgets = list()
                split = layer.order + 1
                dim = int(lassos[-1].beta.shape[0] / split)
                for o in range(split):
                    weight = layer.f_lin[o].weight.T[:,
                                                     mask_out[o * dim:(o + 1) *
                                                              dim]]
                    sv = torch.svd(weight, compute_uv=False)[1]
                    sv_sum = torch.sum(sv)
                    ratio = 1
                    while ratio < sv.shape[0]:
                        if torch.sum(sv[:ratio]) > sv_sum * 0.9:
                            break
                        ratio += 1
                    budgets.append(ratio / sv.shape[0])
                budgets = np.array(budgets)
                budgets /= budgets.mean()
                budgets *= prune_params['budget'][i]
                budgets = 1 - budgets
            elif prune_params['dynamic'] == 'manual':
                budgets = list()
                budgets.append(1 - prune_params['manual_self'])
                budgets.append(1 - prune_params['manual_neigh'])
            clip_separate=False
        else:
            # for middle GCN layers, prune all orders jointly
            optimize_phase = [0]
            stack_feature = layer.order
            mask = [mask]
            # if i == len(layers) - 2 and prune_params['dynamic'] == 'manual':
            #     budgets = [1]
            #     self_budget = 1 - prune_params['manual_self']
            #     neigh_budget = 1 - prune_params['manual_neigh']
            #     clip_separate = True
            # else:
            budgets = [1 - prune_params['budget'][i]]
            clip_separate = False
        for o in optimize_phase:
            print('optimizing {} phase {}:'.format(name, o))
            activation = model_full.get_input_activation(
                *minibatch_full.one_batch(mode='val'), layer_step)
            if stack_feature == 0:
                feat = activation.detach().to(prune_device)
                del activation
                for _ in range(o):
                    feat = torch.sparse.mm(minibatch.adj_val_norm.to(prune_device), feat)
                weight = torch.transpose(layer.f_lin[o].weight, 0, 1).to(prune_device)
            else:
                weight_split = [0]
                _feat = [activation.detach().to(prune_device)]
                del activation
                for _ in range(stack_feature):
                    _feat.append(
                        torch.sparse.mm(minibatch.adj_val_norm.to(prune_device), _feat[-1]))
                feat = torch.cat(_feat, 0)
                del _feat
                weight = list()
                for p in range(stack_feature + 1):
                    weight.append(
                        torch.transpose(layer.f_lin[p].weight, 0, 1).to(prune_device))
                    weight_split.append(weight_split[-1] + weight[-1].shape[1])
                weight = torch.cat(weight, 1)
            ref = torch.mm(feat, weight).detach()
            lassos.append(
                Lasso(weight.shape[0], weight, prune_params['beta_lmbd_1'][i],
                      prune_params['beta_lmbd_2'][i],
                      prune_params['beta_lmbd_1_step'][i],
                      prune_params['beta_lmbd_2_step'][i],
                      prune_params['beta_lr'], prune_params['weight_lr']))
            if args_global.gpu >= 0:
                lassos[-1] = lassos[-1].cuda()
            # make sure dimension for input attribute not too small
            if i == len(layers) - 1:
                dim_in = feat.shape[1] * (1 - budgets[o])
                dim_out = ref[0, mask[o]].shape[0]
                if dim_in < dim_out:
                    budgets[o] = max((feat.shape[1] - dim_out), 0) / feat.shape[1]
            if budgets[o] == 0:
                mask_out = lassos[-1].clip_beta(budgets[o],
                                                prune_params['beta_clip'])
                continue
            train_nodes = minibatch.node_train
            beta_loss = list()
            weight_loss = list()
            for s in range(prune_params['total_step']):
                # optimize beta
                print('  optimizing beta ...')
                for e in range(prune_params['beta_epoch']):
                    np.random.shuffle(train_nodes)
                    batches = np.array_split(
                        train_nodes,
                        (int(feat.shape[0] / prune_params['beta_batch'])))
                    for batch in batches:
                        _feat = feat[batch]
                        _ref = ref[batch]
                        if args_global.gpu >= 0:
                            _feat = _feat.cuda()
                            _ref = _ref.cuda()
                        t_s = time.time()
                        loss = lassos[-1].optimize_beta(
                            _feat, _ref, mask[o])
                        prune_time += time.time() - t_s
                    print('    epoch {} loss: {}'.format(e, loss))
                    beta_loss.append(loss)
                    lassos[-1].lmbd_step()
                # if not clip_separate:
                mask_out = lassos[-1].clip_beta(budgets[o], prune_params['beta_clip'])
                # else:
                #     mask_out = lassos[-1].seperated_clip_beta(self_budget,neigh_budget,prune_params['beta_clip'])
                # optimize weight
                print('  optimizing weight ...')
                for e in range(prune_params['weight_epoch']):
                    np.random.shuffle(train_nodes)
                    batches = np.array_split(
                        train_nodes,
                        (int(feat.shape[0] / prune_params['weight_batch'])))
                    for batch in batches:
                        _feat = feat[batch]
                        _ref = ref[batch]
                        if args_global.gpu >= 0:
                            _feat = _feat.cuda()
                            _ref = _ref.cuda()
                        t_s = time.time()
                        loss = lassos[-1].optimize_weight(
                            _feat, _ref, mask[o])
                        prune_time += time.time() - t_s
                    print('    epoch {} loss: {}'.format(e, loss))
                    weight_loss.append(loss)
                lassos[-1].norm()
            lasso_plot(lassos[-1].beta.detach().cpu().numpy(),
                       lassos[-1].weight.detach().cpu().numpy(), beta_loss,
                       weight_loss, prune_params, name + '_' + str(o))
            lassos[-1].apply_beta()
            del feat
            del weight
            del ref
            # evaluate acc on existing model
            if stack_feature == 0:
                layer.f_lin[o].weight.data.copy_(
                    torch.transpose(lassos[-1].weight, 0, 1).data)
            else:
                for p in range(stack_feature + 1):
                    layer.f_lin[p].weight.data.copy_(
                        torch.transpose(
                            lassos[-1].weight[:,
                                              weight_split[p]:weight_split[p +
                                                                           1]],
                            0, 1).data)
            # print_allocated_tensors()
            loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
                model_full, minibatch_full, mode='test')
            printf(
                "Pruned {} phase {} full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
                .format(name, o, f1mic_test, f1mac_test, f_time),
                style='red')
        mask = mask_out
    print('Pruning total time: {:.2f}s.'.format(prune_time))
    # create pruned model
    dims_in = list()
    dims_out = list()
    masks_in = list()
    pruned_weights = list()
    # dims for classifier
    dims_in.append([torch.where(lassos[0].mask_out == True)[0].shape[0]])
    dims_out.append([model_full.num_classes])
    # dims for each conv layer
    lasso_idx = 1
    feat_length_per_order = model_full._dims[1]
    for i in range(len(model_full.aggregators)):
        dim_in = list()
        dim_out = list()
        mask_in = list()
        pruned_weight = list()
        if i != len(model_full.aggregators) - 1:
            # neighbor and self pruned together, dim equal in each order
            for o in range(model_full.aggregators[i].order + 1):
                dim_in.append(
                    torch.where(
                        lassos[lasso_idx].mask_out == True)[0].shape[0])
                mask_in.append(
                    lassos[lasso_idx -
                           1].mask_out[o * feat_length_per_order:(o + 1) *
                                       feat_length_per_order])
                dim_out.append(torch.where(mask_in[-1] == True)[0].shape[0])
                pruned_weight.append(
                    lassos[lasso_idx].weight[:, o *
                                             feat_length_per_order:(o + 1) *
                                             feat_length_per_order])
            dims_in.insert(0, dim_in)
            dims_out.insert(0, dim_out)
            masks_in.insert(0, mask_in)
            pruned_weights.insert(0, pruned_weight)
            lasso_idx += 1
        else:
            # first layer, neighbor and self pruned seperately
            for o in range(model_full.aggregators[i].order + 1):
                mask_in.append(
                    lassos[lasso_idx -
                           1].mask_out[o * feat_length_per_order:(o + 1) *
                                       feat_length_per_order])
                dim_out.append(torch.where(mask_in[-1] == True)[0].shape[0])
            masks_in.insert(0, mask_in)
            dims_out.insert(0, dim_out)
    mask_0 = list()
    dim_in = list()
    pruned_weight = list()
    for o in reversed(range(model_full.aggregators[0].order + 1)):
        mask_0.append(lassos[-1 - o].mask_out)
        dim_in.append(torch.where(lassos[-1 - o].mask_out == True)[0].shape[0])
        pruned_weight.append(lassos[-1 - o].weight)
    masks_in.insert(0, mask_0)
    dims_in.insert(0, dim_in)
    pruned_weights.insert(0, pruned_weight)
    pruned_weights.append([lassos[0].weight])
    masks_in.append([torch.ones(model_full.num_classes, dtype=bool)])
    printf("Pruned model dims:", style='red')
    printf("  in: {}".format(str(dims_in)), style='red')
    printf("  out: {}".format(str(dims_out)), style='red')
    model_pruned = PrunedGraphSAINT(model_full.num_classes,
                                    model_full.arch_gcn,
                                    retrain_params,
                                    model_full.feat_full,
                                    model_full.label_full,
                                    dims_in,
                                    dims_out,
                                    mask_0,
                                    first_layer_pruned=first_layer_pruned)
    model_pruned_eval = PrunedGraphSAINT(model_full.num_classes,
                                         model_full.arch_gcn,
                                         retrain_params,
                                         model_full.feat_full,
                                         model_full.label_full,
                                         dims_in,
                                         dims_out,
                                         mask_0,
                                         cpu_eval=True,
                                         first_layer_pruned=first_layer_pruned)
    # load pruned weight
    model_pruned.aggregators[0].load_pruned_weight(masks_in[0],
                                                   masks_in[1],
                                                   model_full.aggregators[0],
                                                   pruned_weights[0],
                                                   first_layer=True)
    for i in range(1, len(model_pruned.aggregators)):
        model_pruned.aggregators[i].load_pruned_weight(
            masks_in[i],
            masks_in[i + 1],
            model_full.aggregators[i],
            pruned_weights[i],
            first_layer=False)
    model_pruned.classifier.load_pruned_weight(masks_in[-2],
                                               masks_in[-1],
                                               model_full.classifier,
                                               pruned_weights[-1],
                                               first_layer=False)
    if args_global.gpu >= 0:
        model_pruned = model_pruned.cuda()
    model_pruned_eval.load_state_dict(model_pruned.state_dict())
    if args_global.cpu_eval:
        model_pruned_full = model_pruned_eval
        minibatch_full = minibatch_eval
    else:
        model_pruned_full = model_pruned
        minibatch_full = minibatch
    loss_test, f1mic_test, f1mac_test, f_time = evaluate_full_batch(
        model_pruned_full, minibatch_full, mode='test')
    printf(
        "Pruned new model full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\ttime = {:.4f}s"
        .format(f1mic_test, f1mac_test, f_time),
        style='red')
    return model_pruned, model_pruned_eval


if __name__ == '__main__':
    train_params, train_phases, retrain_params, retrain_phases, train_data, arch_gcn, prune_params, inf_params = parse_n_prepare(
        args_global)
    model, minibatch, minibatch_eval, model_eval = prepare(
        train_data, train_params, arch_gcn)
    model, model_eval = get_model(train_phases, train_params, arch_gcn, model, minibatch,
              minibatch_eval, model_eval, inf_params)
    model_pruned, model_pruned_eval = prune(model, model_eval, retrain_params, prune_params,
                                            minibatch, minibatch_eval)
    del model, model_eval
    train(retrain_phases,
          model_pruned,
          minibatch,
          minibatch_eval,
          model_pruned_eval,
          inf_params=inf_params)
