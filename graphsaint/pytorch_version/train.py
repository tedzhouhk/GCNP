from graphsaint.globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
from graphsaint.pytorch_version.prune import Lasso
from graphsaint.pytorch_version.plot import *
import hashlib
import torch
import time
import os


def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
    When calculating the F1 score, we will mask the relevant root nodes.
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))
    node_val_test = minibatch.node_val if mode=='val' else minibatch.node_test
    f1_scores = calc_f1(to_numpy(labels[node_val_test]),to_numpy(preds[node_val_test]),model.sigmoid_loss)
    node_test=minibatch.node_test
    f1_test=calc_f1(to_numpy(labels[node_test]),to_numpy(preds[node_test]),model.sigmoid_loss)
    # printf(' ******TEST:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss,f1_test[0],f1_test[1]),style='yellow')
    return loss, f1_scores[0], f1_scores[1]



def prepare(train_data,train_params,arch_gcn):
    adj_full,adj_train,adj_val,feat_full,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_val = adj_val.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    adj_val_norm = adj_norm(adj_val)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, adj_val_norm, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    minibatch_eval=Minibatch(adj_full_norm,adj_train,adj_val_norm,role,train_params,cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, path_saver=None):
    if not args_global.cpu_eval:
        minibatch_eval=minibatch
    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    if path_saver is None:
        path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, timestamp)
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip),style='underline')
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle() 
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train,preds_train,labels_train = model.train_step(*minibatch.one_batch(mode='train'))
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
            if args_global.cpu_eval:
                torch.save(model.state_dict(),'tmp.pkl')
                model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
            else:
                model_eval=model
            loss_val, f1mic_val, f1mac_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
            printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'.format(f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr),time_train_ep))
            printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss_val,f1mic_val,f1mac_val),style='yellow')
            if f1mic_val > f1mic_best:
                f1mic_best, ep_best = f1mic_val, e
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                printf('  Saving model ...',style='yellow')
                torch.save(model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end']) 
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver,map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval=model
        printf('  Restoring model ...', style='yellow')
    loss_val, f1mic_val, f1mac_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(ep_best, f1mic_val, f1mac_val), style='red')
    loss_test, f1mic_test, f1mac_test = evaluate_full_batch(model_eval, minibatch_eval, mode='test')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test, f1mac_test), style='red')
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')

def get_model(train_phases, train_params, model, minibatch, minibatch_eval, model_eval):
    text=args_global.data_prefix
    for train_phase in train_phases:
        for k,v in train_phase.items():
            text+=str(k)+str(v)
    for k,v in train_params.items():
        text+=str(k)+str(v)
    path_saver='pytorch_models/'+hashlib.md5(text.encode('utf-8')).hexdigest()+'.pkl'
    if os.path.exists(path_saver):
        printf("Found existing model, loading and evaluating...")
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver,map_location=lambda storage,loc:storage))
        else:
            model_eval.load_state_dict(torch.load(path_saver))
        loss_test, f1mic_test, f1mac_test = evaluate_full_batch(model_eval, minibatch_eval, mode='test')
        printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test, f1mac_test), style='red')
    else:
        train(train_phases, model, minibatch, minibatch_eval, model_eval, path_saver=path_saver)
        

activation=[]
def get_activation(module, input, output):
    activation.clear()
    activation.append(input[0].detach())


def prune(model_eval,prune_params,minibatch_eval):
    mask=torch.ones(model_eval.num_classes,dtype=bool)
    import pdb; pdb.set_trace()
    # layers=[model_eval.classifier]
    layers=[model_eval.aggregators[1]]
    lassos=list()
    for layer in layers:
        if False: # first layer?
        for o in range(layer.order+1):
            print('optimizing {} order {}:'.format(layer._get_name(),o))
            handle=layer.f_lin[o].register_forward_hook(get_activation)
            evaluate_full_batch(model_eval,minibatch_eval,mode='val')
            handle.remove()
            feat=activation[0].cuda()
            weight=torch.transpose(layer.f_lin[o].weight,0,1).cuda()
            ref=torch.mm(feat,weight).detach()
            lassos.append(Lasso(weight.shape[0],weight,prune_params['beta_lmbd_1'],prune_params['beta_lmbd_2'],prune_params['beta_lmbd_1_step'],prune_params['beta_lmbd_2_step'],prune_params['beta_lr'],prune_params['weight_lr']))
            if args_global.gpu>=0:
                lassos[-1]=lassos[-1].cuda()
            train_nodes=minibatch.node_train
            beta_loss=list()
            weight_loss=list()
            for s in range(prune_params['total_step']):
                # optimize beta
                print('  optimizing beta ...')
                for e in range(prune_params['beta_epoch']):
                    np.random.shuffle(train_nodes)
                    batches=np.array_split(train_nodes,(int(feat.shape[0]/prune_params['beta_batch'])))
                    for batch in batches:
                        loss=lassos[-1].optimize_beta(feat[batch],ref[batch],mask)
                    print('    epoch {} loss: {}'.format(e,loss))
                    beta_loss.append(loss)
                    lassos[-1].lmbd_step()
                lassos[-1].clip_beta(prune_params['budget'])
                # optimize weight
                print('  optimizing weight ...')
                for e in range(prune_params['weight_epoch']):
                    np.random.shuffle(train_nodes)
                    batches=np.array_split(train_nodes,(int(feat.shape[0]/prune_params['weight_batch'])))
                    for batch in batches:
                        loss=lassos[-1].optimize_weight(feat[batch],ref[batch],mask)
                    print('    epoch {} loss: {}'.format(e,loss))
                    weight_loss.append(loss)
                lassos[-1].norm()
            lasso_plot(lassos[-1].beta.detach().cpu().numpy(),lassos[-1].weight.detach().cpu().numpy(),beta_loss,weight_loss,prune_params,'{} order {}:'.format(layer._get_name(),o))
            del feat
            del weight
            # import pdb; pdb.set_trace()

if __name__ == '__main__':
    train_params, train_phases, train_data, arch_gcn, prune_params = parse_n_prepare(args_global)
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
    get_model(train_phases ,train_params, model, minibatch, minibatch_eval, model_eval)
    prune(model_eval,prune_params,minibatch_eval)