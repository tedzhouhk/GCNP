from graphsaint.globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
from tensorflow.python import pywrap_tensorflow

import torch
import time


def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
    When calculating the F1 score, we will mask the relevant root nodes.
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))
    node_val_test = minibatch.node_val if mode=='val' else minibatch.node_test
    f1_scores = calc_f1(to_numpy(labels[node_val_test]),to_numpy(preds[node_val_test]),model.sigmoid_loss)
    import pdb; pdb.set_trace()
    return loss, f1_scores[0], f1_scores[1]



def prepare(train_data,train_params,arch_gcn):
    adj_full,adj_train,feat_full,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    minibatch_eval=Minibatch(adj_full_norm,adj_train,role,train_params,cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval

def load_pytorch_from_tensorflow(model_eval,reader,arch_gcn):
    tf_aggr_index=1
    for order in arch_gcn['arch'].split('-'):
        for _order in range(int(order)+1):
            tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_weights'.format(tf_aggr_index,_order))
            model_eval.state_dict()['conv_layers.{}.f_lin.{}.weight'.format(tf_aggr_index-1,_order)].data.copy_(torch.from_numpy(tf_var.T))
            tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_bias'.format(tf_aggr_index,_order))
            model_eval.state_dict()['conv_layers.{}.f_bias.{}'.format(tf_aggr_index-1,_order)].data.copy_(torch.from_numpy(tf_var.T))
            if arch_gcn['bias']=='norm':
                tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_offset'.format(tf_aggr_index,_order))
                model_eval.state_dict()['conv_layers.{}.offset.{}'.format(tf_aggr_index-1,_order)].data.copy_(torch.from_numpy(tf_var).squeeze())
                tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_scale'.format(tf_aggr_index,_order))
                model_eval.state_dict()['conv_layers.{}.scale.{}'.format(tf_aggr_index-1,_order)].data.copy_(torch.from_numpy(tf_var).squeeze())
        tf_aggr_index+=1
    tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_weights'.format(tf_aggr_index,_order))
    model_eval.state_dict()['classifier.f_lin.{}.weight'.format(_order)].data.copy_(torch.from_numpy(tf_var.T))
    tf_var=reader.get_tensor('highorderaggregator_{}_vars/order{}_bias'.format(tf_aggr_index,_order))
    model_eval.state_dict()['classifier.f_bias.{}'.format(_order)].data.copy_(torch.from_numpy(tf_var.T))

if __name__ == '__main__':
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
    if not args_global.cpu_eval:
        minibatch_eval=minibatch
        model_eval=model
    reader=pywrap_tensorflow.NewCheckpointReader(args_global.saved_model_path)
    var_to_shape_map=reader.get_variable_to_shape_map()

    print("Loading model..")
    load_pytorch_from_tensorflow(model_eval,reader,arch_gcn)

    loss_val, f1mic_val, f1mac_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
    printf("Full validation : \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_val, f1mac_val), style='red')
    loss_test, f1mic_test, f1mac_test = evaluate_full_batch(model_eval, minibatch_eval, mode='test')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test, f1mac_test), style='red')
