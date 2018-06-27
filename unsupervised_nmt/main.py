import itertools

import tensorflow as tf

from .nn_helper import LanguageEncoderDecoder as LanEncDec

pretrain_graph = tf.Graph()
backtran_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

with pretrain_graph.as_default():
    pretrain_iterator = ...
    pretrain_model = BuildPretrainModel(pretrain_iterator)
    initializer = tf.global_variables_initializer()

with backtran_graph.as_default():
    backtran_iterator = ...
    backtran_model = BuildBackTranModel(backtran_iterator)


with eval_graph.as_default():
    eval_iterator = ...
    eval_model = BuildEvalModel(eval_iterator)

with infer_graph.as_default():
    infer_iterator, infer_inputs = ...
    infer_model = BuildInferenceModel(infer_iterator)

checkpoints_path = "/tmp/model/checkpoints"

pretrain = tf.Session(graph=pretrain_graph)
backtran = tf.Session(graph=backtran_graph)
eval_sess = tf.Session(graph=eval_graph)
infer_sess = tf.Session(graph=infer_graph)

pretrain.run(initializer)
pretrain.run(pretrain_iterator.initializer)

for i in itertools.count():

    train_model.train(pretrain)

    if i % EVAL_STEPS == 0:
        checkpoint_path = train_model.saver.save(pretrain, checkpoints_path, global_step=i)
        eval_model.saver.restore(eval_sess, checkpoint_path)
        eval_sess.run(eval_iterator.initializer)
    while data_to_eval:
        eval_model.eval(eval_sess)

    if i % INFER_STEPS == 0:
        checkpoint_path = train_model.saver.save(pretrain, checkpoints_path, global_step=i)
        infer_model.saver.restore(infer_sess, checkpoint_path)
        infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
    while data_to_infer:
        infer_model.infer(infer_sess)