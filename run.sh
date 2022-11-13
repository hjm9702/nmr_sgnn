


# for fold_seed in {0..2}
# do
#   for target in 13C 1H
#   do
    
#     # Proposed
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

#     # SG-IR
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode proposed_set2set --graph_representation sparsified

#     #SG-IMP
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode baseline --graph_representation sparsified  

#     # #SG-Only
#     # python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode baseline --graph_representation sparsified

#     # # #FCG
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 50 --node_hidden_dim 250 --readout_n_hidden_dim 500 --message_passing_mode baseline --readout_mode baseline --graph_representation fully_connected --feat_mode onehot

    
#   done
# done


for fold_seed in {0..9}
do
  for target in 13C 1H
  do
    
    # Proposed
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

    # SG-IR
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode proposed_set2set --graph_representation sparsified

    #SG-IMP
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode baseline --graph_representation sparsified  

    # #SG-Only
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode baseline --graph_representation sparsified

    # # #FCG
    #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 50 --node_hidden_dim 250 --readout_n_hidden_dim 500 --message_passing_mode baseline --readout_mode baseline --graph_representation fully_connected --feat_mode onehot

    
  done
done


for fold_seed in {0..9}
do
  for target in 13C 1H
  do
    
    # Proposed
    python make_all_preds.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

    
  done
done