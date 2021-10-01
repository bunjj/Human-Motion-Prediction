# Human Motion Prediction

## Intro
Project Work for MSc Machine Perception course offered at ETH Zurich (263-3710-00L) to predict human motion.
For details on implementation, algorithms used and evaluation check out **machine_perception_report.pdf**.



### How to reproduce our results
We used the virtual environment as described in the project description. We did not install any further packages. 
The setup corresponds to the one in the project.


The following command is used to train our final model with the according parameters and evaluates it on the validation set at the end of training. 

```
python train.py --model dct_att_gcn --n_epochs 1000 --lr 0.0005 --use_lr_decay --lr_decay_step 330 --bs_train 128 --bs_eval 128 --nr_dct_dim 64 --loss_type avg_l1 --lr_decay_rate 0.98 --opt adam --kernel_size 40 --clip_gradient --max_norm 1
```

We ran the command on the leonhard cluster with following configuration: `bsub -n 6 -W 3:00 -o dct_att_gcn_hype -R "rusage[mem=1024, ngpus_excl_p=1]"`


The resulting folder contains the predictions for the test set and a `config.json` which holds the mean joint angle values of the validation set evaluation. 
