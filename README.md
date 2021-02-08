The following instructions are for running the examples seen in "PAC-BUS: Meta-Learning Bounds via PAC-Bayes and Uniform Stability".

# PAC-BUS:

This code uses the following:
- PyTorch ≥ 1.4.0
- learn2learn 
- cvxpy
- Mosek ≥ 9.0
    - An academic license can be acquired here: https://www.mosek.com/products/academic-licenses/
- sklearn
- h5py

If you are using Anaconda, you can run the following commands to install all of the necessary packages. 
```
conda create -n pacbus
conda activate pacbus
conda install pytorch=1.4.0 -c pytorch
pip install learn2learn cvxpy Mosek sklearn h5py
```

### Circle Classification Example: 
Run the following to produce the results from Example 1 in the paper. Note that `--num_val` denotes the number of times we run the resulting
policy on test data. We require a large number of evaluations to produce a tight upper bound (see the end of Section 
4.3 for more information). For testing purposes, you may want to reduce `--num_val` so the program takes less time to finish. 

```
python circleclass_main.py --method maml    --prior train --trials full --verbose True 
python circleclass_main.py --method mlap    --prior train --trials full --verbose True
python circleclass_main.py --method mr_maml --prior train --trials full --num_val 20000 --verbose True
python circleclass_main.py --method pac_bus --prior train --trials full --num_val 20000 --verbose True
```
### Mini-Wiki Example: 
Download
- text_embedding: Clone into `./` from github: https://github.com/NLPrinceton/text_embedding
- glove.6B files: Download from https://nlp.stanford.edu/projects/glove/ and place into `./glove/`
- Mini-Wiki dataset: Download from https://github.com/mkhodak/FMRL/blob/master/data/miniwikipedia.tar.gz, unzip, and place folder `raw` 
into `./data/miniwiki/`.
  
Run the following to generate the dataset:
```
python data_generators/miniwiki_data.py
```
Run the following to produce the results from Example 2 in the paper.
```
python miniwiki_main.py --method maml      --prior train --trials full --verbose True
python miniwiki_main.py --method fli_batch --prior train --trials full --verbose True
python miniwiki_main.py --method mr_maml   --prior train --trials full --num_val 20000 --verbose True
python miniwiki_main.py --method pac_bus   --prior train --trials full --num_val 20000 --verbose True
```

### NME Omniglot Example: 

Run the following to produce the results from Example 3 in the paper for `--seed 1` through `5`. This will automatically download the omniglot dataset if you do not have it.
A gpu is recommended, but you may specify option `--gpu -1` to use the cpu for all computations.
```
python omniglot_main.py --method maml       --k_spt 1 --k_qry 4 --batch 16 --nme True --epochsm 100000 --lrm 0.005 --lrb 0.1 --seed 1
python omniglot_main.py --method maml       --k_spt 5 --k_qry 5 --batch 16 --nme True --epochsm 100000 --lrm 0.005 --lrb 0.1 --seed 1

python omniglot_main.py --method fli_online --k_spt 1 --k_qry 4 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --seed 1  
python omniglot_main.py --method fli_online --k_spt 5 --k_qry 5 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --seed 1

python omniglot_main.py --method mr_maml_w  --k_spt 1 --k_qry 4 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --lrb 0.5 --regscale 2e-7 --seed 1
python omniglot_main.py --method mr_maml_w  --k_spt 5 --k_qry 5 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --lrb 0.5 --regscale 2e-7 --seed 1

python omniglot_main.py --method pac_bus_h  --k_spt 1 --k_qry 4 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --lrb 0.5 --regscale 1e-3 --regscale2 10.0 --seed 1
python omniglot_main.py --method pac_bus_h  --k_spt 5 --k_qry 5 --batch 16 --nme True --epochsm 100000 --lrm 0.001 --lrb 0.5 --regscale 1e-4 --regscale2 10.0 --seed 1
```
### References
Sebastien Arnold, Praateek Mahajan, Debajyoti Datta, Ian Bunner, and Konstantinos Saitas Zarkias. learn2learn: A Library for Meta-Learning Research. *arXiv preprint arXiv:2008.12284*, 2020.


Mikhail Khodak,  Maria-Florina Balcan,  and Ameet Tal-walkar.  Provable Guarantees for Gradient-Based Meta-Learning. *Proceedings of the 36th International Confer-ence on Machine Learning*, 2019.

Liangqu Long 2018. MAML-Pytorch Implementation. Github|https://github.com/dragen1860/MAML-Pytorch

Liangqu Long 2018. MAML-Pytorch Implementation. Github|https://github.com/dragen1860/Reptile-Pytorch

