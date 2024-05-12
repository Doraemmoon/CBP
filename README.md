# Contents of this repository

* Environment information, source code of CBP.
* Hyper-parameters of CBP on the TaFeng dataset.



# CBP

This code contains Pytorch implementation of CBP:

> CBP employs counterfactual inference to exclusively estimate the impact of bias and personalization, especially in the context of repurchasing behavior. This helps balance the reduction of bias with the enhancement of individual user preferences in next-basket prediction.



## Environments

torch 1.10.1+cuda 11.2.
python 3.6.13.
numpy 1.23.4.
scipy 1.5.4.
scikit-learn 0.23.2.
RTX3090.

We suggest you create a new environment with: conda create -n CBP python=3.6



## Dataset

* Tafeng: https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset



## Running the CBP

create folder ./src/all_results/TaFeng

```
cd src
python main.py --dataset TaFeng --lr 0.01 --l2 0.00001 --alpha 0.1 --beta 0.2 --batch_size 100 --dim 32
```

