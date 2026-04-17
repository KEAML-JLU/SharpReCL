# SharpReCL
The source code for "Simple-Sampling and Hard-Mixup with Prototypes to Rebalance Contrastive Learning for Text Classification"

# Usage
You can run the following command. Note that if you meet the CUDA out of memory issue, you can use a small batch size adding a argument such as ``--batch_size 32``, but this may slightly affect the model performance. 
```
cd SharpReCL
```
```
python train.py --dataset ohsumed
```

# Cite
If you find our work can help your research, please cite our work! <br>

```
@inproceedings{li2024simple,
  title={Simple-sampling and hard-mixup with prototypes to rebalance contrastive learning for text classification},
  author={Li, Mengyu and Liu, Yonghao and Giunchiglia, Fausto and Li, Ximing and Feng, Xiaoyue and Guan, Renchu},
  booktitle={WWW},
  year={2026}
}
```

# Contact
If you have any question, feel free to contact via [email](mailto:yonghao20@mails.jlu.edu.cn).
