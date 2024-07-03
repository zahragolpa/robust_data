# Experiments on "Characterizing the Impacts of Instances on Robustness"
In this repo, I take a paper from the literature that introduces a method for improving the robustness of NLP models and experiment with extending that method to vision models. 

## Paper information

- paper: https://aclanthology.org/2023.findings-acl.146.pdf
- Authors: Rui Zheng*, Zhiheng Xi*, Qin Liu, Wenbin Lai, Tao Gui, Qi Zhang, Xuanjing Huang, Jin Ma, Ying Shan, Weifeng Ge.
- Abstract: Building robust deep neural networks (DNNs) against adversarial attacks is an important but challenging task. Previous defense approaches mainly focus on developing new model structures or training algorithms, but they do little to tap the potential of training instances, especially instances with robust patterns carrying innate robustness. In this paper, we show that robust and non-robust instances in the training dataset, though are both important for test performance, have contrary impacts on robustness, which makes it possible to build a highly robust model by leveraging the training dataset in a more effective way. We propose a new method that can distinguish robust instances from nonrobust ones according to the model’s sensitivity to perturbations on individual instances during training. Surprisingly, we find that the model under standard training easily overfits the robust instances by relying on their simple patterns before the model completely learns their robust features. Finally, we propose a new mitigation algorithm to further release the potential of robust instances. Experimental results show that proper use of robust instances in the original dataset is a new line to achieve highly robust models. Our codes are publicly available at https://github.com/ruizheng20/robust_data.
  
  
### Usage

Here's the basic usage of the code found at [robust_data](https://github.com/ruizheng20/robust_data):

- Collect robust statistics of training dataset
```shell script
python data_statistics.py
```

- The data is saved to the following path (see in `data_statistics.py`):
```python
np.save('robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
    .format(args.model_name,
            args.dataset_name,
            args.task_name,
            args.seed,
            args.do_train_shuffle,
            args.dataset_len,
            args.adv_steps,
            args.adv_lr,args.epochs,
            args.lr,
            args.statistic_interval,
            args.with_untrained_model,
            args.use_cur_preds
            ),
    robust_statistics_dict)
```

- Draw plots to show data robustness (remember to set your statistic path in the file)
```shell script
cd plot_utils
python plotting.py
```

- Run Flooding method with robust data
```shell script
cd new_fine_tune_flooding
sh run_sst2_new_finetune_flooding.sh
```

- Run Soft Label method with robust data
```shell script
cd new_fine_tune_flooding
sh run_sst2_new_finetune_soft_label.sh
```

## Plot & Performance
The following plots show the performance of the method introduced in the paper.

- Robust Data Map

<img src="https://spring-security.oss-cn-beijing.aliyuncs.com/img/image-20230726195217008.png" alt="image-20230726195217008" style="zoom:50%;" />

- Final Performance

<img src="https://spring-security.oss-cn-beijing.aliyuncs.com/img/image-20230726195246173.png" alt="image-20230726195246173" style="zoom:50%;" />

## Contributions

- Added image perturbations.
- Added image dataloaders and support for cifar10 and food-101 datasets.
- Improved training speed by adding support for distributed training.
- Integrating vision transformers and torchvision models.

## Results

We tested different settings with cifar10 dataset and vision transformers using a various range of transformations. While there were signs that robust images were being separated from non-robust images, the difference was not significant enough to draw a conclusion. We also applied the flooding method regardless of the insignificant difference between the samples but didn't achieve good results.

Here's an example plot for cifar10 dataset when Blur transformation has been applied:
![image](https://github.com/zahragolpa/robust_data/assets/20627999/de35554b-8808-4b48-a845-91b462f7f9ee)

Note that different perturbations resulted in different patterns. Here are a few other examples of Rotate, Shot noise, and Red shift transformations:



Rotate:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/23d5740c-c21c-4bb4-bd93-c13108099ca6)




Shot noise:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/1c9f6107-a196-40ad-9336-698adc1a191f)




Red shift:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/53bc61ff-e4bc-45c9-9280-b9f4616dd537)




We also visualized the robust, swing, and non-robust samples:

robust samples:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/5f39782c-3ada-4bbb-a838-820f4b68c899)



swing samples:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/98f2a091-543e-4aa4-a7d0-9116e9e59034)



non-robust samples:


![image](https://github.com/zahragolpa/robust_data/assets/20627999/6813b263-85dd-4ee9-b20d-f65cef6d257a)



Unlike the original paper, training on the 50% most robust samples did not result in over-fitting, hence, their work was not reproduced on image data:

Training on cifar10 with Blur transformation, train loss:


<img width="731" alt="image" src="https://github.com/zahragolpa/robust_data/assets/20627999/204f26ff-628c-43d4-9da9-b47324e36ed2">




Training on cifar10 with Blur transformation, test loss:

<img width="709" alt="image" src="https://github.com/zahragolpa/robust_data/assets/20627999/b397455d-6c90-4152-9b42-097d99192f55">



We also tried adversarial attacks instead of image perturbations and experimented with a variety of transformation intensities (weak, medium, strong), but none of them reproduced the results from the paper.

## Conclusion
The method introduced by Zheng et al. did not apply to image data successfully. However, the idea of attributing robustness to samples is still an interesting one. Future work can be directed in investigating other approaches to instance-based robustness improvement in image data.


## Citation

```
@inproceedings{zheng2023characterizing,
  title={Characterizing the Impacts of Instances on Robustness},
  author={Zheng, Rui and Xi, Zhiheng and Liu, Qin and Lai, Wenbin and Gui, Tao and Zhang, Qi and Huang, Xuan-Jing and Ma, Jin and Shan, Ying and Ge, Weifeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={2314--2332},
  year={2023}
}
```
