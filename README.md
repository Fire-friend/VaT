# VaT (ECCV 2024)
Code for Unsupervised Variational Translator for Bridging Image Restoration and High-Level Vision Tasks.

----

## Abstract

Recent research tries to extend image restoration capabilities from human perception to machine perception, thereby enhancing the performance of high-level vision tasks in degraded environments. These methods, primarily based on supervised learning, typically involve the retraining of restoration networks or high-level vision networks. However, collecting paired data in real-world scenarios and retraining large-scale models are challenge. To this end, we propose an unsupervised learning method called Variational Translator (VaT), which does not require retraining existing restoration and high-level vision networks. Instead, it establishes a lightweight network that serves as an intermediate bridge between them. By variational inference, VaT approximates the joint distribution of restoration output and high-level vision input, dividing the optimization objective into preserving content and maximizing marginal likelihood associated with high-level vision tasks. By cleverly leveraging self-training paradigms, VaT achieves the above optimization objective without requiring labels. As a result, the translated images maintain a close resemblance to their original content while also demonstrating exceptional performance on high-level vision tasks. Extensive experiments in dehazing and low-light enhancement for detection and classification show the superiority of our method over other state-of-the-art unsupervised counterparts, even significantly surpassing supervised methods in some complex real-world scenarios.

## Training

#### Environment

```
pip install -r requirement.txt
```

#### Obtaining NUQ features 

```
python getNUQ_f.py
```

#### VaT Training 

```
python -m visdom.server
python main.py
```

Training data can be found on [Baidu network disk](https://pan.baidu.com/s/1g7B3PktRzU2xkxHf5N04-A) (pw: rhqs )\. It may take some time to modify the path in the code. 

The pre-trained object detection model follows the official code completely and is trained on clean datasets.

Since our method is unpaired, if you want to achieve better real-world detection performance, you can merge synthetic low-quality images and real low-quality images into the trainA folder.

## Testing

The [pretrained weight](https://pan.baidu.com/s/1RKTj2iuon5M_f3EItSCmfA)(pw: ggbu ) was trained for 17 epochs, continuing training might yield better results.

```
python val.py 
```

### Statement

If you are interested in our work, please consider citing the following:

```
@inproceedings{Wu2025VaT,
  title={Unsupervised Variational Translator for Bridging Image Restoration and High-Level Vision Tasks},
  author={Wu, Jiawei
and Jin, Zhi},
  booktitle={European Conference on Computer Vision},
  pages={214--231},
  year={2025},
  organization={Springer}
}
```
