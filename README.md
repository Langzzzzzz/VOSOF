# VOSOF

## Performance on Davis16 val set

| Method           | J&F-Mean | J-Mean | J-Recall | J-Decay | F-Mean | F-Recall | F-Decay |
|------------------|----------|--------|----------|---------|--------|----------|---------|
| Original         | 0.9264   | 0.9263 | 0.9846   | 0.0345  | 0.9265 | 0.9674   | 0.0286  |
| Argmax           | 0.9134   | 0.9118 | 0.9824   | 0.0331  | 0.9150 | 0.9606   | 0.0268  |
| Selective Update | 0.8754   | 0.8872 | 0.9773   | 0.0340  | 0.8636 | 0.9610   | 0.0290  |
| NN Approach      | 0.9215   | 0.9219 | 0.9829   | 0.0341  | 0.9211 | 0.9622   | 0.0255  |



## Requirements
- Python >= 3.6
- Pytorch 1.5
- Numpy
- Pillow
- opencv-python
- imgaug
- scipy
- tqdm
- pandas
- resnest

#### [DAVIS](https://davischallenge.org/)

#### Structure
```
 |- data
      |- Davis
          |- JPEGImages
          |- Annotations
          |- ImageSets
```

## Demo
```
python demo.py -g "gpu id" -s "set" -y "year" -D "path to davis" -p "path to weights" -backbone "[resnet50,resnet18,resnest101]"
#e.g.
python demo.py -g 0 -s val -y 16
```

## Evaluation
Evaluating on Davis 2016 val set.
```
python eval.py -g "gpu id" -s "set" -y "year" -D "path to davis" -p "path to weights" -backbone "[resnet50,resnet18,resnest101]"
#e.g.
python eval.py -g 0 -s val -y 16
```

## Citing STM
```
@inproceedings{oh2019video,
  title={Video object segmentation using space-time memory networks},
  author={Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9226--9235},
  year={2019}
}
@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}
```
