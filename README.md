# VOSOF

## Performance on Davis17 val set&Weights

|  | backbone |  training stage | training dataset | J&F | J |  F  | weights |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Ours| resnet-50 |  stage 1 | MS-COCO | 69.5 | 67.8 | 71.2 | [`link`](https://drive.google.com/file/d/1dHiKCOmTGhccG24UJuPYfLl8NcNqN6eC/view?usp=sharing) |
| Origin | resnet-50 | stage 2 | MS-COCO -> Davis&Youtube-vos | 81.8 | 79.2 | 84.3 | [`link`](https://github.com/seoungwugoh/STM) |
| Ours| resnet-50 | stage 2 | MS-COCO -> Davis&Youtube-vos | 82.0 | 79.7 | 84.4 | [`link`](https://drive.google.com/file/d/1M8NesOwct00QftL_bc-Nh_Qn7TgoZFX-/view?usp=sharing) |
| Ours | resnest-101 | stage 2| MS-COCO -> Davis&Youtube-vos | 84.6 | 82.0 | 87.2 | [`link`](https://drive.google.com/file/d/1jQAfCXVSyu2b4DvHeFErCQwP6CKYeJ34/view?usp=sharing)|


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
```
