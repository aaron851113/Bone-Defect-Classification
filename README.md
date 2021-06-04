# Bone-Defect-Classification

### Split Data
- Data
	- Fine (1332x1330)
	- Broke (1332x1330)

using ([augmentation.ipynb](./augmentation.ipynb)) <br />
then data will be splited as follows:
- Train
	- Fine (224x224)
	- Broke (224x224)
- Test
	- Fine (224x224)
	- Broke (224x224)

### Train model 

```
$ python train.py --model=[**selet_model**]
```
select_model = [resnet18,convnet] <br />

then model weight will save in **weights** folder <br />
