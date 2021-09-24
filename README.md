# DVC_Intro

The repository is to showcase how to develop deep learning algorithms also considering data versioning and model versioning. The model architecture used is VGG 16 with pre-trained weights. The execution steps were migrated from TensorFlow to Pytorch. More about the Tensorflow execution can be found in this [link](https://github.com/iterative/example-versioning.git)

Additional modifications were done to save the accuracy of the each class (cats and dogs).

## Setup

* Create a virtual environment: `python3 -m venv .env`
* Activate the environment: `source .env/bin/activate`
* Install all the requirements : `pip install -r requirements.txt`

## Execution steps

* Download the data: `dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip`
* Unzip the data and remove the zip file: `unzip -q data.zip && rm -f data.zip`
* Register the data to DVC: `dvc add data`
* Train a model: `python train.py`
* Register the trained model to DVC: `dvc add checkpoints.pth`
* Commit all the changes and assign a version
* Download the next set of data: `dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip`
* Follow the steps mentioned above from the second point.

## Results observed

### First Iteration

The results observed for the first iteration are:

||train_acc|cats_train_acc|dogs_train_acc|test_acc|cats_test_acc|dogs_test_acc|
|----|----|----|----|----|----|----|
|0|95.57695595406967|96.50152348236993|94.65884451046051|97.25|98.5|96.0|

The logs of the training done are listed below:

```bash
Generating Model
Populating Data
 total_train_records:  1000
 total_test_records:  800
Starting training
EPOCH: 1
Loss=1.431853175163269 Batch_id=99 Accuracy=88.50 Cats Accuracy=89.20 Dogs Accuracy=87.80: 100%|███████████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.85it/s]

Test set: Average loss: 0.0350, Accuracy: 751/800 (93.88%) Cats Accuracy: 392/400 (98.00%) Dogs Accuracy: 359/400 (89.75%)

EPOCH: 2
Loss=0.006366188637912273 Batch_id=99 Accuracy=93.40 Cats Accuracy=93.00 Dogs Accuracy=93.80: 100%|████████████████████████████████████████████████████████████████████████████| 100/100 [00:19<00:00,  5.07it/s]

Test set: Average loss: 0.0279, Accuracy: 766/800 (95.75%) Cats Accuracy: 383/400 (95.75%) Dogs Accuracy: 383/400 (95.75%)

EPOCH: 3
Loss=0.8110159635543823 Batch_id=99 Accuracy=93.40 Cats Accuracy=93.20 Dogs Accuracy=93.60: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:19<00:00,  5.04it/s]

Test set: Average loss: 0.0267, Accuracy: 761/800 (95.12%) Cats Accuracy: 383/400 (95.75%) Dogs Accuracy: 378/400 (94.50%)

EPOCH: 4
Loss=0.00015035287651699036 Batch_id=99 Accuracy=92.00 Cats Accuracy=92.40 Dogs Accuracy=91.60: 100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:19<00:00,  5.03it/s]

Test set: Average loss: 0.0531, Accuracy: 755/800 (94.38%) Cats Accuracy: 396/400 (99.00%) Dogs Accuracy: 359/400 (89.75%)

EPOCH: 5
Loss=3.576278118089249e-08 Batch_id=99 Accuracy=93.00 Cats Accuracy=93.20 Dogs Accuracy=92.80: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.99it/s]

Test set: Average loss: 0.0353, Accuracy: 765/800 (95.62%) Cats Accuracy: 375/400 (93.75%) Dogs Accuracy: 390/400 (97.50%)

EPOCH: 6
Loss=7.152554815093026e-08 Batch_id=99 Accuracy=95.10 Cats Accuracy=94.40 Dogs Accuracy=95.80: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.96it/s]

Test set: Average loss: 0.0823, Accuracy: 736/800 (92.00%) Cats Accuracy: 398/400 (99.50%) Dogs Accuracy: 338/400 (84.50%)

EPOCH: 7
Loss=1.382818254569429e-06 Batch_id=99 Accuracy=95.00 Cats Accuracy=96.40 Dogs Accuracy=93.60: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.97it/s]

Test set: Average loss: 0.0241, Accuracy: 766/800 (95.75%) Cats Accuracy: 390/400 (97.50%) Dogs Accuracy: 376/400 (94.00%)

EPOCH: 8
Loss=0.0 Batch_id=99 Accuracy=95.50 Cats Accuracy=95.40 Dogs Accuracy=95.60: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  5.00it/s]

Test set: Average loss: 0.0312, Accuracy: 763/800 (95.38%) Cats Accuracy: 387/400 (96.75%) Dogs Accuracy: 376/400 (94.00%)

EPOCH: 9
Loss=4.1484063331154175e-06 Batch_id=99 Accuracy=95.30 Cats Accuracy=95.00 Dogs Accuracy=95.60: 100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:19<00:00,  5.01it/s]

Test set: Average loss: 0.0275, Accuracy: 766/800 (95.75%) Cats Accuracy: 389/400 (97.25%) Dogs Accuracy: 377/400 (94.25%)

EPOCH: 10
Loss=1.2568765878677368 Batch_id=99 Accuracy=96.40 Cats Accuracy=96.20 Dogs Accuracy=96.60: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:19<00:00,  5.01it/s]

Test set: Average loss: 0.0321, Accuracy: 764/800 (95.50%) Cats Accuracy: 389/400 (97.25%) Dogs Accuracy: 375/400 (93.75%)
```

### Second Iteration

The results observed for the second iteration are:

||train_acc|cats_train_acc|dogs_train_acc|test_acc|cats_test_acc|dogs_test_acc|
|----|----|----|----|----|----|----|
|0|95.89156880345085|96.5690201109141|95.10276442939261|95.375|96.5|94.25|

The logs of the training done are listed below:

```bash
Generating Model
Populating Data
 total_train_records:  2000
 total_test_records:  800
Starting training
EPOCH: 1
Loss=0.021523717790842056 Batch_id=199 Accuracy=89.10 Cats Accuracy=88.80 Dogs Accuracy=89.40: 100%|███████████████████████| 200/200 [00:50<00:00,  3.99it/s]

Test set: Average loss: 0.0459, Accuracy: 754/800 (94.25%) Cats Accuracy: 395/400 (98.75%) Dogs Accuracy: 359/400 (89.75%)

EPOCH: 2
Loss=1.5382236242294312 Batch_id=199 Accuracy=93.15 Cats Accuracy=93.30 Dogs Accuracy=93.00: 100%|█████████████████████████| 200/200 [00:39<00:00,  5.04it/s]

Test set: Average loss: 0.0895, Accuracy: 726/800 (90.75%) Cats Accuracy: 399/400 (99.75%) Dogs Accuracy: 327/400 (81.75%)

EPOCH: 3
Loss=0.11554104089736938 Batch_id=199 Accuracy=92.05 Cats Accuracy=91.90 Dogs Accuracy=92.20: 100%|████████████████████████| 200/200 [00:39<00:00,  5.05it/s]

Test set: Average loss: 0.0402, Accuracy: 761/800 (95.12%) Cats Accuracy: 391/400 (97.75%) Dogs Accuracy: 370/400 (92.50%)

EPOCH: 4
Loss=0.017473753541707993 Batch_id=199 Accuracy=94.35 Cats Accuracy=94.30 Dogs Accuracy=94.40: 100%|███████████████████████| 200/200 [00:39<00:00,  5.05it/s]

Test set: Average loss: 0.0833, Accuracy: 751/800 (93.88%) Cats Accuracy: 397/400 (99.25%) Dogs Accuracy: 354/400 (88.50%)

EPOCH: 5
Loss=0.0 Batch_id=199 Accuracy=94.35 Cats Accuracy=94.70 Dogs Accuracy=94.00: 100%|████████████████████████████████████████| 200/200 [00:40<00:00,  4.97it/s]

Test set: Average loss: 0.0458, Accuracy: 764/800 (95.50%) Cats Accuracy: 381/400 (95.25%) Dogs Accuracy: 383/400 (95.75%)

EPOCH: 6
Loss=2.741809908002324e-07 Batch_id=199 Accuracy=93.15 Cats Accuracy=92.90 Dogs Accuracy=93.40: 100%|██████████████████████| 200/200 [00:40<00:00,  4.99it/s]

Test set: Average loss: 0.0400, Accuracy: 767/800 (95.88%) Cats Accuracy: 394/400 (98.50%) Dogs Accuracy: 373/400 (93.25%)

EPOCH: 7
Loss=1.3708603382110596 Batch_id=199 Accuracy=94.00 Cats Accuracy=94.10 Dogs Accuracy=93.90: 100%|█████████████████████████| 200/200 [00:40<00:00,  4.96it/s]

Test set: Average loss: 0.0313, Accuracy: 771/800 (96.38%) Cats Accuracy: 392/400 (98.00%) Dogs Accuracy: 379/400 (94.75%)

EPOCH: 8
Loss=3.576278118089249e-08 Batch_id=199 Accuracy=94.05 Cats Accuracy=94.50 Dogs Accuracy=93.60: 100%|██████████████████████| 200/200 [00:40<00:00,  4.93it/s]

Test set: Average loss: 0.0313, Accuracy: 765/800 (95.62%) Cats Accuracy: 384/400 (96.00%) Dogs Accuracy: 381/400 (95.25%)

EPOCH: 9
Loss=1.5497201388825488e-07 Batch_id=199 Accuracy=94.60 Cats Accuracy=94.50 Dogs Accuracy=94.70: 100%|█████████████████████| 200/200 [00:40<00:00,  4.93it/s]

Test set: Average loss: 0.0326, Accuracy: 764/800 (95.50%) Cats Accuracy: 389/400 (97.25%) Dogs Accuracy: 375/400 (93.75%)

EPOCH: 10
Loss=7.247662324516568e-06 Batch_id=199 Accuracy=95.35 Cats Accuracy=95.50 Dogs Accuracy=95.20: 100%|██████████████████████| 200/200 [00:40<00:00,  4.94it/s]

Test set: Average loss: 0.0309, Accuracy: 763/800 (95.38%) Cats Accuracy: 386/400 (96.50%) Dogs Accuracy: 377/400 (94.25%)
```

## Contributors

* [Kaustubh Harapanahalli](mailto:kaustubhharapanahalli@gmail.com)
* [Monimoy Deb Purkayastha](mailto:monimoyd@gmail.com)
* [Rohin Sequeira](mailto:sequeira.rohin@gmail.com)
* [Soma Korada](mailto:somakorada@gmail.com)
