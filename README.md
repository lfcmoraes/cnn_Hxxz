# Using Convolutional Neural Network to solve quantum-spin-chains Ground-States.
The project main idea is to find the the ground state of a quantum spin chain hamiltonian using deep learning. Deep learning is wildily used  as a tool to find abstract information in a lot of kind of problems, image pattern recognition are one of them. An image is nothing more than a tensor of numbers, in that sense an Hamiltonian is just a single chanel image and deep learning can be used to predict non explicit characteristics of the same, like its ground-state throught a regression method.

I've chosen an XXZ Hamiltonian with an uniform external magnect field. The Hamiltonian is in the for:  

![hamiltonian](https://latex.codecogs.com/gif.latex?H%28%5CDelta%29%3D-%5Cfrac%7BJ%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5E%7BL%7D%5Cleft%5B%5Csigma_j%5Ex%20%5Csigma_%7Bj&plus;1%7D%5Ex%20&plus;%20%5Csigma_j%5Ey%20%5Csigma_%7Bj&plus;1%7D%5Ey%20&plus;%5CDelta%5Csigma_j%5Ez%20%5Csigma_%7Bj&plus;1%7D%5Ez&plus;h%5Csigma_j%5Ez%5Cright%20%5D)

---
## Data set

The number of spins of the chain was fixed, ![L=8](https://latex.codecogs.com/gif.latex?L%20%3D%208), and de anisotropy constant, ![\Delta](https://latex.codecogs.com/gif.latex?%5CDelta%20%3D%20-1), was set to garantee the paramagnect regime. To generete the Hamiltonians the magnect fiel was started at ![-6](https://latex.codecogs.com/gif.latex?-6), the matrix is block diagonalized and the minimum of all of the block are stored (![sigmaz](https://latex.codecogs.com/gif.latex?%5Csigma%5Ez) commutes with the Hamiltonian so the last can be written in magnatizations zones). After that the magnect field is incread by ![0.001](https://latex.codecogs.com/gif.latex?0.001). Doing this until the magnect field is ![6](https://latex.codecogs.com/gif.latex?6) so we get ![12k](https://latex.codecogs.com/gif.latex?12000) samples.

The ground state engergy density (![E/L](https://latex.codecogs.com/gif.latex?E_%7BGS%7D/L))  and the magnetization of the data set in function of the external magnect field (![h](https://latex.codecogs.com/gif.latex?h)) are represente in the graphics bellow respectivaly: 

![gs-energy_big](https://user-images.githubusercontent.com/31672811/49387499-1ee6ab00-f709-11e8-9ee4-6c57872df464.png)

![gs-mag_big](https://user-images.githubusercontent.com/31672811/49387539-39b91f80-f709-11e8-839c-84839fa19132.png)

### Create the data set
To create the Hamiltonians, save then in a folder named `matrix/` as  numpy arrays (`h.npy`) and create de graphics above simply run the following command:
```python
python3 data_set.py
```

---
### Code
| Used to | # of samples | % |
| --- | --- | --- |
| Train | 9120 | 76 |
| Validate| 2280 | 19 |
| Test | 600 | 5 |
| Total | 12000 | 100 | 

---
## Results

![loss](https://user-images.githubusercontent.com/31672811/49387540-39b91f80-f709-11e8-8bbf-dd95a546c33a.png)
![predct](https://user-images.githubusercontent.com/31672811/49387542-3a51b600-f709-11e8-8bc2-a83bfe55d616.png)
### Random
#### Loss
##### Retrain 500, 1k and 5k:
![loss_retrain_rand_500](https://user-images.githubusercontent.com/31672811/52968139-18241780-338b-11e9-8be8-fcb5428d25b1.png)
![loss_retrain_rand_1k](https://user-images.githubusercontent.com/31672811/52968136-16f2ea80-338b-11e9-999c-781c4fe17006.png)
![loss_retrain_rand_5k](https://user-images.githubusercontent.com/31672811/52968137-178b8100-338b-11e9-895b-cd535eed2687.png)

##### Train 1k and 10k:
![loss_train_rand_1k](https://user-images.githubusercontent.com/31672811/52968140-18241780-338b-11e9-90ce-90b8b15417f5.png)
![loss__train_rand_10k](https://user-images.githubusercontent.com/31672811/53109491-c52b9b00-3517-11e9-909f-4ab6628e9d95.png)

#### Predictions
##### Just test:
![pred_rand_100](https://user-images.githubusercontent.com/31672811/52970549-efa01b80-3392-11e9-95bc-57e4b861b625.png)
![just_test_rand](https://user-images.githubusercontent.com/31672811/54211367-f1578d80-44bf-11e9-83f7-ef3fd6c2d3f0.png)
##### Retrain 500, 1k and 5k:
![retrain_rand_500](https://user-images.githubusercontent.com/31672811/54211368-f1578d80-44bf-11e9-8956-6a9f026a507e.png)
![retrain_rand_1k](https://user-images.githubusercontent.com/31672811/54212215-5c559400-44c1-11e9-9109-58bf557c4d86.png)
![retrain_rand_5k](https://user-images.githubusercontent.com/31672811/54212127-38924e00-44c1-11e9-8788-98ca66cfa2a8.png)

##### Train 1k and 10k:
![train_rand_1k](https://user-images.githubusercontent.com/31672811/54213385-10a3ea00-44c3-11e9-95f3-2277d3929e72.png)
![train_rand_10k](https://user-images.githubusercontent.com/31672811/54211373-f1f02400-44bf-11e9-9391-c50abf94d70a.png)

##### Train rand 25k
![train_rand_25k](https://user-images.githubusercontent.com/31672811/54612975-b3ff7c80-4a38-11e9-8366-fdbef567c2f7.png)

---
### Pool


#### Loss
##### Retrain 500, 1k and 10k:
![loss_retrain_pool_500](https://user-images.githubusercontent.com/31672811/52968172-2f630500-338b-11e9-856d-c781de57dd5d.png)
![loss_retrain_pool_1k](https://user-images.githubusercontent.com/31672811/52968170-2eca6e80-338b-11e9-8b30-73efbfa868cd.png)
![loss_retrain_pool_5k](https://user-images.githubusercontent.com/31672811/52968171-2eca6e80-338b-11e9-9cc7-b3ea93d782d5.png)

##### Train 1k and 10k:
![loss_train_pool_1k](https://user-images.githubusercontent.com/31672811/52968174-2f630500-338b-11e9-9bb7-3fc4ef2bd51a.png)
![loss_train_pool_10k](https://user-images.githubusercontent.com/31672811/53256914-c4297380-36a7-11e9-94f9-a4f9c26b117d.png)

#### Predictions
##### Just test:
![just_test_pool](https://user-images.githubusercontent.com/31672811/54211366-f1578d80-44bf-11e9-9d94-3208d2cf0a95.png)

##### Retrain 500, 1k and 5k:
![retrain_pool_500](https://user-images.githubusercontent.com/31672811/54214766-9aed4d80-44c5-11e9-903d-d07877c049ab.png)

![retrain_pool_1k](https://user-images.githubusercontent.com/31672811/54214765-9a54b700-44c5-11e9-8988-f8ce80f7e3f4.png)
![retrain_pool_5k](https://user-images.githubusercontent.com/31672811/54214388-e6ebc280-44c4-11e9-8b74-3ded888dcf30.png)

##### Train 1k and 10k:
![train_pool_1k](https://user-images.githubusercontent.com/31672811/54214259-ac822580-44c4-11e9-819b-736699595380.png)
![train_pool_10k](https://user-images.githubusercontent.com/31672811/54211371-f1f02400-44bf-11e9-93cb-62b6304e2c28.png)

---

#### Pool L = 16
##### Loss:



##### Prediction:
![train_l16_10k](https://user-images.githubusercontent.com/31672811/54211370-f1578d80-44bf-11e9-83d2-c2e1dc7c92ab.png)

#### Pool l=16 20k


##### Loss:
![loss_train_l16_20k](https://user-images.githubusercontent.com/31672811/55175872-e3139d80-515e-11e9-96e9-218e4f9ad144.png)
![loss_train_l16_20k_old](https://user-images.githubusercontent.com/31672811/55175936-fd4d7b80-515e-11e9-9f25-936fb3e91b6c.png)
##### Prediction:
![train_l16_20k](https://user-images.githubusercontent.com/31672811/55175510-36392080-515e-11e9-8cc9-e79aa566b084.png)






#### Delta x Negativity

![train_Del_neg](https://user-images.githubusercontent.com/31672811/56671698-758a5c80-668b-11e9-8c05-98308f03b983.png)
![train_Del_neg_cont](https://user-images.githubusercontent.com/31672811/56671699-758a5c80-668b-11e9-97de-18b47f9d39f2.png)
![train_Del_neg_cont_10percent_test](https://user-images.githubusercontent.com/31672811/56671701-758a5c80-668b-11e9-8d12-0382a35d6257.png)
![train_Del_neg_cont_10percent](https://user-images.githubusercontent.com/31672811/56672671-38bf6500-668d-11e9-90f5-8ddaf5e6fcb7.png)

