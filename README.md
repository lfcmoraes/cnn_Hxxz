# Using Convolutional Neural Network to solve quantum-spin-chains Ground-States.
The project main idea is to find the the ground state of a quantum spin chain hamiltonian using deep learning. Deep learning is wildily used  as a tool to find abstract information in a lot of kind of problems, image pattern recognition are one of them. An image is nothing more than a tensor of numbers, in that sense an Hamiltonian is just a single chanel image and deep learning can be used to predict non explicit characteristics of the same, like its ground-state throught a regression method.

I've chosen an XXZ Hamiltonian with an uniform external magnect field. The Hamiltonian is in the for:  

![hamiltonian](https://latex.codecogs.com/gif.latex?H%28%5CDelta%29%3D-%5Cfrac%7BJ%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5E%7BL%7D%5Cleft%5B%5Csigma_j%5Ex%20%5Csigma_%7Bj&plus;1%7D%5Ex%20&plus;%20%5Csigma_j%5Ey%20%5Csigma_%7Bj&plus;1%7D%5Ey%20&plus;%5CDelta%5Csigma_j%5Ez%20%5Csigma_%7Bj&plus;1%7D%5Ez&plus;h%5Csigma_j%5Ez%5Cright%20%5D)

---
## Data set

The number of spins of the chain was fixed, ![L=8](https://latex.codecogs.com/gif.latex?L%20%3D%208), and de anisotropy constant, ![\Delta](https://latex.codecogs.com/gif.latex?%5CDelta%20%3D%20-1), was set to garantee the paramagnect regime. The ground state engergy density (![E/L](https://latex.codecogs.com/gif.latex?E_%7BGS%7D/L))  and the magnetization of the data set in function of the external magnect field (![h](https://latex.codecogs.com/gif.latex?h)) are represente in the graphics bellow respectivaly: 

![gs-energy_big](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/GS-Energy_big.png)

![gs-mag_big](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/GS-Mag_big.png)

### Create the data set
To create the Hamiltonians and save then in a folder named `matrix/` as  numpy arrays (`h.npy`) smply run the following command:
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
![loss](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/loss.png)
![predct](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/predct.png)
