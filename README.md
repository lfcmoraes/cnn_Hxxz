# Using Convolutional Neural Network to solve quantum-spin-chains Ground-States.
The project main idea is to find the the ground state of a quantum spin chain hamiltonian using deep learning. Deep learning is wildily used  as a tool to find abstract information in a lot of kind of problems, image pattern recognition are one of them. 

I've chosen an XXZ Hamiltonian with an uniform external magnect field. The Hamiltonian is in the for:  

---

![hamiltonian](https://latex.codecogs.com/gif.latex?H%28%5CDelta%29%3D-%5Cfrac%7BJ%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5E%7BL%7D%5Cleft%5B%5Csigma_j%5Ex%20%5Csigma_%7Bj&plus;1%7D%5Ex%20&plus;%20%5Csigma_j%5Ey%20%5Csigma_%7Bj&plus;1%7D%5Ey%20&plus;%5CDelta%5Csigma_j%5Ez%20%5Csigma_%7Bj&plus;1%7D%5Ez&plus;h%5Csigma_j%5Ez%5Cright%20%5D)

---
## Data set
![gs-energy_big](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/GS-Energy_big.png)

![gs-mag_big](https://github.com/lfcmoraes/cnn_Hxxz/blob/master/images/GS-Mag_big.png)

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
