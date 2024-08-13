
# 2024 ETRI Competition
___

## Human Understanding AI Paper Challenge 2024
> **Competition** , **May. 2024 ~ Jun. 2024**

---

## Software Stacks
![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


---

## Competition Topic

- **Sleep Quality Estimation From Lifelog Data**
- 

---

## Implementation

### 1. Dataset
- [ETRI LIFELOG DATASET](https://arxiv.org/abs/2403.16509) used in this study consists of nine time series of data collected from eight different participants 

### 2. Data preprocessing
- **Feature Selection and Synchronization** : we selected only three types of data ”mLight, wLight, and wHr” from the nine types of data to train the model.
- we extracted a total of 24 hours time-sequence period from 9 am to 9 am on the previous day for Question type - Q, and 12 hours time-sequence period from 9 pm to 9 am on the previous day for Question type - S, and used them for learning and inference.
  
  ![image](https://github.com/user-attachments/assets/755513d2-a7a1-4d94-af92-662c9bae094c)

- **Normalizing** : we first used logarithmic operation on the mLight and wLight data to reduce the variation of the data. and we used Min-max scaling.

### 3. Designing DL Model
- we used [ConvTran](https://arxiv.org/pdf/2305.16642v1) as timeseries data encoder
![model_design_2](https://github.com/user-attachments/assets/1fb8274c-a440-49d0-9244-155b5ba2a632)

### 4. Training DL Model
- Weighted Validation Loss Propagation

![image](https://github.com/user-attachments/assets/1e4fc15b-a2dc-4519-a0ab-392ed7b0a7ac)

- Set Lower Bound

![image](https://github.com/user-attachments/assets/59ed4a98-aa21-4e78-b059-87cbb6b4c892)

---

## Result & Outputs

- Leader Board (Public Score 5.5575562) : **15th** / 40 teams

 ![image](https://github.com/user-attachments/assets/dbb46b9b-c9e7-4bf9-8d91-40ed2b61a8af)

- **Publication conference paper(accepted)** in The 15th International Conference on ICT Convergence (Oct. 2024)

 
