![Multiple Server](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/654f6670-1240-47ba-b05d-8a144c527ae9)
## Introduction
- This repository explores the concept of federated caching and training in a network with multiple base stations. 
- Federated caching involves storing frequently accessed data at base stations to reduce the need to fetch it from a central server whenever requested. The primary goal is to maximize the time-averaged cache hit rate, considering constraints such as fetching cost and a fixed cache size.
- In the context of federated learning, multiple devices associated with different base stations collaborate to train a shared model. Local updates are sent to a central server for aggregation into a global model without sharing raw data. This setup aims to preserve user privacy while enabling collaborative model training.
- We implemented Federated Averaging (FedAvg) with the Drift Plus Penalty (DPP) cache algorithm within a Flask-based federated learning system. The architecture comprises one central server, a distributed network with two base stations, and multiple users.
- Dataset for fedAvg Description: For fedAvg, we employ the same dataset that was utilized in DPP-cache, [“311 Service Requests Pitt” from Kaggle](https://www.kaggle.com/datasets/yoghurtpatil/311-service-requests-pitt). We evenly distribute this dataset to both base stations for FedAvg.

- This project is a part of our work on the Drift Plus Penalty Caching Algorithm. For more details refer to this [GitHub Repo](https://github.com/SuyashGaurav/DPP-Cache-Main-Flask-Implementation).

 ### <img src="https://i.pinimg.com/originals/3a/36/20/3a36206f35352b4230d5fc9f17fcea92.png" width="20" border-radius="10">  Youtube Video Link: https://www.youtube.com/watch?v=uIn7cSWpbto

## Pictorial representation
![Multiple Server (1)](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/238375b3-6f41-4b35-a843-e66a64f960e5)
![WhatsApp Image 2023-12-24 at 11 36 29 PM](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/6a49ed63-6dcd-4887-9d2c-64e2790e8071)


## Performance Evaluation of FedAvg within the Context of DPP-Cache:
- We present the performance results of FedAvg integrated with DPP-Cache. Figs. show the
plots of cache hit and download rate as a function of time slot for two distinct base stations,
namely Base Station 1 and Base Station 2 when using both DPP-Cache and FedAvg within the framework of DPP-Cache. 
- Our analysis shows improvement in cache hit rates for both base stations on performing FedAvg in conjunction with DPP-Cache as compared to using
DPP-Cache in isolation.
- Furthermore, it is important to note that the fetching cost associated with FedAvg remains approximately equal to that of DPP-Cache for both base stations. This result shows how using FedAvg improves cache performance without adding extra costs.

### Base Station 1- Cache Hit Rate & Download Rate
![cache_hit1_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/2834802b-2971-4867-b3fc-dbff05c441cc)
![cache_replace_rate1_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/521aef29-e524-4bbb-9a7b-56ce5e1ee8a6)

### Base Station 2- Cache Hit Rate & Download Rate
![cache_hit2_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/8c9eb5a3-ef8b-4f0f-8fb0-baf963136493)
![cache_replace_rate2_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/39e7e1b5-e208-4878-8105-c4c9bbe41c8a)

## Extension
The Exponential Weighted Average (EWA) algorithm assigns different weights to base stations based on the inverse of their MSE. The idea is to give more importance to base stations with lower MSE, indicating better model performance. The weight assigned to each base station is calculated exponentially, contributing to high-performing base stations more significantly.

## Contributors
- [Prof. Bharath B.N.](https://bharathbettagere.github.io/mywebpage/)
- [Himanshu Kumar](https://github.com/himansh9u/)
- [Suyash Gaurav](https://github.com/SuyashGaurav/)
