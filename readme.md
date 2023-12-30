![Multiple Server](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/654f6670-1240-47ba-b05d-8a144c527ae9)
## Introduction
This repository explores the concept of federated caching and training in a network with multiple base stations. Federated caching involves storing frequently accessed data at base stations to reduce the need to fetch it from a central server whenever requested. The primary goal is to maximize the time-averaged cache hit rate, considering constraints such as fetching cost and a fixed cache size.
In the context of federated learning, multiple devices associated with different base stations collaborate to train a shared model. Local updates are sent to a central server for aggregation into a global model without sharing raw data. This setup aims to preserve user privacy while enabling collaborative model training.

## Pictorial representation
![Multiple Server (1)](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/238375b3-6f41-4b35-a843-e66a64f960e5)

## Results
**Base Station 1- Cache Hit Rate & Download rate**
![cache_hit1_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/2834802b-2971-4867-b3fc-dbff05c441cc)
![cache_replace_rate1_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/521aef29-e524-4bbb-9a7b-56ce5e1ee8a6)

**Base Station 2- Cache Hit Rate & Download rate**
![cache_hit2_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/8c9eb5a3-ef8b-4f0f-8fb0-baf963136493)
![cache_replace_rate2_page-0001](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration/assets/102952185/39e7e1b5-e208-4878-8105-c4c9bbe41c8a)

## Extension
The Exponential Weighted Average (EWA) algorithm assigns different weights to base stations based on the inverse of their MSE. The idea is to give more importance to base stations with lower MSE, indicating better model performance. The weight assigned to each base station is calculated exponentially, making the contribution of high-performing base stations more significant.
