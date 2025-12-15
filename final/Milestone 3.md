# Milestone 3

### Learning Goal

In this milestone we will aim to turn our experimentation of achieving an optimal pipeline into an automated optimizer that will choose the best pipeline configuration, based on collected data. This is an open-ended milestone where one may design simple rules for optimization or use an ILP solver for optimization. Note the optimization problem is NP-hard so no point designing a provable approach—the best we can do is design efficient heuristics. Use of agents will be a bonus. 

What is an Optimizer? The Optimizer is the auto-configuration module that will periodically (1) fetch the incoming load from the monitoring system, (2) predicts the next reference load based on the observed historical load, (3) obtains the optimal configuration in terms of the variant used, batch size, and number of replicas for the next timestep, and finally (4) applies the new configuration using Kubernetes. 


For example, if there are two available options per model in a analysis pipeline, the Optimizer will be able to choose more accurate models with small batch sizes to ensure low latency but will  choose lightweight models in higher loads which have more replication and larger batch sizes to ensure high throughput for the system.

### Tasks


T1. Define the accuracy of your pipeline. Note accuracy of a pipeline must be defined as a combination of accuracy of different stages of the pipelines. Define this combination. State any assumptions you have made. State any error correlations that you may identify. If there are model drifts, please define accuracy appropriately. 

T2. State the optimization problem you are solving. Each of the group looks at an optimization problem from the paper assigned to them. 

Understand the problem, map it to your pipeline setting and state it in your own words. 

Video:  Francisco Romero, Qian Li, Neeraja J Yadwadkar, and Christos Kozyrakis. INFaaS: Automated model-less inference serving. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 397–411, 2021.
ID Card: See paper Jeff Zhang, Sameh Elnikety, Shuayb Zarar, Atul Gupta, and Siddharth Garg. Model-switching: Dealing with fluctuating workloads in machine-learning-as-a-service systems. In 12th USENIX Workshop on Hot Topics in Cloud Computing (HotCloud 20), 2020.
Audio: Jashwant Raj Gunasekaran, Cyan Subhra Mishra, Prashanth Thinakaran, Bikash Sharma, Mahmut Taylan Kandemir, and Chita R Das. Cocktail: A multidimensional optimization for model serving in cloud. In USENIX NSDI, pages 1041–1057, 2022.
Q&A:  IPA: Inference Pipeline Adaptation to Achieve High Accuracy and Cost-Efficiency, Saeid Ghafouri, Kamran Razavi, Mehran Salmani, Alireza Sanaee, Tania Lorido-Botran, Lin Wang, Joseph Doyle, Pooyan Jamshidi, https://arxiv.org/abs/2308.12871v3

As part of the optimization problem, describe how different data values are collected in your pipeline for the optimization problem. 

T3. (Bonus) Show how you solved the optimization problem. Note if you use Gurobi solver then it is free for academic use. If you just used rules state those. 

T4. (Bonus) Use an agentic framework to solve the optimization problem. 

### Deliverables:

1. A demonstration of your project. This should showcase the different pipeline variants and for each how to collect different metrics.
   In your demonstration also showcase a functional, documented Dockerfile to run your project from start to end. This Dockerfile should be able to collect metrics for the optimization problem, and show how to achieve results for each milestone (setting up the pipelines, collecting metrics and accuracy, setting up assertions and finally optimization.) 
2. A technical report describing in detail the motivation for the pipeline, system design, and optimization problem as it applies to your pipeline. The technical report should also describe tools learned during the class and used in this project. It should include all experiments and results. 


