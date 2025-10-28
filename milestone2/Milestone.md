# Milestone 2


### Learning Goals: 

The primary task in this milestone is to test specific changes in pipeline configurations and determine and analyze how those changes will affect accuracy and performance measures. We will test model versions and measure resource consumption and compare with accuracy. 


### Tasks:

T1. Create multiple variants of your model. Model variants can also be generated using model optimizers such as TensorRT, and ONNX graph optimization by using different quantization level of neural networks. We will study model optimization in Week 8 and 9. Each variant must be containerized so it can be deployed and scaled. Store all variants in a model registry such as MinIO. 

T2. Since different models have different performance, requests between each model will need to be queued. You will need to use something like a gRPC service(https://grpc.io/) to connect two model variant containers. [Here](https://blog.roboflow.com/deploy-machine-learning-models-pytorch-grpc-asyncio/) is a blog that may help


T1. Observer the latency of each step. Break your observed latency for a pipeline configuration into per step latency. Note for this you will now need to containerize each model separately and be able to deploy it so that it can be horizontally scaled. Observe which model inference step is most time consuming. Report it. 

T2. Set an SLA for your pipeline such as 500ms of latency and serving 20 requests per second. For the given step, try out different model versions on different number of replicated nodes (horizontal scaling) at different cost, where cost is # of replicas ×allocated CPU cores per replica. Create a table such as the following for each step.  

Variant Scale Latency Cost Accuracy
YOLOv5n 2 80 2×1 65.7
YOLOv5m 4 374 4×2 44.1
YOLOv5n 2 418 2×1 47.5
YOLOv5m 5 1546 5×2 64.1



Given your step SLA, note which version and core combination results in best achieving your step SLA. 
Try at least four different model versions and four different CPU cores. 

T3. Determine if multiple configurations satisfy the latency constraints of the inference pipeline. State what determines the
"optimal” configuration? Does model selection at an earlier stage of the pipeline affect the optimal model selection at downstream models?


### Deliverables: 

1. Submit notebook code with the specific variants. The grader should be able to generate data in your tables and plots.
   Note the code for horizontal scaling should be within the notebook. 
3. Create a plot comparing model version, accuracy and latency, and another plot comparing model version, accuracy and throughput. 
4. Draw a table which reports Variant, Scale, Latency,  Cost, Accuracy across the different steps of your pipeline.
5. Report your analysis: which model version will you choose and at how many resources. Note, do account for overall SLA and accuracy measures when making this analysis.
6. Project presentation. You should be able to present your work till Milestone 2 as part of class presentation. 
   
### Grading:

This milestone is worth 10 points. 3 points for producing the notebook, 3 points for plot and table generation. 4 points for the report and presentation that describes analysis and is clear, complete, and precise.

