cd model1 && docker build -t ocr-model1:demo . && cd ..
cd model2 && docker build -t ocr-model2:demo . && cd ..
cd router && docker build -t ocr-router:demo . && cd ..

kubectl apply -f k8s/

open http://localhost:30080


if you are redeploying:
kubectl rollout restart deployment/model1
kubectl rollout restart deployment/model2
kubectl rollout restart deployment/router