cd model1
docker build -t ocr-model1:demo .
cd ..


cd model2
docker build -t ocr-model2:demo .
cd ..



cd router
docker build -t ocr-router:demo .
cd ..


kubectl apply -f k8s/


kubectl port-forward svc/router-svc 8080:8080


open http://localhost:8080