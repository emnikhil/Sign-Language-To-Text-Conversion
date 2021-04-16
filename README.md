### Sign-Language-To-Text-Conversion

#### Steps for Building this Project       

* Step 1 : Is of Folder Creation for the DataSet, as it is not easy to create folders from 0 to 10 and A to Z in both training and testing folders. As, we are going to build over own dataSet in this project.        

Library Requirements :      
1. os --> `pip install os-sys`      
2. string --> `pip install strings`     


* Step 2 : Is of Data Creation and Collection for the TrainingData and TestingData, so using computer vision, we create our own data.      
The Image captured is processed using image processing and is firstly converted into black and white image, and then Image Post Gaussian Blur is used for detecting boundaries of the hand gesture, which is then stored in the respective folders.

Library Requirements :
1. os --> `pip install os-sys`      
2. numpy --> `pip install numpy`        
3. cv2 --> `pip install opencv-pytho      


### License
This repository is under the **MIT License**