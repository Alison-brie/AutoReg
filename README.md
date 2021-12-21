# AutoReg

The code is still being sorted......

This is the official code for "Jointly Optimizing Architecture and Training Objective for Medical Image Registration" 


Conventional registration methods optimize an objective function independently for each pair of images, time-consuming for large data.
Recent learning-based methods render fast registration by leveraging deep networks to directly learn the spatial transformation fields between the source and target images. 
However, it needs intensive labor and extensive experience to manually design the network architecture and tuning training objectives for multiple types of medical data.
To tackle the aforementioned problems, this paper proposes an automated registration learning framework that searches both architectures and their corresponding training objectives, friendly to users for medical image analysis. 

![Alt text](pipeline.png)

<img src=pipline.png width=50%>

# Requirements

> Python == 3.6.8, PyTorch == 0.3.1, torchvision == 0.2.0  <br>

> Note: PyTorch 0.4 is not supported at this moment and would lead to OOM.  <br>




