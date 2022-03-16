# AutoReg

This is the official code for "Automated Learning for Deformable Medical Image Registration by Jointly Optimizing Network Architectures and Objective Functions" (https://arxiv.org/pdf/2203.06810)


A successful registration algorithm, either derived from conventional energy optimization or deep networks requires tremendous efforts from computer experts
to well design registration energy or to carefully tune network architectures for the specific type of medical data. To tackle the
aforementioned problems, this paper proposes an automated learning registration algorithm (AutoReg) that cooperatively optimizes both architectures and their corresponding training objectives, enable non-computer experts, e.g., medical/clinical users, to conveniently find off-the-shelf registration algorithms for diverse scenarios. 

<div align=center>
<img src=png/pipline-1.png width=100% />
</div>



# Requirements

> Python == 3.6.8, PyTorch == 0.3.1, torchvision == 0.2.0  <br>

> Note: PyTorch 0.4 is not supported at this moment and would lead to OOM.  <br>




