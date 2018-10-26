# Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods

Pytorch Implementation of Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods


* ÁlvaroArcos-GarcíaJuan A.Álvarez-GarcíaLuis M.Soria-Morillo*
Deep neural network for traffic sign recognition systems: An analysisof spatial transformers and stochastic optimisation methods  
https://reader.elsevier.com/reader/sd/pii/S0893608018300054?token=0656FA2921430AA401BA73A6990A187F32A6FBDD12EAA2FC87FD556B3CDDF6DA8D5BE54F230A979E57369C48AB081452



LCN Implementation is taken from https://github.com/dibyadas/Visualize-Normalizations

##Notes:

-ASGD Works best among all optimizers for me for Learning Rate : 10^-2 
-Class imbalance is removed prior to training 
-Learning Rate Decay worsenes the performance 
-Data Augmentation in general decreases performance
-Architecture is changed slighlty from the  original set of layers
-Currently Gaussian filter is kept constant for LCN, where as ideally it should be chosed at random during run-time 


![Main Architecture](https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems/blob/master/Main%20Architecture.png)


![Spatial Network](https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems/blob/master/Spatial%20Network.png)


![Validation Error](https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems/blob/master/validation2.png)











- *Max Jaderberg Karen Simonyan Andrew Zisserman Koray Kavukcuoglu* 

Spatial Transformer Networks 
(https://arxiv.org/pdf/1506.02025.pdf
)
