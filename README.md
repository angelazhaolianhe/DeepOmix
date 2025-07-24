# DeepOmix: A Multi-Omics Scalable and Interpretable Deep Learning Framework and Application in Cancer Survival Analysis

  ![image](https://github.com/CancerProfiling/DeepOmix/blob/main/Figures/DeepOmix.png)
  
## Introduction of DeepOmix  
Multi-omics data, such as gene expression, methylation, mutation and copy number variation, can elucidates valuable insights of molecular mechanisms for various diseases. However, due to their different modalities and high dimension, utilizing and integrating different types of omics data suffers from great challenges. There is an urgent need to develop a powerful method to improve survival prediction and detect functional gene modules from multi-omics data to overcome these difficulties. In this paper, we developed DeepOmix, a flexible, scalable and interpretable method for extracting relationships between the clinical outcomes and multi-omics data based on deep learning framework. DeepOmix can efficiently implement non-linear combination and incorporate prior biological information defined by users (such as signalling pathway and tissue-specific functional gene modules) 

DeepOmix can be applied to different resolved-omics data and any type of clinical outcomes, including categorical (stages of subtypes) and continuous (survival time) ones. DeepOmix was evaluated on eight cancer datasets publicly available with four types of omics data and clinical data of survival time from TCGA project and compare its performance with other five cutting-edge prediction algorithms. In addition, the relevance of the information extracted when spiting the effect of various signatures and demonstrate the integration of activist of pathways estimated in a multi-omics context for the analysis of inter-cancer survival signalling.

## Install
To use DeepOmix, do the following:

1.	Please first make sure that you have anconda installed. (https://www.anaconda.com/products/individual#Downloads)  
If you don't have conda, you might find it helpful to install Miniconda before beginning:
```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
```
Then, create a conda environment with python v3 and clone the git repo.
```
    conda create --name DeepOmix python=3.6  ##Create the environment and install python 3.6
    git clone https://github.com/CancerProfiling/DeepOmix.git DeepOmix && cd DeepOmix
    conda activate DeepOmix ## activate the environment of conda
```
2.	Next step is to install a set of required python packages, including torch, statmodels, scikit-survival, and all of their respective dependencies.
```
    
    conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch ### install the torch , note that the version of cuda must match your computer
    conda install -c conda-forge statsmodels ## install the required python packages
    pip install numpy pandas scikit-learn scipy dgl ecos joblib numexpr osqp ## install the required python packages
    conda install -c sebp scikit-survival ## install the required python packages
```

## Demo
###### 1.	Preparation of input data
  Three kinds of data are required to train the model, namely patients’ multi-omics data, their clinical survival time data, and functional module data defined by users. Our demo data are in the Data folder. 
  
1.1	Multiple types of omics data (*rna.csv.gz, scna.csv.gz, mutation.csv.gz* and *methy.csv.gz* in the Data folder)
Each omics data at the gene level are input as a format of a matrix, in which each row is a gene and each column is a sample. All the omics data are for the same cohort of patients. For the omics data with continuous value (such as gene expression data, DNA methylation data, CNA data), each variable is scaled into a standard normal distribution.

1.2	Functional module data (*pathway_module.csv* in the Data folder)
Prior biological knowledge of functional module data are input as the format of matrix, in which each row represents a module and each column represents a gene. In our research, signaling pathways were taken as the functional module input. KEGG and Reactome [33, 34] pathway gene sets were obtained from Molecular Signatures Database (MSigDB) (http://www.gsea-msigdb.org/gsea/msigdb/index.jsp).

1.3	Clinical survival time data (*clinical.csv.gz* in the Data folder)
In our research, survival time is the predicted label.
  
###### 2.	Running Experiments
2.1	Transferring the input data into required format
```
python Transfer_data.py 
```
  The three types of data are provided in one folder (*in our demo, we provide LGG data in the Data folder*), transferred into *train.csv*, *test.csv*, *validation.csv* and *pathway_module_input.csv* in the output folder the user defined (*in our demo, is the Multiple and Single folder in the Data folder*). 
  These data are used to train the survival time prediction model.

2.2	Running the tests
Train the model via the following:

```
python RunSingle.py
python RunMultiple.py
```
The users need to change the paramters of *L2_Lambda,Initial_Learning_Rate,num_epochs* and network size in the deep learning model of *In_Nodes,Pathway_Nodes,Hidden_Nodes and Out_Nodes*


## Contact

Please open an issue or contact zhaolianhe@ict.ac.cn with any questions.
