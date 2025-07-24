conda create --name DeepMusics python=3.6
conda activate DeepMusics
conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch


conda install -c conda-forge statsmodels
pip install numpy pandas scikit-learn scipy dgl ecos joblib numexpr osqp
conda install -c sebp scikit-survival
