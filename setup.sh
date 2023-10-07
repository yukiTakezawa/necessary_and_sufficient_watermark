conda create -n llm python=3.8
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
conda install jupyter -y
conda install -c conda-forge cchardet -y
conda install -c anaconda chardet -y
conda install -c conda-forge sentencepiece -y
pip install datasets
pip install accelerate
conda install sacrebleu
