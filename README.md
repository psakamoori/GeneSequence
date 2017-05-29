# GeneSequence
Neon code for Splice Junction Gene Sequence classification (EI / IE / Neither)

# Training Splcie Junction gene data using Neon Framework
Ref: "https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29"

# Setup
Have neon repo sync'd to local host and follow the steps to build the repo: https://github.com/NervanaSystems/neon

## Running "GeneSequence" - Trainig and validation

Place files given in this repo accordingly

gene.py -> ~/neon/neon/data/.
splice_train_val.py -> ~/neon/examples/gene/.   (make dir gene)
spliace_data_CAGT.csv -> ~/neon/examples/gene/.

# Command to Traing and Validate

~/neon/examples/gene/$ python splice_train_val.py

# Expected output

(.venv2) username@linuxhost:~/MyProjects/neon/examples/gene$ python splice_train_val.py
Epoch 0   [Train |████████████████████| 2552/2552 batches, 0.14 cost, 12.77s]
2017-05-28 22:02:33,525 - neon - DISPLAY - Misclassification error = 0.6%

Known Issues:
  Temporary fix to avoid evaluation failure due to 'nan'





