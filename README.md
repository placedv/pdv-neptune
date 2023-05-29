## Placedv Neptune

Placedv Neptune is a proprietary AI model that can be trained and utilized across a wide range of technological domains. Its flexibility and adaptability make it a valuable asset in various industries, from healthcare to finance, from e-commerce to entertainment. Whatever the field, Placedv Neptune is ready to tackle the most complex challenges and provide intelligent solutions.

### Get started

First of all, install all the requirement packages:

```shell
pip install -r requirements.txt
```

Download and add in the main folder `GoogleNews-vectors-negative300.bin`, the pre-trained model of Google.

### Run your first training

```shell
python training/tr_pdv_neptune.py

Type your question here: What kind of government does Italy have?
Answer: The Italian Republic is a parliamentary democracy located in southern Europe, with Rome as its capital city.
Was this answer helpful? (yes/no): yes
Type your question here: Where is born Mattarella?
Answer: The Italian Republic is a parliamentary democracy located in southern Europe, with Rome as its capital city.
Was this answer helpful? (yes/no): no
```

### Folder structure

```bash
|-- pdv-neptune
|   |-- training
|   |   |-- tr_pdv_neptune.py               # training module
|   |   |-- feedback.json                   # feedback score
|   |   |-- sentence_vectors.json           # module + feedback vectorize
|   |-- example_module.csv                  # module.cv example
|   |-- pdv_neptune.py                      # python module
|   |-- GoogleNews-vectors-negative300.bin  # module pre-trained
```

### Start using Placedv Neptune

```shell
python pdv_neptune.py
```

if you haven't already do the training yet:

```shell
No trainined model to load. Before start it, do the training.
```

else:
```shell
Type your question here (type 'exit' to exit): How many clicks did BroadBandBank Factoring 2022?
Answer:
0     BroadBandBank Factoring 2022 did 976 clicks  # right answare
85    BroadBandBank_credito did 318369 clicks
40    BroadBandBank_credito did 318369 clicks
20    BroadBandBank_conto did 71955 clicks
10    BroadBandBank_Green did 15411 clicks
```


