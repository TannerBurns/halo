# Halo

![Python3.7 badge](https://img.shields.io/badge/python-v3.7-blue)

    Library to dynamically train and test different classifiers in bulk


# Example

```python
from halo.multimodelingtools import Covenant
halo = Covenant()
train, test, train_labels, test_labels = halo.split_training_set(features, labels)
halo.fit_all()
halo.test_all()
```
