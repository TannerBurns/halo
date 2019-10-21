# Halo

![Python3.7 badge](https://img.shields.io/badge/python-v3.7-blue)

    Utility library for parsing, analysis, clustering, and classifying.


# Example

```python
from halo import Covenant
halo = Covenant()
train, test, train_labels, test_labels = halo.split_training_set(features, labels)
halo.fit_all()
halo.test_all()
```
