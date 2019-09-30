# Halo

    Library to dynamically train and test different classifiers in bulk


# Example

```python
halo = halo()
train, test, train_labels, test_labels = halo.split_training_set(features, labels)
halo.fit(train, train_labels)
halo.test(test, test_labels)
halo.to_csv(filename='features.csv')
```
