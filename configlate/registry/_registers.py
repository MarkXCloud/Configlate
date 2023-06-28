from ._registry import Registry

dataset = Registry(name='dataset')
model = Registry(name='model')
loss = Registry(name='loss')
optimizer = Registry(name='optimizer')
scheduler = Registry(name='scheduler')
metric = Registry(name='metric')