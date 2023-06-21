from ._registry import Registry

register_dataset = Registry(name='dataset')
register_model = Registry(name='model')
register_loss = Registry(name='loss')
register_optimizer = Registry(name='optimizer')
register_scheduler = Registry(name='scheduler')
register_paradigm = Registry(name='paradigm')