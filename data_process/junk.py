from torch_lr_finder import LRFinder

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(trainloader, end_lr=10, num_iter=1564, step_mode='exp')
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state



a = zip(lr_finder.history['lr'], lr_finder.history['loss'])
best_lrloss = sorted(a, key=take_lr, reverse=False)[:50]

def take_lr(x):
    # print(x)
    return x[1]


tup = zip(lr_finder.history['loss'], lr_finder.history['lr'])
sorted(tup,key=take_lr,  reverse=False)[:50]


class shrink:
    def __init__(self, config):
        self.config = config
    
    def apply_augmentations(self):
        pass
