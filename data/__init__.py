from importlib import import_module
from dataloader import MSDataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        # 训练集加载
        if not args.test_only:
            # 加载多个训练数据集
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K') == 0 else 'DIV2KJPEG'
                module_train = import_module('data.' + module_name.lower())  
                datasets.append(getattr(module_train, module_name)(args, name=d))

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        # 测试集加载
        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                module_test = import_module('data.' + module_name.lower())   # data.div2k
                testset = getattr(module_test, module_name)(args, train=False, name=d)

            self.loader_test.append(MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu
            ))

