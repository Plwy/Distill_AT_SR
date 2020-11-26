from options import args
import torch.nn.functional as F

def test():

    for d in args.data_train:
        print(d)
        # module_name = d if d.find('DIV2K') == 0 else 'DIV2KJPEG'
        module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'

        print(module_name)

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    print(args.scale)

# def test2():
#     x = 
#     F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def test3():
    for d in args.data_train:
        module_name = d if d.find('DIV2K') == 0 else 'DIV2KJPEG'
        print(module_name)

if __name__ == '__main__':
    test3()