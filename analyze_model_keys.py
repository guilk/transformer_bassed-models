import os
import torch

if __name__ == '__main__':
    model_path = '/data/liangkeg/detectron2/mae_pretrain_vit_base.pth'
    model = torch.load(model_path)
    print(model['model'].keys())

    # for n,p in model.named_parameters():
    #     print(n)
    # print(model)