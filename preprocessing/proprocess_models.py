import torch


def main():
    bart_model = torch.load('modules/facebook/bart-base/pytorch_model.bin', map_location='cpu')
    univl_model = torch.load('weight/univl.pretrained.bin', map_location='cpu')
    univl_model = {k: v for k, v in univl_model.items() if not k.startswith('bert.')}
    for k, v in bart_model.items():
        if k.startswith('model.'):
            k = k.replace('model.', 'bert.')
            univl_model[k] = v
    torch.save(univl_model, 'weight/univl.bart.pretrained.bin')
    print('done')

if __name__ == '__main__':
    main()
