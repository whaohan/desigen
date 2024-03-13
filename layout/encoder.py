import torch
import timm

def get_encoder(name='vit', grad=False, pretrain=True):
    print(f'encoder status: name-{name}, grad-{grad}, pretrain-{pretrain}')
    if name == 'vit': # ([1, 50, 768])
        model = timm.create_model('vit_base_patch32_224', pretrained=pretrain, num_classes=0)
    elif name == 'swin': # ([1, 49, 1024])
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrain, num_classes=0)
    elif name == 'resnet': # ([1, 2048, 7, 7])
        model = timm.create_model('resnet50', pretrained=pretrain, num_classes=0)
    elif name == 'clip':
        model = timm.create_model('vit_base_patch32_224_clip_laion2b', pretrained=pretrain, num_classes=0)
    else:
        raise NotImplementedError
    
    # disable gradient    
    if not grad:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    
    return model

if __name__ == '__main__':
    # NOTE: no pretrained
    model = get_encoder(name='clip')
    input = torch.randn(1, 3, 224, 224)
    out = model.forward_features(input)
    print(out.shape) 

    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print(flops/1e9,params/1e6)