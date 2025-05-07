from LIDCDataLoader import LIDCDataLoader
from LIDC_UNet import UnNet, train_unet, eval
from LDIC_retina_U_net import retinaUnNet, train_retunet, evaluate_retina_unet
# import models.LIDC_UNet as LIDC_UNet
# print(dir(LIDC_UNet))


if __name__ == "__main__":
    ds = LIDCDataLoader(r'LIDC-IDRI-slices',max_samples=1200)
    print(len(ds))
    img, mask = ds[0]
    # print("img shape:", img.shape)
    # print("msk shape:", mask.shape)
    unet = UnNet()
    train_unet(unet, ds)

    eval(unet, ds)

    # unet = UnNet()
    model = retinaUnNet(unet)
    train_retunet(model, ds, epochs=5, batch_size=4)

    evaluate_retina_unet(model, ds)

