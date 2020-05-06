from verification_net import VerificationNet
import config
import os


def get_verification_model(datamode, generator):
    path = f"./models/verification_model_{datamode.name}/"
    if not os.path.exists(path):
        os.mkdir(path)

    net = VerificationNet(force_learn=False,
                          file_name=path + "model.tf")

    if not config.LOAD_VERIFICATION_NET_MODEL:
        net.train(generator=generator, epochs=5)
    return net
