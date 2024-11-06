import numpy as np





def convert_to_db(img, I0 = None, EPS = 1e-3):

    if I0 is None:
        return np.clip(-np.log(img + EPS), 0, None)
    else:
        return -np.log(img + EPS) + np.log(I0 + EPS)


def convert_db_to_img(db, I0 = None):
    if I0 is None:
        return np.exp(-db)
    else:
        return I0*np.exp(-db)


def try_inversing():
    from matplotlib import pyplot as plt
    img = np.ones((32,32,3))
    img[:,:,0] = 1/2
    img[:, :, 1] = 1 / 4
    img[:, :, 2] = 1 / 6
    reconstructed = convert_db_to_img(convert_to_db(img))
    I0 = np.array([0.2, 0.7, 0.1])
    reconstructed_I = convert_db_to_img(convert_to_db(img, I0=I0), I0=I0)
    fig, ax = plt.subplots(1,3, figsize = (8,8))
    ax[0].imshow(img)
    ax[0].set_title('ground truth')
    ax[1].imshow(reconstructed)
    ax[1].set_title('img -> db -> img')
    ax[2].imshow(reconstructed_I)
    ax[2].set_title('img -> db -> img with I0')

    plt.show()

if __name__ == '__main__':
    try_inversing()

