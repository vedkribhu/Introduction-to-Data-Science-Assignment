import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
x = cv.imread("sample")
print(x.shape)
reconstructed_image = {1:np.ones((768, 1024, 3)), 2:np.ones((768, 1024, 3)), 4:np.ones((768, 1024, 3)), 16:np.ones((768, 1024, 3))}
frobenous = {1:0, 2:0, 4:0, 16:0}
total_frobenous = 0
for j in range(3):
    m = x[:,:, j]
    print(m.shape)
    U, sigma, V = np.linalg.svd(m)
    for i in [1,2,4,16]:
        reconstructed = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        reconstructed_image[i][:, :, j] = reconstructed
        frobenous[i] += np.sum(sigma[:i]**2, axis = 0)
    total_frobenous+=np.sum(sigma[:]**2, axis = 0)
# print(np.array(reconstructed_image[2]))
for i in [1,2,4,16]:
    l = reconstructed_image[i].astype('uint8')
    # print(l.shape)
    print(frobenous[i]/total_frobenous)
    cv.imwrite("sigma_"+str(i)+".png", l)
    # cv.imshow("dh", l)
    # cv.waitKey(0)
# x = np.reshape(l, (768, 1024, 3))
# plt.savefig("main")
# plt.imshow(l)
# plt.show()

