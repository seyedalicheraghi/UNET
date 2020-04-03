import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt
from PIL import Image


M = cv2.imread('1.png')
M = M[:,:,0]
print(M.shape)
cv2.imwrite('aaa.BMP', M)
# cv2.imwrite(M,"aaa.bmp")


# image = cv2.imread("Untitled.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = image//255
# new = cv2.imread("Untitled1.png")
# new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
# new = new//255
#
# P = image
# M = new
# TP = len(np.where(P * M > 0)[0])  # AND between prediction (P) and ground truth (M)
# FP = len(np.where(cv2.subtract(P, M) > 0)[0])  # white pixels in P that don't have corresponding white pixels in M
# FN = len(np.where(cv2.subtract(M, P) > 0)[0])  # white pixels in M that don't have correspondence in P
# print(TP / (TP + FN))
# print(TP / (TP + FP))
# print(TP, FP, FN)
# TP = 0
# FP = 0
# FN = 0
# # AND between prediction and ground truth
# TP += len(np.where(P * M > 0)[0])
# # white pixels in P that don't have corresponding white pixels in ground truth
# FP += len(np.where(P - M > 0)[0])
# # white pixels in M that don't have correspondence in P
# FN += len(np.where(M - P > 0)[0])
# print(TP / (TP + FN))
# print(TP / (TP + FP))
# print(TP, FP, FN)
# # minY = np.amin(np.where(image > 0)[0])
# # minX = np.amin(np.where(image > 0)[1])
# # maxY = np.amax(np.where(image > 0)[0])
# # maxX = np.amax(np.where(image > 0)[1])
# # rows = np.where(image > 0)[0] # height
# # columns = np.where(image > 0)[1] # width
# # av_height = []
# # bottom_height = []
# #
# # for items in range(minX, maxX + 1):
# #     flag = False
# #     for i in range(0, len(rows)):
# #         if columns[i] == items:
# #             if not flag:
# #                 flag = True
# #                 min_Y = rows[i]
# #                 max_Y = rows[i]
# #             if min_Y > rows[i]:
# #                 min_Y = rows[i]
# #             if max_Y < rows[i]:
# #                 max_Y = rows[i]
# #     av_height.append((max_Y - min_Y) + 1)
# #     bottom_height.append(max_Y)
# # print(statistics.mean(av_height), statistics.median(bottom_height), maxY)

a = [[68.9175329626902, 61.43969953661444, 64.5067384053156, 66.52854167204181, 62.12823634329582, 66.84143140883444, 67.69653958223917, 67.64808346988923, 71.74167447659566, 64.06894454561186, 66.3060989802545, 66.61051275228688, 65.28903147228348, 66.1181656688518, 68.0524453373935, 66.62557809009253, 66.47086403135626, 66.53593081528044, 63.257545707580945, 70.35248550955164, 65.53810336952635, 61.33054502046354, 66.13627983627741, 66.2630898221878, 71.86528057464307, 65.30766273786459, 62.662635786476116, 64.97676063582043, 61.88316669440896, 66.22806966273328, 68.03855313699889, 66.00779774259773, 71.98375975256685, 66.46515999798648, 68.00105443694869, 65.68829955273267, 67.70953069733835, 66.51614470571452, 68.10971115015211, 67.34145738863207, 65.5620830332176, 66.36694583518751, 66.02869920558777, 65.79158922319422, 69.68485284262586, 65.1852330959034, 69.80909051350885, 60.948255532608606, 62.86942747566793, 63.31924600105316, 67.90743396571278, 64.94235130526022, 66.960758419958, 64.10966435455485, 59.88593460310368, 68.35067394275825], [76.0323270565199, 69.77178456802113, 81.24449610491199, 75.08659614146593, 64.88691725019105, 81.01119623632765, 77.57401311167999, 74.58230259383524, 81.85081096249144, 68.25490986284085, 74.70965080877859]]


# for items in a:
#       axes = plt.gca()
#       axes.plot(items)
#       axes.plot(items, label="PR 1", color='red')
#       plt.legend()
# plt.savefig('sss.png')
