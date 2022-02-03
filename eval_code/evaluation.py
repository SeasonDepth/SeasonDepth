# Authors: Hanjiang Hu, Baoquan Yang and Zhijian Qiao
# Shanghai Jiao Tong University

# This is the evaluation code for SeasonDepth benchmark.

import argparse
import os
import cv2
import numpy as np
import tqdm
import math
import xlrd
import xlwt
import json

# metrics
xl_head = ('abs_rel', 'a1')
digit_list = ['13033', '12833', '12845', '12859', '12875', '12881', '12887', '12895', '12904', '12929', '12992',
              '13118']
env_list = ['env00', 'env01', 'env02', 'env03', 'env04', 'env05', 'env06', 'env07', 'env08', 'env09', 'env10', 'env11']

parser = argparse.ArgumentParser()
parser.add_argument('--results_pth', type=str, default='./results',
                    help='path to save the results.xls file.')
parser.add_argument('--pred_pth', type=str, default=None, required=True,
                    help='path of folder to your own results.')
parser.add_argument('--gt_pth', type=str, default=None, required=True,
                    help='path of folder to the ground truth.')
parser.add_argument('--gui', action="store_true", help='show the results of evaluation.')
parser.add_argument('--not_clean', action="store_true", help='not to delete all the intermediate files.')
parser.add_argument('--disp2depth', action="store_true",
                    help='convert disparity results to depth map for evaluation')


def compute_errors(ground_truth, predication):
    """
    Compute abs_rel and a1

    :param ground_truth:
    :param predication:
    :return:
    """
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    threshold = np.maximum((ground_truth / predication), (predication / ground_truth))
    a1 = (threshold < 1.25).mean()

    return abs_rel, a1


def align_img(pred_depth, ground_depth, c):
    """
    align predicted depth maps with ground truth.

    :param pred_depth:
    :param ground_depth:
    :param c:
    :return: aligned predicted depth

    """
    pred_depth_f = pred_depth.astype(float)
    # ground_depth_f = ground_depth.astype(float)
    pred_avg = np.average(pred_depth_f[c])
    pred_var = np.var(pred_depth_f[c])
    gt_avg = np.average(ground_depth[c])
    gt_var = np.var(ground_depth[c])
    # print('pred: ', pred_avg, pred_var, 'gt: ', gt_avg, gt_var)

    if args.gui:
        print('pred: ', pred_avg, pred_var, 'gt: ', gt_avg, gt_var)

    pred_depth_f = (pred_depth_f - pred_avg) * math.sqrt(gt_var / pred_var) + gt_avg

    pred_depth_f[pred_depth_f > 65535] = 65535
    pred_depth_f[pred_depth_f < 0] = 0

    if args.gui:
        print('aligned: ', np.average(pred_depth_f[c]), np.var(pred_depth_f[c]))

    pred_aligned = pred_depth_f.astype(np.uint16)

    return pred_aligned


def process(pred_pth, gt_pth):
    """
    Read predicted image and ground truth image, and align them to the same scale.
    Finally, compute the metric of a pair of aligned images

    :param pred_pth:
    :param gt_pth:
    :return:
    """
    if not (os.path.exists(pred_pth) and os.path.exists(gt_pth)):
        return None
    pred_depth = cv2.imread(pred_pth, -1)
    gt_depth = cv2.imread(gt_pth, -1)
    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(pred_depth, gt_depth.shape[::-1])
    if args.disp2depth:
        pred_depth[pred_depth > 0] = 1 / pred_depth[pred_depth > 0]

    a = pred_depth == 0
    b = gt_depth == 0
    c = gt_depth > 0
    if len(c) == 0:
        return None

    pred_depth[a] = 1
    pred_depth[b] = 1
    gt_depth[b] = 1

    pred_aligned = align_img(pred_depth, gt_depth, c)
    pred_aligned[b] = 1
    pred_aligned[pred_aligned == 0] = 1
    pred_aligned[gt_depth == 0] = 1

    gt_depth = gt_depth.astype(float)
    abs_rel, a1 = compute_errors(gt_depth[c], pred_aligned[c])

    # If customer choose to visualize the difference between predicted imgs and ground truth.
    if args.gui:
        pred_depth = cv2.resize(pred_depth, (512, 256))
        cv2.imshow('img', pred_depth)
        pred_aligned = cv2.resize(pred_aligned, (512, 256))
        cv2.imshow('adjust', pred_aligned / 65536)
        gt_depth = cv2.resize(gt_depth, (512, 256))
        cv2.imshow('ground', gt_depth / 65536)

        print("abs_rel: {} \na1: {} ".format(abs_rel, a1))
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            exit()

    return abs_rel, a1


# For a given environment, calculate the error in the current environment
def process_dataset(_pred_path, _gt_path):
    res_dict = {}
    # Go through each image
    for img in tqdm.tqdm(os.listdir(_pred_path)):
        img_base = img.split('.')[0]
        if img.endswith('jpg') or img.endswith('png'):
            pred_depth = os.path.join(_pred_path, img)
            gt_depth = os.path.join(_gt_path, img_base + '.png')
            res = process(pred_depth, gt_depth)
            if res is not None:
                res_dict[img] = res
    return res_dict


def xl_write_line(worksheet, row, col, t):
    for i in range(len(t)):
        if type(t[i]) != str:
            worksheet.write(row, col + i, float(t[i]))
        else:
            worksheet.write(row, col + i, t[i])


# Given two image sets, compare their difference and compute abs_rel and a1
def evaluation(pred_path, gt_path, xls_path):
    workbook = xlwt.Workbook(encoding='utf-8')
    xls_counter = {}
    for env in env_list:
        worksheet = workbook.add_sheet(env)
        worksheet.write(0, 0, "img_name")
        xl_write_line(worksheet, 0, 1, xl_head)
        worksheet.write(0, 3, "img_name")
        xl_write_line(worksheet, 0, 4, xl_head)
        xls_counter[env] = [1, 1]
    if os.path.exists("{}/result.json".format(xls_path)):
        process_result = json.load(open("{}/result.json".format(xls_path), 'r'))
    else:
        process_result = process_dataset(pred_path, gt_path)
        with open("{}/result.json".format(xls_path), 'w') as f:
            json.dump(process_result, f)
    for img in process_result:
        res = process_result[img]
        index = digit_list.index(img[13:18])
        env = env_list[index]
        c = int(img[11])
        worksheet = workbook.get_sheet(env)
        worksheet.write(xls_counter[env][c], 3 * c, img)
        xl_write_line(worksheet, xls_counter[env][c], 3 * c + 1, res)
        xls_counter[env][c] += 1
    workbook.save(xls_path + "/result.xls")

    if args.gui:
        cv2.destroyAllWindows()
    return xls_counter


def rng(arr):
    return (np.max(arr) - np.min(arr)) / np.average(arr)


def rng_a1(arr):
    return (np.max(arr) - np.min(arr)) / (1 - np.average(arr))


def reg_path(path):
    reg = os.path.abspath(path)
    return reg


def write_ind(ws, ind_dict: dict, counter):
    ws.write(counter * 3 + 1, 1, np.average(ind_dict["abs_rel"]))
    ws.write(counter * 3 + 2, 1, np.average(ind_dict["a1"]))
    ws.write(counter * 3 + 1, 2, np.var(ind_dict["abs_rel"]))
    ws.write(counter * 3 + 2, 2, np.var(ind_dict["a1"]))
    ws.write(counter * 3 + 1, 3, rng(ind_dict["abs_rel"]))
    ws.write(counter * 3 + 2, 3, rng_a1(ind_dict["a1"]))


if __name__ == "__main__":
    args = parser.parse_args()
    pred_pth = reg_path(args.pred_pth)
    results_pth = reg_path(args.results_pth)
    gt_pth = reg_path(args.gt_pth)
    assert (os.path.exists(pred_pth) and os.path.exists(gt_pth))

    # Iterate through all slices
    print('**************************************************')
    print("Start evaluation ...")
    if not os.path.exists(results_pth):
        os.makedirs(results_pth)

    slices = os.listdir(pred_pth)
    eval_path = os.path.join(results_pth, "evaluation.xls")
    workbook = xlwt.Workbook(encoding='utf-8')
    total_dict = {"abs_rel": [], "a1": []}
    for s in sorted(slices):
        print("Start evaluating {}".format(s))
        _pred_pth = os.path.join(pred_pth, s)
        _gt_pth = os.path.join(gt_pth, s)
        xls_path = os.path.join(results_pth, s)
        if not os.path.exists(os.path.join(results_pth, s)):
            os.makedirs(os.path.join(results_pth, s))
        xls_counter = evaluation(_pred_pth, _gt_pth, xls_path)

        result_xls = xlrd.open_workbook(xls_path + "/result.xls")
        worksheet = workbook.add_sheet(s)
        slice_dict = {"abs_rel": [], "a1": []}
        e_counter = 0
        for env in env_list:
            # worksheet = workbook.add_sheet(env)
            xl_write_line(worksheet, e_counter * 3, 1, ['avg', 'var', 'rng'])
            worksheet.write_merge(e_counter * 3 + 1, e_counter * 3 + 2, 0, 0, env)
            worksheet.write(e_counter * 3 + 1, 4, "abs_rel")
            worksheet.write(e_counter * 3 + 2, 4, "a1")
            result_sheet = result_xls.sheet_by_name(env)
            env_dict = {"abs_rel": [], "a1": []}
            for row in range(1, xls_counter[env][0]):
                env_dict["abs_rel"].append(float(result_sheet.cell_value(row, 1)))
                env_dict["a1"].append(float(result_sheet.cell_value(row, 2)))
                total_dict["abs_rel"].append(float(result_sheet.cell_value(row, 1)))
                total_dict["a1"].append(float(result_sheet.cell_value(row, 2)))
                slice_dict["abs_rel"].append(float(result_sheet.cell_value(row, 1)))
                slice_dict["a1"].append(float(result_sheet.cell_value(row, 2)))
            for row in range(1, xls_counter[env][1]):
                env_dict["abs_rel"].append(float(result_sheet.cell_value(row, 4)))
                env_dict["a1"].append(float(result_sheet.cell_value(row, 5)))
                total_dict["abs_rel"].append(float(result_sheet.cell_value(row, 4)))
                total_dict["a1"].append(float(result_sheet.cell_value(row, 5)))
                slice_dict["abs_rel"].append(float(result_sheet.cell_value(row, 4)))
                slice_dict["a1"].append(float(result_sheet.cell_value(row, 5)))
            if len(env_dict["abs_rel"]) > 0:
                write_ind(worksheet, env_dict, e_counter)
            e_counter += 1
        worksheet.write_merge(e_counter * 3 + 1, e_counter * 3 + 2, 0, 0, s + "_total")
        worksheet.write(e_counter * 3 + 1, 4, "abs_rel")
        worksheet.write(e_counter * 3 + 2, 4, "a1")
        write_ind(worksheet, slice_dict, e_counter)

    worksheet = workbook.add_sheet("total")
    xl_write_line(worksheet, 0, 1, ['avg', 'var', 'rng'])
    worksheet.write(1, 0, "abs_rel")
    worksheet.write(2, 0, "a1")
    write_ind(worksheet, total_dict, 0)
    workbook.save(eval_path)

    print('**************************************************')
    print('Well done!')
    print('Results:')
    print('AbsRel Average:', format(np.average(total_dict["abs_rel"]), '.4f'))
    print('a1 Average:', format(np.average(total_dict["a1"]), '.4f'))
    print('AbsRel Variance 10^(-2):', format(np.var(total_dict["abs_rel"]) * 100, '.4f'))
    print('a1 Variance 10^(-2):', format(np.var(total_dict["a1"]) * 100, '.4f'))
    print('AbsRel Relative Range:', format(rng(total_dict["abs_rel"]), '.4f'))
    print('a1 Relative Range:', format(rng_a1(total_dict["a1"]), '.4f'))
    print('See more details in:', os.path.join(eval_path))
