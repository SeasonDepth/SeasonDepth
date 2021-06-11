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

# metrics
xl_head = ('abs_rel', 'a1')

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default='./eval_results', help='path to save the results.xls file.')
parser.add_argument('--pred_path', type=str, default=None, required=True, help='path of folder to your own results.')
parser.add_argument('--ground_path', type=str, default=None, required=True, help='path of folder to the ground truth.')
parser.add_argument('--gui', action="store_true", default='', help='show the results of evaluation.')
parser.add_argument('--not_clean', action="store_true", default='', help='not to delete all the intermediate files.')
parser.add_argument('--disp2depth', action="store_true", default='convert disparity results to depth map for evaluation')

# Compute abs_rel and a1
def compute_errors(ground_truth, predication):
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    threshold = np.maximum((ground_truth / predication), (predication / ground_truth))
    a1 = (threshold < 1.25).mean()

    return (abs_rel, a1)

# Align predicted_depth with ground_depth
def align_img(predicted_depth, ground_depth, c):
    b_avg = np.average(ground_depth[c])
    b_var = np.var(ground_depth[c])

    im = predicted_depth.astype(float)
    a_avg = np.average(im[c])
    a_var = np.var(im[c])
    if args.gui:
        print('P', a_avg, a_var, ' G', b_avg, b_var)
    im = (im - a_avg) * math.sqrt(b_var / a_var) + b_avg

    im[im > 65535] = 65535
    im[im < 0] = 0
    if args.gui:
        print('A', np.average(im[c]), np.var(im[c]))
    img_c = im.astype(np.uint16)

    return img_c

# Read predicted image and ground truth image, and align them to the same scale.
# Finally, compute the metric of a pair of aligned images
def process(src, dst):
    predicted_depth = cv2.imread(src, -1)
    ground_depth = cv2.imread(dst, -1)
    if type(ground_depth) == type(None):
        try:
            ground_depth = cv2.imdecode(np.fromfile(dst, dtype=np.uint8), -1)
        except FileNotFoundError:
            ground_depth = None
    if type(predicted_depth) == type(None):
        try:
            predicted_depth = cv2.imdecode(np.fromfile(src, dtype=np.uint8), -1)
        except FileNotFoundError:
            predicted_depth = None
    if type(predicted_depth) == type(None) or type(ground_depth) == type(None):
        return None
    if predicted_depth.shape != ground_depth.shape:
        ground_depth = cv2.resize(ground_depth, predicted_depth.shape[::-1])
    if args.disp2depth:
        predicted_depth = 1 / predicted_depth

    a = predicted_depth == 0
    b = ground_depth == 0
    c = ground_depth > 0
    if len(c) == 0:
        return None
    predicted_depth[a] = 1
    predicted_depth[b] = 1
    ground_depth[b] = 1
    u = predicted_depth[c]
    v = ground_depth[c]

    pt = align_img(predicted_depth, ground_depth, c)
    pt[b] = 1
    pt[pt == 0] = 1
    pt[ground_depth == 0] = 1

    gt = ground_depth.astype(float)
    abs_rel, a1 = compute_errors(gt[c], pt[c])

    # If customer choose to visualize the difference between predicted imgs and ground truth.
    if args.gui:
        predicted_depth = cv2.resize(predicted_depth, (512, 256))
        cv2.imshow('img', predicted_depth)
        pt = cv2.resize(pt, (512, 256))
        cv2.imshow('adjust', pt / 65536)
        gt = cv2.resize(gt, (512, 256))
        cv2.imshow('ground', gt / 65536)

        print(abs_rel, a1)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            exit()

    return (abs_rel, a1)

# For a given environment, calculate the error in the current environment
def process_env(env):
    pred = os.path.join(pred_path, env)
    ground = os.path.join(ground_path, env)
    ground = os.path.join(ground, pred_path[-2:])

    img_list = os.listdir(pred)
    res_table = []
    # Go through each image
    for i in (img_list):
        i_base = i.split('.')[0]
        if i.endswith('jpg') or i.endswith('png'):
            pred_file = os.path.join(pred, i)
            truth_file = os.path.join(ground, i_base + '.png')
            r = process(pred_file, truth_file)
            if r != None:
                res_table.append((i, r))
    return res_table


def xl_write_line(worksheet, row, col, t):
    for i in range(len(t)):
        if type(t[i]) != str:
            worksheet.write(row, col + i, float(t[i]))
        else:
            worksheet.write(row, col + i, t[i])

# Given two image sets, compare their difference and compute abs_rel and a1
def eval(pred_path, ground_path, xls_path, slice, c = "0"):
    workbook = xlwt.Workbook(encoding='utf-8')
    if os.path.exists(ground_path):
        print('start to process {} environments in slice {} c{}'.format(len(os.listdir(ground_path)), slice, c))
        # Iterate through each environment
        for env in tqdm.tqdm(os.listdir(ground_path)):
            ground_env_path = os.path.join(ground_path, env)
            if os.path.isdir(ground_env_path):
                pred_env_path = os.path.join(pred_path, env)
                if not os.path.exists(pred_env_path):
                    os.mkdir(pred_env_path)
                worksheet = workbook.add_sheet(env)
                env_result = process_env(env)
                xl_write_line(worksheet, 0, 1, xl_head)
                for i in range(len(env_result)):
                    worksheet.write(i + 1, 0, env_result[i][0])
                    xl_write_line(worksheet, i + 1, 1, env_result[i][1])
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    workbook.save(xls_path)

    if args.gui:
        cv2.destroyAllWindows()


def cp_line(src, r1, dst, r2):
    for i in range(src.ncols):
        dst.write(r2, i, src.cell_value(r1, i))


def append_table(data, workbook, sheet_info):
    for sheet_name in data.sheet_names():
        table = data.sheet_by_name(sheet_name)

        if sheet_name in sheet_info:
            sheet = sheet_info[sheet_name][0]
            row = sheet_info[sheet_name][1]
        else:
            sheet = workbook.add_sheet(sheet_name)
            sheet_info[sheet_name] = [sheet, 1]
            cp_line(table, 0, sheet, 0)
            row = 1

        nrows = table.nrows
        for i in range(1, nrows):
            cp_line(table, i, sheet, i + row - 1)
        sheet_info[sheet_name][1] += nrows - 1


def append_xls(s1, s2, d):
    data1 = xlrd.open_workbook(s1)
    data2 = xlrd.open_workbook(s2)
    workbook = xlwt.Workbook(encoding='utf-8')
    row_info = {}
    append_table(data1, workbook, row_info)
    append_table(data2, workbook, row_info)
    workbook.save(d)

def reg_path(path):
    reg = os.path.abspath(path)
    return reg


if __name__ == "__main__":
    args = parser.parse_args()
    args.pred_path = reg_path(args.pred_path)
    args.results_path = reg_path(args.results_path)
    args.ground_path = reg_path(args.ground_path)
    args.ground_path = os.path.join(args.ground_path, "depth")
    # Iterate through all slices
    print('**************************************************')
    print("Start evaluation ...")
    for slice in [2, 3, 7, 8]:
        # c0 and c1
        for j in range(2):
            pred_path = os.path.join(args.pred_path, "results{}_c{}".format(slice, j))
            ground_path = os.path.join(args.ground_path, "slice{}".format(slice))
            xls_name = 'evals{}_c{}.xls'.format(slice, j)
            xls_path = args.results_path
            xls_path = os.path.join(xls_path, xls_name)
            if args.results_path and not os.path.exists(args.results_path):
                os.makedirs(args.results_path)
            eval(pred_path, ground_path, xls_path, slice, c = j)
        append_xls(os.path.join(args.results_path, 'evals{}_c0.xls'.format(slice)),
                   os.path.join(args.results_path, 'evals{}_c1.xls'.format(slice)),
                   os.path.join(args.results_path, 'evals{}.xls'.format(slice)))
    append_xls(os.path.join(args.results_path, 'evals2.xls'), os.path.join(args.results_path, 'evals3.xls'),
               os.path.join(args.results_path, 'evals23.xls'))
    append_xls(os.path.join(args.results_path, 'evals7.xls'), os.path.join(args.results_path, 'evals8.xls'),
               os.path.join(args.results_path, 'evals78.xls'))
    append_xls(os.path.join(args.results_path, 'evals23.xls'), os.path.join(args.results_path, 'evals78.xls'),
               os.path.join(args.results_path, 'eval_total.xls'))
    os.remove(os.path.join(args.results_path, 'evals23.xls'))
    os.remove(os.path.join(args.results_path, 'evals78.xls'))

    xl_path = os.path.join(args.results_path)
    xl_scores = []
    xl_score = []
    workbook = xlwt.Workbook(encoding='utf-8')
    total_absrel_avg = 0
    total_a1_avg = 0
    total_absrel_var = 0
    total_a1_var = 0
    total_absrel_rng = 0
    total_a1_rng = 0
    for filename in ['eval_total.xls', 'evals2.xls', 'evals3.xls', 'evals7.xls', 'evals8.xls']:
        sheet = workbook.add_sheet(os.path.basename(filename))
        xl_write_line(sheet, 0, 1, xl_head)

        tab_id = 1
        data = xlrd.open_workbook(os.path.join(xl_path, filename))
        nsheets = len(data.sheet_names())

        all_avg = np.zeros([2, nsheets])
        all_var = np.zeros([2, nsheets])
        all_std = np.zeros([2, nsheets])
        for sheet_name in data.sheet_names():
            xl_avg = []
            xl_var = []
            xl_std = []
            table = data.sheet_by_name(sheet_name)
            nrows = table.nrows
            ncols = table.ncols
            arr = np.zeros([ncols - 1, nrows - 1])
            for col in range(1, ncols):
                for row in range(1, nrows):
                    arr[col - 1][row - 1] = float(table.cell_value(row, col))
                xl_avg.append(np.mean(arr[col - 1]))
                xl_var.append(np.var(arr[col - 1]))
                xl_std.append(np.std(arr[col - 1]))

            sheet.write(tab_id * 3 + 2, 0, sheet_name + '#avg')
            sheet.write(tab_id * 3 + 3, 0, sheet_name + '#var')
            sheet.write(tab_id * 3 + 4, 0, sheet_name + '#std')

            xl_write_line(sheet, tab_id * 3 + 2, 1, xl_avg)
            xl_write_line(sheet, tab_id * 3 + 3, 1, xl_var)
            xl_write_line(sheet, tab_id * 3 + 4, 1, xl_std)

            for col in range(2):
                all_avg[col][tab_id - 1] = xl_avg[col]
                all_var[col][tab_id - 1] = xl_var[col]
                all_std[col][tab_id - 1] = xl_std[col]

            tab_id = tab_id + 1
        if filename == 'eval_total.xls':
            total_absrel_avg = np.mean(all_avg[0])
            total_a1_avg = np.mean(all_avg[1])
            total_absrel_var = np.var(all_avg[0])
            total_a1_var = np.var(all_avg[1])
            total_absrel_rng = (np.max(all_avg[0]) - np.min(all_avg[0])) / np.mean(all_avg[0])
            total_a1_rng = (np.max(all_avg[1]) - np.min(all_avg[1])) / (1 - np.mean(all_avg[1]))
        sheet.write(1, 0, 'avg')
        sheet.write(2, 0, 'var')
        sheet.write(3, 0, 'std')
        sheet.write(4, 0, 'rng')
        for col in range(2):
            sheet.write(1, 1 + col, np.mean(all_avg[col]))
            sheet.write(2, 1 + col, np.var(all_avg[col]))
            sheet.write(3, 1 + col, np.std(all_avg[col]))
            if col == 0:
                sheet.write(4, 1 + col, (np.max(all_avg[col]) - np.min(all_avg[col])) / np.mean(all_avg[col]))
            else:
                sheet.write(4, 1 + col, (np.max(all_avg[col]) - np.min(all_avg[col])) / (1 - np.mean(all_avg[col])))

    workbook.save(os.path.join(args.results_path, 'results.xls'))
    data = xlrd.open_workbook(os.path.join(args.results_path, 'results.xls'))
    for it in ['avg', 'std', 'var']:
        workbook = xlwt.Workbook(encoding='utf-8')

        nsheets = len(data.sheet_names())
        for sheet_name in data.sheet_names():

            sheet = workbook.add_sheet(sheet_name)
            table = data.sheet_by_name(sheet_name)

            lines = 0
            for row in range(table.nrows):
                env_title = table.cell_value(row, 0)
                if '#' in env_title:
                    p = env_title.find('#')
                    env_num = env_title[:p]
                    env_iter = env_title[p + 1:]
                    if env_iter in it:
                        cp_line(table, row, sheet, lines)
                        lines = lines + 1
                else:
                    cp_line(table, row, sheet, lines)
                    lines = lines + 1

        workbook.save(os.path.join(args.results_path, "res_{}.xls".format(it)))
    if not args.not_clean:
        for xls in os.listdir(args.results_path):
            if xls != 'results.xls':
                os.remove(os.path.join(args.results_path, xls))

    print('**************************************************')
    print('Well done!')
    print('Results:')
    print('AbsRel Average:', format(total_absrel_avg, '.4f'))
    print('a1 Average:', format(total_a1_avg, '.4f'))
    print('AbsRel Variance 10^(-2):', format(total_absrel_var*100, '.4f'))
    print('a1 Variance 10^(-2):', format(total_a1_var*100, '.4f'))
    print('AbsRel Relative Range:', format(total_absrel_rng, '.4f'))
    print('a1 Relative Range:', format(total_a1_rng, '.4f'))
    print('See more details in:', os.path.join(args.results_path, 'results.xls'))
