import sys
import os
import getopt

from functions import *
import numpy as np

### HELPER FUNCS ###
def print_confidence_intervals(intervals: list):
    for i in range(len(intervals)):
        print("\t\t({:.4f}, {:.4f})".format(intervals[i][0], intervals[i][1]))

def print_test_outcomes(C: list, d: list, test_outcomes: list):
    for i in range(len(test_outcomes)):
        ci = C[i]
        di = d[i]
        print("\t\tH0: ", end="")
        for indx, j in enumerate(ci):
            print("({} * mu_{}) {}".format(ci[indx], indx + 1, "+" if indx < (len(ci) - 1) else ""), end=" ")
        print(" = ", di, end = "")
        print(", {}.".format("rejected" if test_outcomes[i] else "not rejected"))
### HELPER FUNCS ###

def test_ANOVA1_partition_TSS():
    x = [[6.9, 5.4, 5.8, 4.6, 4.0], [8.3, 6.8, 7.8, 9.2, 6.5], [8.0, 10.5, 8.1, 6.9, 9.3], [5.8, 3.8, 6.1, 5.6, 6.2]]
    sstotal, sswithin, ssbetween = ANOVA1_partition_TSS(x)

    print("Test: ANOVA1_partition_TSS\n")
    print("X: {}\n".format(x))
    print("SStotal: {}, SSwithin: {}, SSbetween: {}".format(sstotal, sswithin, ssbetween))

def test_ANOVA1_test_equality():
    alpha = 0.05
    x = [[6.9, 5.4, 5.8, 4.6, 4.0], [8.3, 6.8, 7.8, 9.2, 6.5], [8.0, 10.5, 8.1, 6.9, 9.3], [5.8, 3.8, 6.1, 5.6, 6.2]]

    print("Test: ANOVA1_test_equality\n")
    print("X: {}\nalpha: {}\n".format(x, alpha))
    ANOVA1_test_equality(x, alpha, print_info=True)


def test_ANOVA1_is_contrast():
    print("Test: ANOVA1_is_contrast\n")

    # contrast
    C = [[-1, 0, 1, 0], [0, 1, 0, -1]]
    print("C: {}".format(C))
    print("Is contrast : {}\n".format(ANOVA1_is_contrast(C)))

    # non contrast
    C = [[-1, 0, 1, 0], [0, 0, 0, -1]]
    print("C: {}".format(C))
    print("Is contrast : {}\n".format(ANOVA1_is_contrast(C)))


def test_ANOVA1_is_orthogonal():
    print("Test: ANOVA1_is_orthogonal\n")

    # orthogonal
    C = [[-1, 0, 1, 0], [0, 1, 0, -1]]
    n = [4, 5, 4, 6]
    print("c1: {}, c2: {}".format(C[0], C[1]))
    print("Is orthogonal contrast : {}\n".format(ANOVA1_is_orthogonal(n, C[0], C[1], print_info=True)))

    # non orthogonal
    C = [[-1, 0, 1, 0], [1, 0, 0, -1]]
    n = [4, 5, 4, 6]
    print("c1: {}, c2: {}".format(C[0], C[1]))
    print("Is orthogonal contrast : {}\n".format(ANOVA1_is_orthogonal(n, C[0], C[1], print_info=True)))

    # non contrast
    C = [[-1, 0, 1, 0], [1, 0, 0, 0]]
    n = [4, 5, 4, 6]
    print("c1: {}, c2: {}".format(C[0], C[1]))
    print("Is orthogonal contrast : {}\n".format(ANOVA1_is_orthogonal(n, C[0], C[1], print_info=True)))

def test_Bonferroni_correction():
    print("Test: Bonferroni_correction\n")
    
    fwer_alpha = 0.1
    m = 5
    print("FWER alpha: {}, m: {}".format(fwer_alpha, m))
    print("Bonferroni corrected alpha: {:.4f}\n".format(Bonferroni_correction(fwer_alpha, m = m)))

    fwer_alpha = 0.05
    m = 3
    print("FWER alpha: {}, m: {}".format(fwer_alpha, m))
    print("Bonferroni corrected alpha: {:.4f}".format(Bonferroni_correction(fwer_alpha, m = m)))


def test_Sidak_correction():
    print("Test: Sidak_correction\n")
    
    fwer_alpha = 0.1
    m = 5
    print("FWER alpha: {}, m: {}".format(fwer_alpha, m))
    print("Sidak corrected alpha: {:.4f}\n".format(Sidak_correction(fwer_alpha, m = m)))

    fwer_alpha = 0.05
    m = 3
    print("FWER alpha: {}, m: {}".format(fwer_alpha, m))
    print("Sidak corrected alpha: {:.4f}".format(Sidak_correction(fwer_alpha, m = m)))


def test_ANOVA1_CI_linear_combs():
    print("Test: ANOVA1_CI_linear_combs\n")

    # DATA from https://itl.nist.gov/div898/handbook/prc/section4/prc436.htm#contrastex2
    alpha = 0.05
    x = [[6.9, 5.4, 5.8, 4.6, 4.0], [8.3, 6.8, 7.8, 9.2, 6.5], [8.0, 10.5, 8.1, 6.9, 9.3], [5.8, 3.8, 6.1, 5.6, 6.2]]

    print("X: {}\nalpha: {}\n".format(x, alpha))

    print("**********")
    print("Orthogonal Contrast")
    C = [[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]]
    print("Linear combinations C: {}\n".format(C))
    bonferroni_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Bonferroni")
    print_confidence_intervals(bonferroni_intervals)
    print()
    scheffe_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Scheffe")
    print_confidence_intervals(scheffe_intervals)
    print()
    sidak_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Sidak")
    print_confidence_intervals(sidak_intervals)
    print()
    tukey_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Tukey")
    print_confidence_intervals(tukey_intervals)
    print("\n**********")

    print("One Pairwise Comparison")
    C = [[1, 0, -1, 0], ]
    print("Linear combinations C: {}\n".format(C))
    bonferroni_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Bonferroni")
    print_confidence_intervals(bonferroni_intervals)
    print()
    scheffe_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Scheffe")
    print_confidence_intervals(scheffe_intervals)
    print()
    tukey_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Tukey")
    print_confidence_intervals(tukey_intervals)
    print()
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "Best")
    print_confidence_intervals(best_intervals)
    print("\n**********")

    print("Orthogonal Contrasts with Best Option")
    C = [[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]]
    print("Linear combinations C: {}\n".format(C))
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "best")
    print_confidence_intervals(best_intervals)
    print("\n**********")

    print("Contrast but not Orthogonal with Best Option")
    C = [[1, 0, -1, 0], [-1, -1, 1, 1]]
    print("Linear combinations C: {}\n".format(C))
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "best")
    print_confidence_intervals(best_intervals)
    print("\n**********")

    print("Contrast and Pairwise with Best Option")
    C = [[1, 0, -1, 0], [-1, 0, 0, 1]]
    print("Linear combinations C: {}\n".format(C))
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "best")
    print_confidence_intervals(best_intervals)
    print("\n**********")

    print("Orthogonal Contrast and Pairwise with Best Option")
    C = [[1, 0, -1, 0], [0, 1, 0, -1]]
    print("Linear combinations C: {}\n".format(C))
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "best")
    print_confidence_intervals(best_intervals)
    print("\n**********")

    print("Not Contrast with Best Option")
    C = [ [1, 1, 1, 1]]
    print("Linear combinations C: {}\n".format(C))
    best_intervals = ANOVA1_CI_linear_combs(x, alpha, C, "best")
    print_confidence_intervals(best_intervals)


def test_ANOVA1_test_linear_combs():
    print("Test: ANOVA1_test_linear_combs\n")

    # DATA from https://itl.nist.gov/div898/handbook/prc/section4/prc436.htm#contrastex2
    alpha = 0.05
    x = [[6.9, 5.4, 5.8, 4.6, 4.0], [8.3, 6.8, 7.8, 9.2, 6.5], [8.0, 10.5, 8.1, 6.9, 9.3], [5.8, 3.8, 6.1, 5.6, 6.2]]

    print("X: {}\n".format(x))

    print("**********")
    print("Orthogonal Contrast")
    C = [[1, 0, -1, 0]]
    d = [0]
    print("Linear combinations C: {}\nalpha: {}\nTested values d: {}\n".format(C, alpha, d))
    bonferroni_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Bonferroni")
    print_test_outcomes(C, d, bonferroni_test_outcomes)
    print()
    scheffe_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Scheffe")
    print_test_outcomes(C, d, scheffe_test_outcomes)
    print()
    sidak_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Sidak")
    print_test_outcomes(C, d, sidak_test_outcomes)
    print()
    tukey_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Tukey")
    print_test_outcomes(C, d, tukey_test_outcomes)
    print("\n**********")

    print("Orthogonal Contrast")
    C = [[1, 0, 0, -1]]
    d = [0]
    print("Linear combinations C: {}\nalpha: {}\nTested values d: {}\n".format(C, alpha, d))
    bonferroni_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Bonferroni")
    print_test_outcomes(C, d, bonferroni_test_outcomes)
    print()
    scheffe_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Scheffe")
    print_test_outcomes(C, d, scheffe_test_outcomes)
    print()
    sidak_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Sidak")
    print_test_outcomes(C, d, sidak_test_outcomes)
    print()
    tukey_test_outcomes = ANOVA1_test_linear_combs(x, alpha, C, d, "Tukey")
    print_test_outcomes(C, d, tukey_test_outcomes)

def test_ANOVA2_partition_TSS():
    print("Test: ANOVA2_partition_TSS\n")
    # Detergant vs Water Temp data
    # https://www.youtube.com/watch?v=whp4bSeJC7I
    x = [[[45, 39, 46], [43, 46, 41], [55, 48, 53]], 
        [[37, 32, 43], [40, 37, 46], [56, 51, 53]], 
        [[53, 47, 46], [37, 41, 40], [36, 32, 35]]]

    ssa, ssb, ssab, sse, sstotal = ANOVA2_partition_TSS(x)
    print("X: {}\n".format(x))
    print("SStotal:\t{:.4f}\nSSa:\t\t{:.4f}\nSSb:\t\t{:.4f}\nSSab:\t\t{:.4f}\nSSe:\t\t{:.4f}".format(sstotal, ssa, ssb, ssab, sse))

def test_ANOVA2_MLE():
    print("Test: ANOVA2_MLE\n")
    # Detergant vs Water Temp data
    # https://www.youtube.com/watch?v=whp4bSeJC7I
    x = [[[45, 39, 46], [43, 46, 41], [55, 48, 53]], 
        [[37, 32, 43], [40, 37, 46], [56, 51, 53]], 
        [[53, 47, 46], [37, 41, 40], [36, 32, 35]]]

    xbar, ai_hat_list, bj_hat_list, interaction_hat_list = ANOVA2_MLE(x)
    print("X: {}\n".format(x))
    print("Mu:\t\t{:.4f}\nai:\t\t{}\nbj:\t\t{}\ninteraction:\t{}".format(xbar, ai_hat_list, bj_hat_list, interaction_hat_list))

def test_ANOVA2_test_equality():
    print("Test: ANOVA2_test_equality\n")
    # Detergant vs Water Temp data
    # https://www.youtube.com/watch?v=whp4bSeJC7I
    x = [[[45, 39, 46], [43, 46, 41], [55, 48, 53]], 
        [[37, 32, 43], [40, 37, 46], [56, 51, 53]], 
        [[53, 47, 46], [37, 41, 40], [36, 32, 35]]]

    print("X: {}\n".format(x))
    ANOVA2_test_equality(x, 0.1, "a", print_table = True)
    ANOVA2_test_equality(x, 0.1, "b", print_table = False)
    ANOVA2_test_equality(x, 0.1, "ab", print_table = False)

def print_lr_data(X: list, y: list, predictor_names: list, response_name: str):
    print("x0\t{}\t{}\t{}".format(predictor_names[0], predictor_names[1], response_name))
    for i in range(len(X)):
        print("{}\t{}\t{}\t{}".format(X[i][0], X[i][1], X[i][2], y[i]))

def prepare_mult_lr_data() -> list:
    k = 2
    predictor_names = ["Height", "Weight"]
    response_name = "Distance"
    height = [42.8, 63.5, 37.5, 39.5, 45.5, 38.5, 43.0, 22.5, 37.0, 23.5, 33.0, 58.0]
    weight = [40.0, 93.5, 35.5, 30.0, 52.0, 17.0, 38.5, 8.5, 33.0, 9.5, 21.0, 79.0]
    distance = [37.0, 49.5, 34.5, 36.0, 43.0, 28.0, 37.0, 20.0, 33.5, 30.5, 38.5, 47.0]

    # form the data
    y = np.array(distance)
    X = np.ones((len(height), k + 1))
    X[:, 1] = height
    X[:, 2] = weight

    return [X, y, predictor_names, response_name]

def test_Mult_LR_least_squares():
    print("Test: Mult_LR_least_squares\n")

    X, y, predictor_names, response_name = prepare_mult_lr_data()
    print_lr_data(X, y, predictor_names, response_name)
    beta_hat, biased_sigma_square_hat, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    print("\nBetaHat:\t\t\t{}\nBiased Sigma Square Hat:\t{:.4f}\nUnbiased Sigma Square Hat:\t{:.4f}".format(beta_hat, biased_sigma_square_hat, unbiased_sigma_square_hat))

def test_Mult_LR_partition_TSS():
    print("Test: Mult_LR_partition_TSS\n")

    X, y, predictor_names, response_name = prepare_mult_lr_data()
    print_lr_data(X, y, predictor_names, response_name)
    Totalss, REGss, RSS = Mult_LR_partition_TSS(X, y)
    print("\nTotalSS:\t{:.4f}\nREGss:\t\t{:.4f}\nRSS:\t\t{:.4f}".format(Totalss, REGss, RSS))

def test_Mult_norm_LR_simul_CI():
    print("Test: Mult_norm_LR_simul_CI\n")

    alpha = 0.1
    X, y, predictor_names, response_name = prepare_mult_lr_data()
    print("alpha: ", alpha)
    print_lr_data(X, y, predictor_names, response_name)
    print()
    intervals = Mult_norm_LR_simul_CI(X, y, alpha)

    for i in range(len(intervals)):
        print("Beta{}: ({:.4f}, {:.4f}))".format(i, intervals[i][0], intervals[i][1]))

def test_Mult_norm_LR_CR():
    print("Test: Mult_norm_LR_CR\n")

    C = [[1, 0, 1], [0, 1, 1]]
    alpha = 0.1
    X, y, predictor_names, response_name = prepare_mult_lr_data()
    print_lr_data(X, y, predictor_names, response_name)
    print("\nC:\t{}\nalpha:\t{}".format(C, alpha))
    print()

    eigen_vals, center = Mult_norm_LR_CR(X, y, C, alpha)
    print("Eigen Values:\t{}\nCenter:\t\t{}".format(eigen_vals, center))

def test_Mult_norm_LR_is_in_CR():
    print("Test: Mult_norm_LR_is_in_CR\n")

    alpha = 0.1
    c0 = [20]
    C = [[1, 0, 1],]
    X, y, predictor_names, response_name = prepare_mult_lr_data()

    print_lr_data(X, y, predictor_names, response_name)
    beta_hat, _, _ = Mult_LR_Least_squares(X, y)
    print("BetaHat:\n\t{}".format(beta_hat))
    print()

    in_region = Mult_norm_LR_is_in_CR(X, y, C, c0, alpha)
    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))
    print("is c0 in confidence region: {}\n".format(in_region))
    print("****************")

    c0 = [40]
    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))
    in_region = Mult_norm_LR_is_in_CR(X, y, C, c0, alpha)
    print("is c0 in confidence region: {}".format(in_region))

def test_Mult_norm_LR_test_general():
    def print_lr_hypothesis(C:list, c0: list):
        print("H0: ")
        for i in range(len(C)):
            ci = C[i]
            for j in range(len(ci)):
                print("\t({} * Beta{}) {}".format(ci[j], j + 1, "+" if j < (len(ci) - 1) else ""), end=" ")

            print("= ", c0[i], end = "")
            print()

    print("Test: Mult_norm_LR_test_general\n")

    alpha = 0.1
    C = [[1, 0, 1], [1, 1, 0]]
    c0 = [21, 21]
    X, y, predictor_names, response_name = prepare_mult_lr_data()

    print_lr_data(X, y, predictor_names, response_name)
    beta_hat, _, _ = Mult_LR_Least_squares(X, y)
    print("BetaHat:\n\t{}".format(beta_hat))
    print()

    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))

    rejected = Mult_norm_LR_test_general(X, y, C, c0, alpha)
    print_lr_hypothesis(C, c0)
    print("{}\n".format("rejected" if rejected else "not rejected"))
    print("****************")

    C = [[1, 0, 1], [1, 1, 0]]
    c0 = [21, 22]
    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))

    rejected = Mult_norm_LR_test_general(X, y, C, c0, alpha)
    print_lr_hypothesis(C, c0)
    print("{}\n".format("rejected" if rejected else "not rejected"))
    print("****************")

    C = [[1, 0, 1]]
    c0 = [15]
    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))

    rejected = Mult_norm_LR_test_general(X, y, C, c0, alpha)
    print_lr_hypothesis(C, c0)
    print("{}\n".format("rejected" if rejected else "not rejected"))
    print("****************")

    C = [[1, 0, 1]]
    c0 = [40]
    print("C:\t{}\nc0:\t{}\nalpha:\t{}\n".format(C, c0, alpha))

    rejected = Mult_norm_LR_test_general(X, y, C, c0, alpha)
    print_lr_hypothesis(C, c0)
    print("{}".format("rejected" if rejected else "not rejected"))

def test_Mult_norm_LR_test_comp():
    def print_lr_hypothesis():
        print("H0: ")
        print("\t", end="")
        for j in j_list:
            print("Beta{} = ".format(j), end = "")
        print("0, {}".format("rejected" if rejected else "not rejected"))

    print("Test: Mult_norm_LR_test_comp\n")

    alpha = 0.1
    j_list = [2]
    X, y, predictor_names, response_name = prepare_mult_lr_data()

    print_lr_data(X, y, predictor_names, response_name)
    beta_hat, _, _ = Mult_LR_Least_squares(X, y)
    print("BetaHat:\n\t{}".format(beta_hat))
    print()

    rejected = Mult_norm_LR_test_comp(X, y, j_list, alpha)
    print("j_list:\t{}\nalpha:\t{}\n".format(j_list, alpha))
    print_lr_hypothesis()
    print("\n****************")

    j_list = [0]
    rejected = Mult_norm_LR_test_comp(X, y, j_list, alpha)
    print("j_list:\t{}\nalpha:\t{}\n".format(j_list, alpha))
    print_lr_hypothesis()
    print("\n****************")

    j_list = [0, 1]
    rejected = Mult_norm_LR_test_comp(X, y, j_list, alpha)
    print("j_list:\t{}\nalpha:\t{}\n".format(j_list, alpha))
    print_lr_hypothesis()

def test_Mult_norm_LR_test_linear_reg():
    def print_lr_hypothesis():
        print("H0: ")
        print("\t", end="")
        for j in range(len(X[0])):
            print("Beta{} = ".format(j), end = "")
        print("0, {}".format("rejected" if rejected else "not rejected"))

    print("Test: Mult_norm_LR_test_linear_reg\n")

    alpha = 0.1
    X, y, predictor_names, response_name = prepare_mult_lr_data()

    print_lr_data(X, y, predictor_names, response_name)
    beta_hat, _, _ = Mult_LR_Least_squares(X, y)
    print("BetaHat:\n\t{}".format(beta_hat))
    print()

    rejected = Mult_norm_LR_test_linear_reg(X, y, alpha)
    print_lr_hypothesis()

def test_Mult_norm_LR_pred_CI():
    def print_output():
        for i in range(len(D)):
            di = D[i]
            bound_i = bound[i]
            print("\tdi: {}, interval: ({:.4f}, {:.4f})".format(di, bound_i[0], bound_i[1]))

    print("Test: Mult_norm_LR_pred_CI\n")

    alpha = 0.1
    D = [[1, 63.5, 93.5]]
    X, y, predictor_names, response_name = prepare_mult_lr_data()

    print_lr_data(X, y, predictor_names, response_name)
    print()

    bound = Mult_norm_LR_pred_CI(X, y, D, alpha, "best")
    print("Best Option")
    print_output()
    print("\n****************")

    D = [[1, 63.5, 93.5], [1, 42.8, 40.0]]
    bound = Mult_norm_LR_pred_CI(X, y, D, alpha, "best")
    print("Best Option")
    print_output()
    print("\n****************")

    D = [[1, 63.5, 93.5], [1, 42.8, 40.0]]
    bound = Mult_norm_LR_pred_CI(X, y, D, alpha, "scheffe")
    print("Scheffe Option")
    print_output()
    print("\n****************")

    bound = Mult_norm_LR_pred_CI(X, y, D, alpha, "bonferroni")
    print("Bonferroni Option")
    print_output()


def main():
    # use -w argument to wait a user input between tests for better examination of the test
    args = sys.argv[1:]
    short_ops = "w"
    use_input_to_wait = False

    arguments, values = getopt.getopt(args, short_ops)
    if len(arguments) > 0:
        arg = arguments[0][0]
        if arg == '-w':
            use_input_to_wait = True

    functions_to_test = [test_ANOVA1_partition_TSS, test_ANOVA1_test_equality, test_ANOVA1_is_contrast, test_ANOVA1_is_orthogonal,
                        test_Bonferroni_correction, test_Sidak_correction, test_ANOVA1_CI_linear_combs, test_ANOVA1_test_linear_combs,
                        test_ANOVA2_partition_TSS, test_ANOVA2_MLE, test_ANOVA2_test_equality, test_Mult_LR_least_squares,
                        test_Mult_LR_partition_TSS, test_Mult_norm_LR_simul_CI, test_Mult_norm_LR_CR, test_Mult_norm_LR_is_in_CR, test_Mult_norm_LR_test_general,
                        test_Mult_norm_LR_test_comp, test_Mult_norm_LR_test_linear_reg, test_Mult_norm_LR_pred_CI]

    #functions_to_test = []

    for func in functions_to_test:
        print("--------------------------")
        func()
        print("--------------------------")

        if use_input_to_wait:
            input()
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            print("\n")

if __name__ == '__main__':
    main()