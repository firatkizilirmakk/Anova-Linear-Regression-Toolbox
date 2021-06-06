import numpy as np
from scipy.stats import f as F_test
from scipy.stats import t as t_test
from qdist import qdist

################################ ONE WAY ANOVA HELPER FUNCTIONS END ###############################

def ANOVA_sample_mean(x: list) -> float:
    """
        Calculates Xbar for one-way anova
    """
    assert len(x) > 0, "Empty list"

    I = len(x)
    total_num = 0
    total_sum = 0

    for i in range(I):
        ni = len(x[i])
        total_num += ni
        for j in range(ni):
            total_sum += x[i][j]

    return total_sum / total_num

def ANOVA_num_of_total_elements(x: list) -> int:
    """
        Calculate total number of elements in one-way anova
        n1 + n2 ... + nI = n
    """
    assert len(x) > 0, "Empty list"

    I = len(x)
    total = 0
    for i in range(I):
        total += len(x[i])
    return total

def ANOVA_print_table(args: list) -> None:
    df_between, df_within, df_total,\
        ss_between, ss_within,\
        ms_between, ms_within, F_value = args

    print("{:15s}|{:15s}|{:15s}|{:15s}|{:15s}".format("Source", "df", "SS", "MS", "F"))
    print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.4f}".format("Between Groups", df_between, ss_between, ms_between, F_value))
    print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|".format("Within Groups", df_within, ss_within, ms_within))
    print("{:15s}|{:15.4f}|".format("Total", df_total, ss_between + ss_within))

def ANOVA_get_group_sizes(groups: list) -> list:
    """
        Finds each group size and returns within a list
    """
    assert len(groups) > 0, "Empty list"

    group_sizes = []
    for group in groups:
        group_sizes.append(len(group))
    return group_sizes

def ANOVA_get_group_means(groups: list) -> list:
    """
        Calculates means of groups seperately
    """
    assert len(groups) > 0, "Empty list"

    means = []
    for group in groups:
        means.append(np.mean(group))
    return means

def ANOVA_all_pairwise_linear_combs(C: list) -> bool:
    """
        Checks whether the given C matrix consists of only pairwise comparisons or not
    """
    assert len(C) > 0, "Empty list"

    all_pairwise = True
    for linear_comb in C:
        if not (linear_comb.count(1) == 1 and linear_comb.count(-1) == 1):
            all_pairwise = False
            break
    return all_pairwise

def ANOVA_all_equal_group_sizes(groups: list) -> bool:
    """
        Checks whether all the group sizes are equal
    """
    assert len(groups) > 0, "Empty list"

    equal_sizes = True
    group_zero_size = len(groups[0])
    for group in groups[1:]:
        if len(group) != group_zero_size:
            equal_sizes = False
            break
    return equal_sizes

def ANOVA_all_contrasts(C: list) -> bool:
    """
        Checks whether the given C matrix consists of only contrasts
    """
    assert len(C) > 0, "Empty list"

    all_contrasts = True
    for c in C:
        if not ANOVA1_is_contrast(c):
            all_contrasts = False
            break

    return all_contrasts

def ANOVA_all_orthogonal_contrasts(C: list, group_sizes: list) -> bool:
    """
        Checks whether the given C matrix consists of only orthogonal contrasts
    """
    # need at least 2 contrasts
    if len(C) < 2:
        print("\tWarning! There must be at least 2 contrasts for orthogonality")
        return False

    all_orthogonal = True
    for i in range(len(C)):
        for j in range(i, len(C)):
            if i != j:
                c1 = C[i]
                c2 = C[j]
                if ANOVA1_is_contrast(c1) and ANOVA1_is_contrast(c2):
                    if not ANOVA1_is_orthogonal(group_sizes, c1, c2, print_info = False):
                        all_orthogonal = False
                else:
                    all_orthogonal = False

    return all_orthogonal

def ANOVA_compare_confidence_intervals(intervals1: list, intervals2: list) -> int or None:
    """
        Compares two confidence intervals and returns an integer stating difference.
        :param intervals1: List of confidence intervals(tuples) to be compared
        :param intervals2: List of confidence intervals(tuples) to be compared
        :return: Returns an integer. 
                 It is positive if intervals1 >  intervals2
                       negative if intervals1 <  intervals2
                       zero     if intervals1 == intervals2
    """
    assert len(intervals1) == len(intervals2), "Lenghts of intervals must be equal"

    num_narrower_interval1 = 0
    num_narrower_interval2 = 0
    for i in range(len(intervals1)):
        interval1 = intervals1[i]
        interval2 = intervals2[i]

        if abs(interval1[0] - interval1[1]) > abs(interval2[0] - interval2[1]):
            num_narrower_interval2 += 1
        elif abs(interval1[0] - interval1[1]) < abs(interval2[0] - interval2[1]):
            num_narrower_interval1 += 1

    return num_narrower_interval2 - num_narrower_interval1

################################# ONE WAY ANOVA HELPER FUNCTIONS END ################################

################################### ONE WAY ANOVA FUNCTIONS START ###################################
def ANOVA1_partition_TSS(x: list) -> list:
    """
        :param x: Dataset Xij for j = 1, ..., n and i = 1, ..., I
        :return: Returns SStotal, SSwithin, SSbetween wrt one-way layout
    """

    I  = len(x)
    xbar = ANOVA_sample_mean(x)

    ss_within  = 0
    ss_between = 0
    for i in range(I):
        ni = len(x[i])
        xbar_i = np.mean(x[i])
        ss_between += (ni * ((xbar_i - xbar) ** 2))
        for j in range(ni):
            ss_within += ((x[i][j] - xbar_i) ** 2)

    ss_total = ss_within + ss_between

    return [ss_total, ss_within, ss_between]

def ANOVA1_test_equality(x: list, alpha: float, print_info: bool) -> list:
    """
        Tests the equality of the means

        :param x: Dataset Xij for j = 1, ..., n and i = 1, ..., I
        :param alpha: Significance level
        :param print_info: Determines whether to print info on terminal

        :return: returns all the necessary stuff in one-way anova table
    """

    I = len(x)
    n = ANOVA_num_of_total_elements(x)

    df_between = I - 1
    df_within  = n - I
    df_total   = n - 1

    ss_total, ss_within, ss_between = ANOVA1_partition_TSS(x)
    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within
    F_value    = ms_between / ms_within

    p_value  = F_test.sf(F_value, df_between, df_within)
    rejected = p_value < alpha

    args_to_print = [df_between, df_within, df_total, ss_between, ss_within, ms_between, ms_within, F_value]
    if print_info:
        ANOVA_print_table(args_to_print)
        if rejected:
            print("\np-value = {:.6f} < alpha = {:.6f}.\nNull hypothesis is rejected at significance level alpha = {:.6f}".format(p_value, alpha, alpha))
        else:
            print("\np-value = {:.6f} >= alpha = {:.6f}.\nFailed to reject null hypothesis at significance level alpha = {:.6f}".format(p_value, alpha, alpha))

    return args_to_print


def ANOVA1_is_contrast(c: list) -> bool:
    """
        Checks whether given vector is a contrast

        :param c: vector (1d list expecting) to check whether it is a contrast
        :return: True if the given vector is contrast, false otherwise
    """
    if c.count(0) == len(c):
        return False
    return np.sum(c) == 0

def ANOVA1_is_orthogonal(n: list, c1: list, c2:list, print_info: bool) -> bool:
    """
        Checks whether given vector is an orthogonal contrast

        :param n: list of group sizes
        :param c1: vector (1d list expecting) to check whether it is a contrast
        :param c2: vector (1d list expecting) to check whether it is a contrast

        :return: True if the given vector is contrast, false otherwise
    """
    assert len(c1) == len(c2), "Shapes of c vectors must be equal"
    assert len(n) == len(c1),  "Shapes of n and c vectors must be equal"

    I = len(c1)
    is_contrast_c1 = ANOVA1_is_contrast(c1)
    is_contrast_c2 = ANOVA1_is_contrast(c2)

    both_contrasts = True
    if print_info:
        if not is_contrast_c1:
            print("Warning! Vector {} is not a contrast".format(c1))
            both_contrasts = False
        if not is_contrast_c2:
            print("Warning! Vector {} is not a contrast".format(c2))
            both_contrasts = False

    # it is not valid to be orthogonal when one of them is not a contrast
    if not both_contrasts:
        return False

    total_sum = 0
    for i in range(I):
        ni = n[i]
        c1_i = c1[i]
        c2_i = c2[i]
        total_sum += ((c1_i * c2_i) / ni)

    return total_sum == 0

def Bonferroni_correction(fwer_alpha: float, m: int) -> float:
    """
        Calculates the significance level using Bonferroni method

        :param fwer_alpha: Significance level of family wise error
        :param m: Number of tests
        :return: Returns a Bonferroni corrected FWER alpha value
    """
    assert m != 0, "m can not be 0"
    return fwer_alpha / float(m)

def Sidak_correction(fwer_alpha: float, m: int) -> float:
    """
        Calculates the significance level using Sidak method

        :param fwer_alpha: Significance level of family wise error
        :param m: Number of tests
        :return: Returns a Sidak corrected FWER alpha value
    """
    assert m != 0, "m can not be 0"
    return 1 - pow((1 - fwer_alpha), (1 / float(m)))


def ANOVA1_CI_linear_combs(x: list, alpha: float, C: list, method: str) -> list:
    """
        Calculates simultaneous confidence intervals for the given linear combinations in C matrix.
        If the given method is not valid for the given linear combinations then an empty list is returned.

        :param x: Dataset Xij for j = 1, ..., n and i = 1, ..., I
        :param alpha: Significance level of alpha
        :param C: m x I matrix where each row is a linear combination
        :param method: Method to calculate the linear combination e.g. Bonferroni
        :return: Returns a list of confidence intervals holding simultaneously with 1-alpha probability
                 if the given method and linear combinations do not conflict.
    """

    def ANOVA_select_confidence_interval_by_comparing(intervals: list, method_names: list, print_info: bool) -> list:
        """
            Compares the given confidence intervals and select the narrow one
        """
        intervals1, intervals2 = intervals[0], intervals[1]
        method_name1, method_name2 = method_names[0], method_names[1]

        interval_comparison = ANOVA_compare_confidence_intervals(intervals1, intervals2)
        selected_methods_confidence_intervals = None
        if interval_comparison < 0:
            selected_methods_confidence_intervals = intervals1
            if print_info:
                print("\t{} has a narrower CI than {}".format(method_name1, method_name2))
        elif interval_comparison > 0:
            selected_methods_confidence_intervals = intervals2
            if print_info:
                print("\t{} has a narrower CI than {}".format(method_name2, method_name1))
        else:
            selected_methods_confidence_intervals = intervals1
            if print_info:
                print("\tThere is no dominant CI between {} and {}.\nSelecting the one produced by {}".format(method_name1, method_name2, method_name1))

        return selected_methods_confidence_intervals

    def ANOVA_scheffe_common_computation(f_value, f_value_coefficient) -> list:
        """
            Common computation part of the scheffe method when it is used with 
            either a contrast or no contrast.
            :return: Returns the confidence interval holding simultaneosuly
                     using Scheffe's method
        """
        intervals = []
        for linear_comb in C:
            point_estimate = 0
            estimator_variance = 0
            for i, ci in enumerate(linear_comb):
                ni = len(x[i])
                point_estimate += (ci * group_means[i])
                estimator_variance += ((ci * ci) / ni)

            std_error = np.sqrt(estimator_variance * ms_within)
            tail = np.sqrt(f_value_coefficient * f_value) * std_error
            interval = (point_estimate - tail, point_estimate + tail)
            intervals.append(interval)

        return intervals

    def ANOVA_scheffe_with_conrasts() -> list:
        """
            Calculates the simultaneous confidence intervals using Scheffe's method
            with contrasts
            :return: Returns the confidence interval holding simultaneosuly
                     using Scheffe's method
        """
        f_value = F_test.ppf(1 - alpha, I - 1, df_within)
        intervals = ANOVA_scheffe_common_computation(f_value, I - 1)
        return intervals

    def ANOVA_scheffe_with_no_contrasts() -> list:
        """
            Calculates the simultaneous confidence intervals using Scheffe's method
            without contrasts
            :return: Returns the confidence interval holding simultaneosuly
                     using Scheffe's method
        """
        f_value = F_test.ppf(1 - alpha, I, df_within)
        intervals = ANOVA_scheffe_common_computation(f_value, I)
        return intervals

    def ANOVA_bonferroni_sidak_common(t_value) -> list:
        """
            Common computation part for both Bonferroni and Sidak approaches
            :return: Returns the confidence interval holding simultaneosuly
                     using Bonferroni or Sidak wrt given t_value
        """
        intervals = []

        for linear_comb in C:
            point_estimate = 0
            estimator_variance = 0
            for i, ci in enumerate(linear_comb):
                ni = len(x[i])
                point_estimate += (ci * group_means[i])
                estimator_variance += ((ci * ci) / ni)

            std_error = np.sqrt(estimator_variance * ms_within)
            tail = t_value * std_error
            interval = (point_estimate - tail, point_estimate + tail)
            intervals.append(interval)

        return intervals

    def ANOVA_bonferroni() -> list:
        """
            Calculates the simultaneous confidence intervals using Bonferroni method.
            :return: Returns the confidence interval holding simultaneosuly
                     using Bonferroni method
        """
        alpha_zero = Bonferroni_correction(alpha, num_of_hypothesis)
        t_value = t_test.ppf(1 - (alpha_zero / 2), (df_within))
        intervals = ANOVA_bonferroni_sidak_common(t_value)

        return intervals

    def ANOVA_sidak() -> list:
        """
            Calculates the simultaneous confidence intervals using Sidak method.
            :return: Returns the confidence interval holding simultaneosuly
                     using Sidak method
        """
        alpha_zero = Sidak_correction(alpha, num_of_hypothesis)
        t_value = t_test.ppf(1 - (alpha_zero / 2), (df_within))
        intervals = ANOVA_bonferroni_sidak_common(t_value)

        return intervals

    def ANOVA_tukey() -> list:
        """
            Calculates the simultaneous confidence intervals using Tukey method
            which is only called for the pairwise comparisons.
            :return: Returns the confidence interval holding simultaneosuly
                     using Tukey method
        """
        intervals = []
        n = len(x[0]) # since all groups sizes are equal

        q_value = qdist(alpha, I, df_within)
        tail = (q_value / np.sqrt(2)) * np.sqrt(ms_within * 2 / n)

        for linear_comb in C:
            first_index  = linear_comb.index(1)
            second_index = linear_comb.index(-1)
            first_group_mean  = group_means[first_index]
            second_group_mean = group_means[second_index]

            point_estimate = first_group_mean - second_group_mean
            interval = (point_estimate - tail, point_estimate + tail)
            intervals.append(interval)

        return intervals

    # check the input types
    assert len(x) != 0 and len(C) != 0, "Invalid input"
    assert str(method).lower() in ["scheffe", "tukey", "bonferroni", "sidak", "best"], "Invalid method type"

    I = len(x)
    for c in C:
        assert len(c) == I, "Invalid c: {}".format(c)

    [df_between, df_within, df_total, ss_between, ss_within, ms_between, ms_within, F_value] = ANOVA1_test_equality(x, alpha, print_info = False)

    group_means = ANOVA_get_group_means(x)
    group_sizes = ANOVA_get_group_sizes(x)
    num_of_hypothesis = len(C)

    confidence_intervals = []
    if str(method).lower() == "scheffe":
        if ANOVA_all_contrasts(C):
            print("\tScheffe 2.8")
            confidence_intervals = ANOVA_scheffe_with_conrasts()
        else:
            print("\tScheffe 2.7")
            confidence_intervals = ANOVA_scheffe_with_no_contrasts()

    elif str(method).lower() == "tukey":
        print("\tTukey")
        if ANOVA_all_equal_group_sizes(x) and ANOVA_all_pairwise_linear_combs(C):
            confidence_intervals = ANOVA_tukey()
        else:
            print("\tError! Given C matrix is not valid for Tukey (expecting pairwise comparisons)")

    elif str(method).lower() == "bonferroni":
        print("\tBonferroni")
        confidence_intervals = ANOVA_bonferroni()

    elif str(method).lower() == "sidak":
        print("\tSidak")
        if ANOVA_all_orthogonal_contrasts(C, group_sizes):
            confidence_intervals = ANOVA_sidak()
        else:
            print("\tError! Given C matrix is not valid for Sidak (expecting independent tests, e.g. orthogonal contrasts)")

    elif str(method).lower() == "best":
        if ANOVA_all_contrasts(C):
            if ANOVA_all_orthogonal_contrasts(C, group_sizes):
                if ANOVA_all_pairwise_linear_combs(C):
                    if ANOVA_all_equal_group_sizes(x):
                        print("\tComparing Tukey with Sidak")
                        tukey_confidence_intervals = ANOVA_tukey()
                        sidak_confidence_intervals = ANOVA_sidak()
                        confidence_intervals = ANOVA_select_confidence_interval_by_comparing([tukey_confidence_intervals, sidak_confidence_intervals], 
                                                                            ["Tukey", "Sidak"], print_info=True)
                    else:
                        print("\tError! Not implemented Tukey with different group sizes")
                else:
                    print("\tComparing Scheffe 2.8 with Sidak")
                    scheffe_confidence_intervals = ANOVA_scheffe_with_conrasts()
                    sidak_confidence_intervals   = ANOVA_sidak()
                    confidence_intervals = ANOVA_select_confidence_interval_by_comparing([scheffe_confidence_intervals, sidak_confidence_intervals], 
                                                                            ["Scheffe 2.8", "Sidak"], print_info=True)
            else:
                if ANOVA_all_pairwise_linear_combs(C):
                    print("\tComparing Tukey with Bonferroni")
                    tukey_confidence_intervals = ANOVA_tukey()
                    bonferroni_confidence_intervals = ANOVA_bonferroni()
                    confidence_intervals = ANOVA_select_confidence_interval_by_comparing([tukey_confidence_intervals, bonferroni_confidence_intervals], 
                                                                            ["Tukey", "Bonferroni"], print_info=True)
                else:
                    print("\tComparing Scheffe 2.8 with Bonferroni")
                    scheffe_confidence_intervals = ANOVA_scheffe_with_conrasts()
                    bonferroni_confidence_intervals = ANOVA_bonferroni()
                    confidence_intervals = ANOVA_select_confidence_interval_by_comparing([scheffe_confidence_intervals, bonferroni_confidence_intervals], 
                                                                            ["Scheffe 2.8", "Bonferroni"], print_info=True)
        else:
            print("\tComparing Scheffe 2.7 with Bonferroni")
            scheffe_confidence_intervals = ANOVA_scheffe_with_no_contrasts()
            bonferroni_confidence_intervals = ANOVA_bonferroni()
            confidence_intervals = ANOVA_select_confidence_interval_by_comparing([scheffe_confidence_intervals, bonferroni_confidence_intervals], 
                                                                            ["Scheffe 2.7", "Bonferroni"], print_info=True)

    return confidence_intervals

def ANOVA1_test_linear_combs(x: list, alpha: float, C: list, d: list, method: str) -> list:
    """
        Tests the null hypotheses defined by the linear combinations given in C
        being equal to di in d list.

        :param x: Dataset Xij for j = 1, ..., n and i = 1, ..., I
        :param alpha: Significance level of alpha
        :param C: m x I matrix where each row is a linear combination
        :param d: list of values to check whether the linear combinations are equal to di
        :param method: Method to calculate the linear combination e.g. Bonferroni
        :return: Returns a list of null hypothesis test outputs consisting of 0s and 1s
                 where 0 means the corresponding hypothesis not rejected
                 while 1 means the hypothesis is rejected.
    """
    confidence_intervals = ANOVA1_CI_linear_combs(x, alpha, C, method)

    if len(confidence_intervals) != len(d):
        print("\tUnmatched confidence intervals and d list")
        return []

    test_outcomes = []
    for i, interval in enumerate(confidence_intervals):
        di = d[i]
        if interval[0] <= di <= interval[1]:
            test_outcomes.append(0) # not rejected
        else:
            test_outcomes.append(1) # rejected

    return test_outcomes

################################### ONE WAY ANOVA FUNCTIONS END ###################################

################################### TWO WAY ANOVA HELPER FUNCTIONS START ###################################

def ANOVA2_sample_mean(x: list) -> float:
    """
        Calculates the Xbar over all samples
    """
    assert len(x) != 0, "Invalid lenght of list"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    sum = 0
    for i in range(I):
        for j in range(J):
            for k in range(K):
                sum += x[i][j][k]
    return sum / (I * J * K)

def ANOVA2_compute_xbar_i(x: list) -> list:
    """
        Calculates the Xbar_i for the I factor
    """
    assert len(x) != 0, "Invalid lenght of list"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    xbar_i_list = []
    for i in range(I):
        xbar_i = 0
        for j in range(J):
            for k in range(K):
                xbar_i += x[i][j][k]
        xbar_i_list.append(xbar_i / (J * K))

    return xbar_i_list

def ANOVA2_compute_xbar_j(x: list) -> list:
    """
        Calculates the Xbar_j for the J factor
    """
    assert len(x) != 0, "Invalid lenght of list"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    xbar_j_list = []
    for j in range(J):
        xbar_j = 0
        for i in range(I):
            for k in range(K):
                xbar_j += x[i][j][k]
        xbar_j_list.append(xbar_j / (I * K))

    return xbar_j_list

def ANOVA2_compute_xbar_ij(x: list) -> list:
    """
        Calculates the Xbar_ij for the IJ interaction.
    """
    assert len(x) != 0, "Invalid lenght of list"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    xbar_ij_list = []
    for i in range(I):
        xbar_j_list = []
        for j in range(J):
            xbar_ij = 0
            for k in range(K):
                xbar_ij += x[i][j][k]
            xbar_j_list.append(xbar_ij / K)
        xbar_ij_list.append(xbar_j_list)

    return xbar_ij_list

################################### TWO WAY ANOVA HELPER FUNCTIONS END ###################################

###################################### TWO WAY ANOVA FUNCTIONS START #####################################

def ANOVA2_partition_TSS(x: list) -> list:
    """
        Calculates the partitions of the two-way anova
        :param x: Two way anova dataset Xijk, expecting to be 3 dimensional
        :return:  Returns a list of partitions like SStotal, SSa (for factor A) etc.
    """

    assert len(x) != 0 and len(x[0]) != 0 and len(x[0][0]) != 0, "Invalid input"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    xbar = ANOVA2_sample_mean(x)
    xbar_i_list  = ANOVA2_compute_xbar_i(x)
    xbar_j_list  = ANOVA2_compute_xbar_j(x)
    xbar_ij_list = ANOVA2_compute_xbar_ij(x)

    ssa = ssb = ssab = sse = sstotal = 0
    for i in range(I):
        ssb = 0
        ssa += ((xbar_i_list[i] - xbar) ** 2)
        for j in range(J):
            ssb += ((xbar_j_list[j] - xbar) ** 2) # computed more than once, it is ok
            ssab += ((xbar_ij_list[i][j] - xbar_i_list[i] - xbar_j_list[j] + xbar) ** 2)
            for k in range(K):
                sse += ((x[i][j][k] - xbar_ij_list[i][j]) ** 2)
                sstotal += ((x[i][j][k] - xbar) ** 2)

    ssa  = ssa * J * K
    ssb  = ssb * I * K
    ssab = ssab * K

    return [ssa, ssb, ssab, sse, sstotal]

def ANOVA2_MLE(x: list) -> list:
    """
        Calculates maximum likelihood estimates for mu, ai, bj, interaction
        where X = mu + ai + bj + interaction
        :param x: Two way anova dataset Xijk, expecting to be 3 dimensional
        :return:  Returns a list of estimates like xbar, ai_hat
    """

    assert len(x) != 0 and len(x[0]) != 0 and len(x[0][0]) != 0, "Invalid input"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    xbar = ANOVA2_sample_mean(x)
    xbar_i_list  = ANOVA2_compute_xbar_i(x)
    xbar_j_list  = ANOVA2_compute_xbar_j(x)
    xbar_ij_list = ANOVA2_compute_xbar_ij(x)

    ai_hat_list = []
    bj_hat_list = []
    interaction_hat_list = []

    for i in range(I):
        ai_hat_list.append(xbar_i_list[i] - xbar)
        bj_hat_list = []
        for j in range(J):
            bj_hat_list.append(xbar_j_list[j] - xbar) # computed more than once, but it is ok
            interaction_hat_list.append(xbar_ij_list[i][j] - xbar_j_list[j] - xbar_i_list[i] + xbar)

    return [xbar, ai_hat_list, bj_hat_list, interaction_hat_list]

def ANOVA2_test_equality(x: list, alpha: float, choice: str, print_table: bool):
    """
        Tests the null hypotheses given as a choice and prints the 2 way anova table.
        :param x: Two way anova dataset Xijk, expecting to be 3 dimensional
        :param alpha: Significance level of alpha
        :param choice: Either A, B or AB stating to test the null hypothesis 
                       for factor A, B or interaction
    """

    def ANOVA2_print_table():
        print("{:15s}|{:15s}|{:15s}|{:15s}|{:15s}|{:15s}".format("Source", "df", "SS", "MS", "F", "p value"))
        print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.10f}".format("A", df_a, ssa, msa, Fa, p_a_value))
        print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.10f}".format("B", df_b, ssb, msb, Fb, p_b_value))
        print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.4f}|{:15.10f}".format("A x B", df_ab, ssab, msab, Fab, p_ab_value))
        print("{:15s}|{:15.4f}|{:15.4f}|{:15.4f}|".format("Within", df_e, sse, mse))
        print("{:15s}|{:15.4f}|{:15.4f}|".format("Total", I * J * K - 1, sstotal))

    assert len(x) != 0 and len(x[0]) != 0 and len(x[0][0]) != 0, "Invalid input"
    assert alpha > 0, "Invalid alpha value"
    assert choice.lower() in ['a', 'b', 'ab'], "Invalid choice"

    I = len(x)
    J = len(x[0])
    K = len(x[0][0])

    df_a  = I - 1
    df_b  = J - 1
    df_ab = (I - 1) * (J - 1)
    df_e  = I * J * (K - 1)

    [ssa, ssb, ssab, sse, sstotal] = ANOVA2_partition_TSS(x)

    msa  = ssa / df_a
    msb  = ssb / df_b
    msab = ssab / df_ab
    mse  = sse / df_e

    Fa  = msa / mse
    Fb  = msb / mse
    Fab = msab / mse

    # compute p values
    p_a_value  = F_test.sf(Fa, df_a, df_e)
    p_b_value  = F_test.sf(Fb, df_b, df_e)
    p_ab_value = F_test.sf(Fab, df_ab, df_e)

    if print_table:
        # print the ANOVA2 table
        ANOVA2_print_table()

    test_outcome_a  = p_a_value  < alpha
    test_outcome_b  = p_b_value  < alpha
    test_outcome_ab = p_ab_value < alpha

    print()
    print("alpha = ", alpha)
    if choice.lower() == 'a':
        print("H0: a1 = ... a{} = 0 is {}".format(I, "rejected" if test_outcome_a else "not rejected"))
    elif choice.lower() == 'b':
        print("H0: b1 = ... b{} = 0 is {}".format(J, "rejected" if test_outcome_b else "not rejected"))
    else:
        print("H0: there is no interaction between a and b {}".format("rejected" if test_outcome_ab else "not rejected"))

######################################## TWO WAY ANOVA FUNCTIONS END ########################################

###################################### LINER REGRESSION FUNCTIONS START #####################################

def check_mult_lr_errors(X: list, y: list):
    assert len(X) != 0, "Empty X"
    assert len(y) != 0, "Empty y"
    assert len(X) == len(y), "Unmatched X and y lenghts"

    assert len(X[0]) > 1, "No predictor"
    assert ((X[:, 0] == 1).sum()) == len(X), "First column contains elements different than 1"

    k = len(X[0]) - 1
    rank = np.linalg.matrix_rank(X)
    assert (k + 1) == rank, "Invalid rank of X"

def least_square_sol_beta_heat(X: list, y: list) -> list:
    """
        Calculates the least square estimation for the Beta vector
        :param X: Design matrix
        :param y: Response vector
        :return:  Returns a list of estimated parameters of the Beta vector
    """
    check_mult_lr_errors(X, y)
    beta_hat  = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))
    return beta_hat

def rss_mult_lr(X: list, y: list) -> float:
    """
        Calculates the residual sum of squares
        :param X: Design matrix
        :param y: Response vector
        :return:  Returns the calculated RSS value
    """
    beta_hat = least_square_sol_beta_heat(X, y)
    RSS = np.matmul((y - np.matmul(X, beta_hat)).transpose(), (y - np.matmul(X, beta_hat)))
    return RSS

def regss_mult_lr(X: list, y: list) -> float:
    """
        Calculates the regression sum of squares
        :param X: Design matrix
        :param y: Response vector
        :return:  Returns the calculated REGss value
    """
    beta_hat = least_square_sol_beta_heat(X, y)
    y_bar = np.mean(y)
    y_hat = np.matmul(X, beta_hat)

    regss = 0
    for i in range(len(y)):
        diff = (y_hat[i] - y_bar) ** 2
        regss += diff

    return regss

def Mult_LR_Least_squares(X: list, y: list) -> list:
    """
        Calculates the least square estimates for the beta vector
        and the biased and unbiased sigma estimators

        :param X: Design matrix
        :param y: Response vector
        :return:  Returns a list of estimated parameters
    """

    # number of predictors
    k = len(X[0]) - 1

    beta_hat  = least_square_sol_beta_heat(X, y)
    RSS = rss_mult_lr(X, y)

    biased_sigma_square_hat   = RSS / len(y)
    unbiased_sigma_square_hat = RSS / (len(y) - k - 1)

    return [beta_hat, biased_sigma_square_hat, unbiased_sigma_square_hat]

def Mult_LR_partition_TSS(X: list, y: list) -> list:
    """
        Calculates the partitions into Totalss, REGss, RSS
        :param X: Design matrix
        :param y: Response vector
        :return:  Returns a list of partitions
    """
    RSS     = rss_mult_lr(X, y)
    REGss   = regss_mult_lr(X, y)
    Totalss = REGss + RSS

    return [Totalss, REGss, RSS]

def Mult_norm_LR_simul_CI(X: list, y: list, alpha: float) -> list:
    """
        Calculates simultaneous confidence intervals for the beta parameters
        using Bonferroni correction
        :param X: Design matrix
        :param y: Response vector
        :param alpha: Significance level of the tests
        :return:  Returns a list of confidence intervals(tuples)
    """

    check_mult_lr_errors(X, y)

    k = len(X[0]) - 1
    t_value = t_test.ppf(1 - (alpha / (2 * (k + 1))), (len(y) - k - 1))

    beta_hat, _, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    se = np.sqrt(unbiased_sigma_square_hat)

    in_sqrt = np.linalg.inv(np.matmul(X.transpose(), X))

    confidence_intervals = []
    for i in range(k + 1):
        beta_hat_i = beta_hat[i]
        tail = t_value * se * np.sqrt(in_sqrt[i][i])
        confidence_interval_i = (beta_hat_i - tail, beta_hat_i + tail)
        confidence_intervals.append(confidence_interval_i)

    return confidence_intervals

def Mult_norm_LR_CR(X: list, y: list, C: list, alpha: float) -> list:
    """
        Calculates 100(1 - alpha)% confidence region for C*Beta using
        (x-v)^T * A * (x-v) where A is the matrix whose eigenvalues define
        the width, height and other dimensions of the ellipsoid. Beta_hat stands
        for the center of the ellipsoid. A list containing this items is returned.

        :param X: Design matrix
        :param y: Response vector
        :param C: Matrix containing linear combinations of the beta parameters
        :param alpha: Significance level of the tests
        :return: Returns a list containing parameters of the ellipsoid.
                First item of the list is the decomposed eigenvalues for the A matrix.
    """
    check_mult_lr_errors(X, y)
    assert len(C) > 1, "There must be at least two rows of C"
    assert np.linalg.matrix_rank(C) == len(C), "Invalid rank of C matrix"

    k = len(X[0]) + 1
    r = len(C)
    beta_hat, _, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    C = np.array(C)

    cov_X = np.matmul(np.matmul(C, np.linalg.inv(np.matmul(X.transpose(), X))), C.transpose()) * unbiased_sigma_square_hat
    A = np.linalg.inv(cov_X) / (r * F_test.ppf(1 - alpha, r, (len(y) - k - 1)))
    eigvals = np.linalg.eigvals(A)

    return [eigvals, beta_hat]

def Mult_norm_LR_is_in_CR(X: list, y: list, C: list, c0: list, alpha: float) -> bool:
    """
        Checks whether given c0 vector is in the confidence region for C*Beta
        :param X: Design matrix
        :param y: Response vector
        :param C: Matrix containing linear combinations of the beta parameters
        :param c0: Vector containing values to check the linear combinations against.
        :param alpha: Significance level of the tests
        :return:  Returns a boolen stating whether c0 is in the confidence for C*Beta or not
    """

    check_mult_lr_errors(X, y)
    assert len(C) == len(c0), "C and c0 have different shapes"
    assert np.linalg.matrix_rank(C) == len(C), "Invalid rank of C matrix"

    beta_hat, _, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    k = len(X[0]) - 1
    r = len(C)
    C = np.array(C)

    # C(X^TX)^-1C^T
    center_term = np.linalg.inv(np.matmul(np.matmul(C, np.linalg.inv(np.matmul(X.transpose(), X))), C.transpose()))
    left_term   = (np.matmul(C, beta_hat) - c0).transpose()
    right_term  = (np.matmul(C, beta_hat) - c0)

    left_value  = np.matmul(np.matmul(left_term, center_term), right_term)
    right_value = r * unbiased_sigma_square_hat * F_test.ppf(1 - alpha, r, len(y) - k - 1)

    c0_is_in_region = False
    if left_value <= right_value:
        # ??
        c0_is_in_region = True

    return c0_is_in_region

def Mult_norm_LR_test_general(X: list, y: list, C: list, c0: list, alpha: float) -> bool:
    """
        Tests the null hypothesis H0: C*Beta = c0
        :param X: Design matrix
        :param y: Response vector
        :param C: Matrix containing linear combinations of the beta parameters
        :param c0: Vector containing values to check the linear combinations against.
        :param alpha: Significance level of the tests
        :return:  Returns true if the null hypothesis is rejected, false othwerwise
    """
    check_mult_lr_errors(X, y)
    assert len(C) == len(c0), "C and c0 have different shapes"
    assert np.linalg.matrix_rank(C) == len(C), "Invalid rank of C matrix"

    beta_hat, _, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    k = len(X[0]) - 1
    r = len(C)

    C = np.array(C)

    # C(X^TX)^-1C^T
    center_term = np.linalg.inv(np.matmul(np.matmul(C, np.linalg.inv(np.matmul(X.transpose(), X))), C.transpose()))
    left_term   = (np.matmul(C, beta_hat) - c0).transpose()
    right_term  = (np.matmul(C, beta_hat) - c0)

    left_value  = np.matmul(np.matmul(left_term, center_term), right_term)
    right_value = r * unbiased_sigma_square_hat * F_test.ppf(1 - alpha, r, len(y) - k - 1)

    rejection = False
    if left_value > right_value:
        # H0: CB = c0 rejected
        rejection = True

    return rejection

def Mult_norm_LR_test_comp(X: list, y: list, j_list: list, alpha: float) -> bool:
    """
        Tests the null hypothesis that some of the beta parameters are equal to 0.
        :param X: Design matrix
        :param y: Response vector
        :param j_list: List of parameter numbers e.g. [0, 1, 3]
        :param alpha: Significance level of the tests
        :return:  Returns true if the null hypothesis is rejected, false othwerwise
    """

    check_mult_lr_errors(X, y)
    assert len(j_list) <= len(X[0]), "Invalid number of J's"
    assert max(j_list) < len(X[0]), "Invalid index for j"
    assert min(j_list) >= 0, "Invalid index for j"

    # jr x k + 1 matrix
    C = np.zeros((len(j_list), len(X[0])))
    c0 = [0] * len(j_list)
    for i in range(len(j_list)):
        j = j_list[i]
        C[i][j] = 1

    return Mult_norm_LR_test_general(X, y, C, c0, alpha)

def Mult_norm_LR_test_linear_reg(X: list, y: list, alpha: float) -> bool:
    """
        Tests the null hypothesis whether there is a linear regression or not
        :param X: Design matrix
        :param y: Response vector
        :param alpha: Significance level of the tests
        :return:  Returns true if the null hypothesis is rejected, false othwerwise
    """
    j_list = []
    for i in range(1, len(X[0])):
        j_list.append(i)

    return Mult_norm_LR_test_comp(X, y, j_list, alpha)

def Mult_norm_LR_pred_CI(X: list, y: list, D: list, alpha: float, method: str) -> list:
    """
        Calculates simultaneous confidence bounds wrt given D matrix
        :param X: Design matrix
        :param y: Response vector
        :param D: Matrix containing multiple x0 vectors, new predictors
        :param alpha: Significance level of the tests
        :param method: Method to calculate the confidence bound. Either Scheffe, Bonferroni or Best
        :return:  Returns the confidence bound for x0*Beta
    """

    def scheffe() -> list:
        """
            Calculates the confidence bound using Scheffe's method
        """
        intervals = []
        for i in range(m):
            di = D[i]
            base = np.matmul(di, beta_hat)
            tail = np.sqrt((k + 1) * F_test.ppf(1 - alpha, k + 1, len(y) - k - 1)) * \
                    np.sqrt(unbiased_sigma_square_hat) * \
                    np.sqrt(np.matmul(np.matmul(di.transpose(), np.linalg.inv(np.matmul(X.transpose(), X))), di))

            interval = (base - tail, base + tail)
            intervals.append(interval)

        return intervals

    def bonferroni() -> list:
        """
            Calculates the confidence bound using Bonferroni's method
        """
        intervals = []

        for i in range(m):
            di = D[i]
            base = np.matmul(di, beta_hat)
            tail = t_test.ppf(1 - (alpha / (2 * m)), (len(y) - k - 1)) * \
                    np.sqrt(unbiased_sigma_square_hat) * \
                    np.sqrt(np.matmul(np.matmul(di.transpose(), np.linalg.inv(np.matmul(X.transpose(), X))), di))

            interval = (base - tail, base + tail)
            intervals.append(interval)

        return intervals

    check_mult_lr_errors(X, y)
    assert len(D[0]) == len(X[0]), "D and X must have same number of columns"
    assert method.lower() in ["bonferroni", "scheffe", "best"], "Invalid method name"

    beta_hat, _, unbiased_sigma_square_hat = Mult_LR_Least_squares(X, y)
    m = len(D)
    k = len(X[0]) - 1
    D = np.array(D)

    intervals = []
    if method.lower() == 'bonferroni':
        intervals = bonferroni()
    elif method.lower() == 'scheffe':
        intervals = scheffe()
    elif method.lower() == 'best':
        bonferroni_interval = bonferroni()
        scheffe_interval    = scheffe()

        interval_comparison = ANOVA_compare_confidence_intervals(bonferroni_interval, scheffe_interval)
        if interval_comparison < 0:
            intervals = bonferroni_interval
        else:
            intervals = scheffe_interval

    return intervals

###################################### LINER REGRESSION FUNCTIONS END #####################################