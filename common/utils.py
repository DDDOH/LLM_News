from scipy.stats import fisher_exact



# split test data into significant and insignificant
# for each news in the test data, run significance test
def significance_test_one_news(impressions, clicks, CTRs):
    """
    Run significance test for one news. We compare the headline with the highest CTR with the rest of the headlines. If all such pairs are significant, we return True, otherwise False.

    :param impressions: list of impressions for each headline
    :param clicks: list of clicks for each headline
    :param CTRs: list of CTRs for each headline
    :return: True if all pairs are significant, False otherwise
    """

    # find the headline with the highest CTR.
    highest_CTR_index = CTRs.index(max(CTRs))

    # run pairwise significance test between the headline with the highest CTR and the rest of the headlines
    for i in range(len(CTRs)):
        if i == highest_CTR_index:
            continue
        pair_clicks = [clicks[highest_CTR_index], clicks[i]]
        pair_non_clicks = [
            impressions[highest_CTR_index] - clicks[highest_CTR_index],
            impressions[i] - clicks[i],
        ]
        contingency_table = [pair_clicks, pair_non_clicks]
        odds_ratio, p_value = fisher_exact(contingency_table)
        if p_value < 0.05:  # this pair is significant
            continue
        else:
            return False
    return True
