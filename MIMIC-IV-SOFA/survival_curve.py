import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter    
from lifelines.statistics import logrank_test


def survival_curve(data, time_label, event_label, group_label, group_1, group_2, group_1_name, group_2_name, x_label, y_label, xlim, ylim):

    kmf = KaplanMeierFitter()
    fit = kmf.fit(data[time_label], data[event_label])
    fit.median_survival_time_
    print(pd.concat([fit.event_table, fit.survival_function_], axis=1))

    g1 = data[group_label] == group_1  
    g2 = data[group_label] == group_2
    kmf_A = KaplanMeierFitter() 
    kmf_A.fit(data[time_label].loc[g1], data[event_label].loc[g1], label=group_1_name+"(N=%s)" % data[time_label].loc[g1].shape[0])
    kmf_B = KaplanMeierFitter()
    kmf_B.fit(data[time_label].loc[g2], data[event_label].loc[g2], label=group_2_name+"(N=%s)" % data[time_label].loc[g2].shape[0])

    fig, axes = plt.subplots()
    kmf_A.plot(ax=axes, show_censors=True)
    kmf_B.plot(ax=axes, show_censors=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title("Sruvival Curve")
    plt.xlim(xlim)
    plt.ylim(ylim )
    plt.show()

    lr = logrank_test(data[time_label].loc[g1],
                    data[time_label].loc[g2],
                    data[event_label].loc[g1], 
                    data[event_label].loc[g2])
    print(lr.p_value)