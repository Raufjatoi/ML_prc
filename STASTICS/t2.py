from scipy import stats as s 
import numpy as n

g1 = n.array([80, 90, 100, 70, 85])
g2 = n.array([75, 85, 95, 65, 80])

t_stat, p_value = s.ttest_ind(g1, g2)

print(f'T-statistic: {t_stat:0f}')
print(f'P-value: {p_value:0f}')