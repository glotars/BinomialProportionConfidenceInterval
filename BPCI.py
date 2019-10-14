import scipy.stats as stats
import numpy as np
from collections import OrderedDict

class BPCI(object):
    """
    Binomial Proportion Confidence Interval:
        sample – sample to calculate confidence interval (e.g. [100, 200, 300]);
        alpha = (1 - confidence level);
    """
    def __init__(self, sample, alpha = 0.05):
        self.sample = np.asarray(sample)
        self.alpha = alpha
        self.shape = self.sample.shape[0]
        self.sum = self.sample.sum().astype(np.int32)
        self.percentage = self.sample / self.sum
    
    def wald(self):
        """
        Wald method (also known as "normal approximation interval"). This approximation
        is based on the central limit theorem and is unreliable when the 
        sample size is small or the success probability is close to 0 or 1.
        Returns:
            Method – Method name ('Wald');
            alpha – alpha value;
            Sample – Sample;
            N – Sum of sample;
            Percentage – Percentage of sample;
            Cl – Confidence lower limit;
            Cu – Confidence upper limit;
            Conclusion – for the confidence interval for each option, 
            independently of the others, we conclude that it is more plausible:
                1 – if this option is plausible;
                -1 – if negation of this option is plausible;
                0 – if it's not clear;
        """
        Z = abs(stats.norm.ppf(self.alpha / 2))
        yates_coef = 0.5 / self.sum
        delta = Z * abs(self.sample * (self.sum - self.sample) / (self.sum**self.shape))**(1/2) + yates_coef
        Cl = self.percentage - delta
        Cu = self.percentage + delta
        return BPCI_Result(Method = 'Wald',
                           alpha = self.alpha,
                           Sample = self.sample,
                           N = self.sum,
                           Percentage = self.percentage,
                           Cl = Cl, 
                           Cu = Cu)
    
    def clopper_pearson(self):
        """
        The Clopper–Pearson interval is an early and 
        very common method for calculating binomial confidence intervals.
        This is often called an 'exact' method, because it is based on 
        the cumulative probabilities of the binomial distribution 
        (i.e., exactly the correct distribution rather than an approximation). 
        However, in cases where we know the population size, the intervals 
        may not be the smallest possible, because they include impossible 
        proportions: for instance, for a population of size 10, an interval of [0.35, 0.65]
        would be too large as the true proportion cannot lie between 0.35 and 0.4, or 
        between 0.6 and 0.65.
        Returns:
            Method – Method name ('Clopper-Pearson');
            alpha – alpha value;
            Sample – Sample;
            N – Sum of sample;
            Percentage – Percentage of sample;
            Cl – Confidence lower limit;
            Cu – Confidence upper limit;
            Conclusion – for the confidence interval for each option, 
            independently of the others, we conclude that it is more plausible:
                1 – if this option is plausible;
                -1 – if negation of this option is plausible;
                0 – if it's not clear;
        """
        F_inv_rt_Cl = stats.f.isf(self.alpha/2, 
                            dfn = 2 * (1 + self.sum - self.sample), 
                            dfd = 2 * self.sample)
        F_inv_rt_Cu = stats.f.isf(self.alpha/2, 
                            dfn = 2 * (1 + self.sample), 
                            dfd = 2 * (self.sum - self.sample))
        Cl_vect = np.vectorize(BPCI.__clopper_pearson_Cl)
        Cu_vect = np.vectorize(BPCI.__clopper_pearson_Cu)
        Cl = Cl_vect(self.sample, self.sum, F_inv_rt_Cl)
        Cu = Cu_vect(self.sample, self.sum, F_inv_rt_Cu)
        return BPCI_Result(Method = 'Clopper-Pearson',
                           alpha = self.alpha,
                           Sample = self.sample,
                           N = self.sum,
                           Percentage = self.percentage,
                           Cl = Cl, 
                           Cu = Cu)
    
    @staticmethod    
    def __clopper_pearson_Cl(s, amount, f_inv_rt):
        if s == 0:
            return 0
        else:
            return s / (s + (1 + amount - s) * f_inv_rt)
    
    @staticmethod
    def __clopper_pearson_Cu(s, amount, f_inv_rt):
        if s == amount:
            return 1
        else:
            return (s + 1) * f_inv_rt / (amount - s + (s + 1) * f_inv_rt)

class BPCI_Result(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(BPCI_Result, self).__init__(*args, **kwargs)
        self.__values = list(super(BPCI_Result, self).values())[4:7]
        super(BPCI_Result, self).update(Conclusion = self.__conclusion())
    
    def __conclusion(self):
        return np.vectorize(BPCI_Result.__make_conclusion)(*self.__values)
    
    @staticmethod
    def __make_conclusion(p, cl, cu):
        if 0.5 > cl and 0.5 < cu:
            return 0
        return 1 if p >= 0.5 else -1
    
if __name__ == '__main__':
    """
    Lets make some tests!
    You've conducted a survey of shop visitors. The question is "How did prices change?".
    Here are the results:
        74122 votes – "Prices have increased";
        11098 votes – "Prices have not changed";
        85989 votes – "Prices have decreased".
    Now you need to decide which one is true with 95% confidence level.
    """
    
    shop_survey = BPCI([74122, 11098, 85989], alpha = 0.05)
    print(shop_survey.wald())
    
    """
    BPCI_Result([('Method', 'Wald'),
             ('alpha', 0.05),
             ('Sample', array([74122, 11098, 85989])),
             ('N', 171209),
             ('Percentage', array([0.43293285, 0.06482136, 0.50224579])),
             ('Cl', array([0.43058293, 0.06365219, 0.4998745 ])),
             ('Cu', array([0.43528276, 0.06599053, 0.50461709])),
             ('Conclusion', array([-1, -1,  0]))])
    """
    
    print(shop_survey.clopper_pearson())
    
    """
    BPCI_Result([('Method', 'Clopper-Pearson'),
             ('alpha', 0.05),
             ('Sample', array([74122, 11098, 85989])),
             ('N', 171209),
             ('Percentage', array([0.43293285, 0.06482136, 0.50224579])),
             ('Cl', array([0.43058408, 0.06365957, 0.49987448])),
             ('Cu', array([0.43528388, 0.06599791, 0.50461703])),
             ('Conclusion', array([-1, -1,  0]))])
             
    As you can see, only the last option ("Prices have decreased")
    is more likely to be true in both cases.
    """
